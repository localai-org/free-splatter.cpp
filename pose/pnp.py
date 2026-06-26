"""PnP camera recovery from the engine's per-pixel gaussians.

Mirrors FreeSplatter's estimate_poses (freesplatter/models/model.py, which calls
the DUSt3R-derived fast_pnp in utils/recon_util.py):

  * correspondences : every pixel -> its predicted 3D gaussian center X (channels
    0:3) <-> its 2D pixel coordinate.
  * validity mask   : opacity > 0.05 (drops floaters before PnP).
  * focal           : Weiszfeld per view, averaged into one shared focal (focal.py).
  * pose            : cv2.solvePnPRansac(reprojectionError=5, SOLVEPNP_SQPNP,
                      iterationsCount=10); world2cam, then cam2world = inv(.);
                      OpenCV convention; view 0 comes out ~identity by construction
                      (gaussians are predicted in view 0's frame), no re-anchor.
  * scene scale     : upstream's `estimate_poses` does NOT normalize; the runner
                      (run_freesplatter_scene) rescales camera centers so the
                      view0->view1 baseline is 1.0 (s = 1/(||t1-t0||+1e-2)). See
                      `rescale_to_baseline`. (An earlier 1/mean||pts|| guess was
                      wrong -- verified against .cache/upstream/{model,runner}.py.)

Two solver backends, selected by `backend`:
  - "cv2"  : the exact upstream call (needs opencv; available in the CUDA
             container or after adding opencv to the nix shell). Use this for
             numerical parity against the original.
  - "numpy": a dependency-free DLT + RANSAC reference solver, so the pipeline and
             its golden tests run anywhere numpy is present. Verified against
             analytic ground truth in test_pose.py.
  - "auto" : cv2 if importable, else numpy.

Gaussian channel layout (scene, 23 ch): xyz[0:3] SH[3:15] opacity[15] scale[16:19]
rotation[19:23]; opacity is assumed already activated (sigmoid) in [0,1].
"""
import numpy as np

OPACITY_CHANNEL = 15


# --- numpy reference PnP (known intrinsics) --------------------------------

def _dlt(Xw, xn):
    """Direct Linear Transform for world2cam [R|t] from normalized image coords.

    Xw : (N,3) world points. xn : (N,2) normalized coords ( K^-1 [u v 1] ).
    Returns the raw 3x4 matrix p (defined up to sign/scale); decode with _decode.
    """
    N = len(Xw)
    A = np.zeros((2 * N, 12))
    Xh = np.hstack([Xw, np.ones((N, 1))])         # (N,4)
    A[0::2, 0:4] = Xh
    A[0::2, 8:12] = -xn[:, 0:1] * Xh
    A[1::2, 4:8] = Xh
    A[1::2, 8:12] = -xn[:, 1:2] * Xh
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 4)


def _decode(p, Xw, xn):
    """Turn a DLT matrix into a proper (R,t) world2cam, resolving sign by
    reprojection error and cheirality (points must sit in front of the camera)."""
    best = None
    for sgn in (1.0, -1.0):
        M = sgn * p[:, :3]
        tau = sgn * p[:, 3]
        U, Sv, Vt = np.linalg.svd(M)
        d = np.linalg.det(U @ Vt)
        R = U @ np.diag([1, 1, d]) @ Vt           # nearest proper rotation
        lam = Sv.mean()                            # M ~= lam * R
        if lam < 1e-12:
            continue
        t = tau / lam
        Xc = (R @ Xw.T).T + t
        if np.any(Xc[:, 2] <= 0):
            err = np.inf
        else:
            proj = Xc[:, :2] / Xc[:, 2:3]
            err = float(np.sqrt(((proj - xn) ** 2).sum(1)).mean())
        if best is None or err < best[0]:
            best = (err, R, t)
    if best is None:
        raise np.linalg.LinAlgError("PnP decode failed (all-behind camera)")
    return best[1], best[2], best[0]


def solve_pnp_numpy(Xw, pixels, K, thresh_px=5.0, iters=100, seed=0):
    """RANSAC DLT PnP. Returns (cam2world 4x4, inlier_mask)."""
    Kinv = np.linalg.inv(K)
    uv1 = np.hstack([pixels, np.ones((len(pixels), 1))])
    xn = (Kinv @ uv1.T).T[:, :2]                  # normalized image coords
    thresh = thresh_px / K[0, 0]                   # px tol -> normalized tol

    rng = np.random.default_rng(seed)
    n = len(Xw)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 6, replace=False)
        try:
            R, t, _ = _decode(_dlt(Xw[idx], xn[idx]), Xw[idx], xn[idx])
        except np.linalg.LinAlgError:
            continue
        Xc = (R @ Xw.T).T + t
        good = Xc[:, 2] > 0
        proj = np.full((n, 2), 1e9)
        proj[good] = Xc[good, :2] / Xc[good, 2:3]
        inl = np.linalg.norm(proj - xn, axis=1) < thresh
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 6:
        best = np.ones(n, bool)
    R, t, _ = _decode(_dlt(Xw[best], xn[best]), Xw[best], xn[best])
    world2cam = np.eye(4)
    world2cam[:3, :3] = R
    world2cam[:3, 3] = t
    return np.linalg.inv(world2cam), best


def solve_pnp_cv2(Xw, pixels, K, iters=20, reproj=5.0):
    """Exact upstream call (needs cv2). Returns (cam2world, inlier_mask)."""
    import cv2
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        Xw.astype(np.float64), pixels.astype(np.float64), K.astype(np.float64),
        None, iterationsCount=iters, reprojectionError=reproj,
        flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("cv2.solvePnPRansac failed")
    R, _ = cv2.Rodrigues(rvec)
    world2cam = np.eye(4)
    world2cam[:3, :3] = R
    world2cam[:3, 3] = tvec[:, 0]
    mask = np.zeros(len(Xw), bool)
    if inliers is not None:
        mask[inliers[:, 0]] = True
    return np.linalg.inv(world2cam), mask


def _have_cv2():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def solve_pnp(Xw, pixels, K, backend="auto", **kw):
    if backend == "auto":
        backend = "cv2" if _have_cv2() else "numpy"
    if backend == "cv2":
        return solve_pnp_cv2(Xw, pixels, K, **kw)
    return solve_pnp_numpy(Xw, pixels, K, **kw)


# --- orchestration ---------------------------------------------------------

def pixel_grid(H, W):
    """(H*W, 2) INTEGER pixel coords [x=col, y=row], row-major -- matches upstream
    xy_grid / np.mgrid[:W,:H].T exactly (NO half-pixel offset). Row-major so it
    lines up with a C-order (H,W,3) gaussian map flattened by reshape(-1,3)."""
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))   # both (H,W): xs=col, ys=row
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)


def rescale_to_baseline(cam2world):
    """Scene normalization from runner.run_freesplatter_scene: rescale camera
    centers so the view0->view(last) baseline is 1.0. Returns (list, scale_factor).

    Mirrors upstream exactly: baseline = ||t_last - t_0|| + 1e-2; s = 1/baseline;
    every camera translation *= s (rotations untouched)."""
    t0 = cam2world[0][:3, 3]
    baseline = float(np.linalg.norm(cam2world[-1][:3, 3] - t0)) + 1e-2
    s = 1.0 / baseline
    out = []
    for c in cam2world:
        c = c.copy()
        c[:3, 3] *= s
        out.append(c)
    return out, s


def estimate_poses(points, opacities=None, focal=None, pp=None,
                   opacity_threshold=0.05, backend="auto", cam_frame_points=None,
                   use_first_focal=True, focal_use_opacity_mask=False,
                   pnp_iter=10, normalize=False):
    """Faithful port of FreeSplatter's scene estimate_poses (.cache/upstream/model.py)
    plus the runner's optional baseline rescale.

    Defaults mirror the scene recipe: focal from view 0 over ALL pixels
    (use_first_focal=True, masks=None), per-view PnP masked by opacity>thr,
    pnp_iter=10, cam2world=inv(world2cam). estimate_poses returns the RAW poses
    (view 0 comes out ~identity because the gaussians live in view 0's frame -- no
    explicit re-anchor, matching upstream). Set normalize=True for the runner's
    1/baseline camera rescale (the scene viewer normalization).

    points          : list of (H,W,3) gaussian-center maps (view 0 frame).
    opacities       : list of (H,W) activated opacities for the PnP mask (or None).
    cam_frame_points: per-view points in each view's OWN frame for focal. With
                      use_first_focal only view 0 is used, and view 0's frame IS the
                      world frame, so points[0] suffices; pass None to use `points`.
    focal_use_opacity_mask: scene=False (all pixels, faithful); object would pass an
                      alpha mask -- approximated here by the opacity mask if True.
    Returns dict: cam2world (list of 4x4), focal, scale.
    """
    from focal import estimate_focal

    backend_cv2 = backend == "cv2" or (backend == "auto" and _have_cv2())

    N = len(points)
    H, W = points[0].shape[:2]
    if pp is None:
        pp = np.array([W / 2.0, H / 2.0])
    grid = pixel_grid(H, W)

    masks = []
    for i in range(N):
        if opacities is None:
            masks.append(np.ones(H * W, bool))
        else:
            masks.append(opacities[i].ravel() > opacity_threshold)

    # focal: mirror estimate_focals -- view 0 only (use_first_focal), all pixels
    # unless an opacity mask is explicitly requested.
    if focal is None:
        cfp = cam_frame_points if cam_frame_points is not None else points
        n_focal = 1 if use_first_focal else N
        fs = []
        for i in range(n_focal):
            m = masks[i] if (focal_use_opacity_mask and opacities is not None) \
                else np.ones(H * W, bool)
            fs.append(estimate_focal(cfp[i].reshape(-1, 3)[m], grid[m], pp))
        focal = float(np.mean(fs))

    K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1.0]])

    cam2world = []
    for i in range(N):
        Xw = points[i].reshape(-1, 3)[masks[i]]
        px = grid[masks[i]]
        if backend_cv2:
            c2w, _ = solve_pnp_cv2(Xw, px, K, iters=pnp_iter, reproj=5.0)
        else:
            c2w, _ = solve_pnp_numpy(Xw, px, K)
        cam2world.append(c2w)

    scale = 1.0
    if normalize:
        cam2world, scale = rescale_to_baseline(cam2world)

    return {"cam2world": cam2world, "focal": focal, "scale": scale}
