"""Coordinate-system normalization: align one inference's world into another's.

DOWNSTREAM of the engine seam. Each free_splatter_run over a different photo
pair invents its OWN coordinate system (gaussians in the first view's camera
frame, then rescaled by s = 1/mean-point-distance -- see FreeSplatter
estimate_poses). Two runs that share a photo therefore describe the SAME surface
at a DIFFERENT scale, because each run's 1/d normalization is computed over a
different point support. So aligning run B into run A is a SIMILARITY transform
(rotation + translation + ONE uniform scale, 7 DoF), not a rigid one.

This module fits that similarity in closed form (Umeyama 1991), provides a
RANSAC variant for floaters, and -- crucially -- a residual ladder that *detects*
whether a single uniform scale actually suffices or whether the mismatch is
non-uniform (focal error / affine depth ambiguity), which is the open question.

Pure numpy. A similarity is represented as the tuple T = (s, R, t) acting on
column-stacked points by  x -> s * R @ x + t.
"""
import numpy as np


def apply_sim(T, X):
    """Apply T=(s,R,t) to X (N,3) -> (N,3)."""
    s, R, t = T
    return (s * (R @ X.T)).T + t


def _rms(resid):
    """Root-mean-square point error of an (N,3) residual array."""
    return float(np.sqrt((resid ** 2).sum(axis=1).mean()))


def fit_similarity(X, Y, with_scale=True):
    """Umeyama: best (s,R,t) minimizing ||s R X + t - Y||^2 in closed form.

    with_scale=False degenerates to rigid Kabsch (s fixed at 1). Reflections are
    rejected (proper rotation, det R = +1) exactly as Umeyama prescribes.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = X.shape[0]
    mx = X.mean(0)
    my = Y.mean(0)
    Xc = X - mx
    Yc = Y - my
    Sigma = (Yc.T @ Xc) / n               # 3x3 cross-covariance, maps X -> Y
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:   # reflection guard
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_x = (Xc ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / var_x) if var_x > 0 else 1.0
    else:
        s = 1.0
    t = my - s * (R @ mx)
    return s, R, t


def fit_rigid(X, Y):
    """Rigid (6-DoF) fit -- similarity with the scale DoF removed."""
    return fit_similarity(X, Y, with_scale=False)


def fit_affine(X, Y):
    """Full affine (12-DoF) fit: Y ~= A X + b. Returns ((A,b), prediction).

    Used only as the next rung of the residual ladder: how much of the leftover
    error after a similarity is explained by *non-uniform* (shear/anisotropic)
    deformation -- the signature of focal error or affine-invariant depth.
    """
    n = X.shape[0]
    Xh = np.hstack([X, np.ones((n, 1))])          # (n,4)
    M, *_ = np.linalg.lstsq(Xh, Y, rcond=None)    # (4,3)
    pred = Xh @ M
    A = M[:3].T
    b = M[3]
    return (A, b), pred


def fit_similarity_ransac(X, Y, thresh, iters=300, with_scale=True, seed=0):
    """RANSAC similarity: robust to gross outliers (floaters in the overlap).

    Minimal sample is 3 point pairs (a similarity has 7 DoF; 3 pairs give 9
    constraints). Returns (s,R,t, inlier_mask).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            T = fit_similarity(X[idx], Y[idx], with_scale)
        except np.linalg.LinAlgError:
            continue
        res = np.linalg.norm(apply_sim(T, X) - Y, axis=1)
        inl = res < thresh
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:                            # fall back to plain fit
        best = np.ones(n, bool)
    T = fit_similarity(X[best], Y[best], with_scale)
    return (*T, best)


def residual_ladder(X, Y):
    """Fit rigid -> similarity -> affine and report each RMS residual.

    The DROP from rigid->similarity is how much was pure uniform scale; the drop
    from similarity->affine is how much non-uniform warp remains. depth_corr is
    the Pearson correlation between the per-point similarity residual magnitude
    and the point's distance from the centroid -- a positive value means the
    leftover error GROWS with depth, the fingerprint of focal/projective error.
    All residuals are absolute (same units as Y); compare against `scene`.
    """
    scene = _rms(Y - Y.mean(0))                   # overall extent of the target

    Tr = fit_rigid(X, Y)
    r_rigid = _rms(apply_sim(Tr, X) - Y)

    Ts = fit_similarity(X, Y, with_scale=True)
    res_sim = apply_sim(Ts, X) - Y
    r_sim = _rms(res_sim)

    _, pred_aff = fit_affine(X, Y)
    r_aff = _rms(pred_aff - Y)

    depth = np.linalg.norm(X - X.mean(0), axis=1)
    rmag = np.linalg.norm(res_sim, axis=1)
    depth_corr = float(np.corrcoef(depth, rmag)[0, 1]) if rmag.std() > 1e-12 else 0.0

    return {
        "scene": scene,
        "rigid": r_rigid,
        "similarity": r_sim,
        "affine": r_aff,
        "scale": float(Ts[0]),
        "depth_corr": depth_corr,
    }


def diagnose(X, Y, tol=1e-3, corr_tol=0.3):
    """Classify the A<->B mismatch from the residual ladder.

    Residuals are judged relative to scene extent. Verdicts:
      rigid_ok    : a rigid transform already fits -- same scale (rare across runs)
      needs_scale : a SIMILARITY fits but rigid does not -- the expected case
      needs_affine: even similarity fails but affine fits -- non-uniform warp
      nonlinear   : affine fails too -- genuinely nonlinear (e.g. quadratic depth)
    `depth_corr` is reported as corroborating evidence of structure.
    """
    L = residual_ladder(X, Y)
    sc = L["scene"] or 1.0
    rr, rs, ra = L["rigid"] / sc, L["similarity"] / sc, L["affine"] / sc
    # Is the leftover after a similarity STRUCTURED (a real warp) or just noise?
    # Two independent signatures: affine meaningfully beats similarity, OR the
    # residual correlates with depth. Neither => unstructured noise floor.
    aff_gain = (L["similarity"] - L["affine"]) / L["similarity"] if L["similarity"] > 0 else 0.0
    structured = abs(L["depth_corr"]) > corr_tol or aff_gain > 0.25
    if rr < tol:
        v = "rigid_ok"                       # same scale AND pose: rare across runs
    elif rs < tol:
        v = "needs_scale"                    # a clean similarity fits: the expected case
    elif ra < tol:
        v = "needs_affine"                   # similarity fails, affine fits: non-uniform
    elif structured:
        v = "nonlinear"                      # even affine fails AND residual is structured
    else:
        v = "similarity_plus_noise"          # similarity is the model; rest is noise floor
    L["verdict"] = v
    L["structured"] = bool(structured)
    L["aff_gain"] = float(aff_gain)
    return L


# --- chaining similarities along the photo stream --------------------------

def compose(T2, T1):
    """Return T2 . T1 (apply T1 then T2)."""
    s1, R1, t1 = T1
    s2, R2, t2 = T2
    return (s2 * s1, R2 @ R1, s2 * (R2 @ t1) + t2)


def invert(T):
    """Inverse similarity."""
    s, R, t = T
    si = 1.0 / s
    Ri = R.T
    return (si, Ri, -si * (Ri @ t))


def identity():
    return (1.0, np.eye(3), np.zeros(3))


# --- Sim(3) as 4x4 matrices (for pose-graph relaxation) --------------------

def sim_matrix(s, R, t):
    """Sim(3) as a 4x4 homogeneous matrix [[sR, t],[0,1]] (compose == matmul)."""
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def sim_frac_power(M, f):
    """Fractional power M^f of a Sim(3) 4x4, via eigendecomposition (real part).

    The even loop-closure relaxation: distribute an accumulated drift D over n
    nodes by applying D^(k/n) at node k. M^0=I, M^1=M, (M^0.5)^2=M; valid while the
    rotation is < 180deg (principal branch). Tested in test_pose.py."""
    w, V = np.linalg.eig(M)
    return (V @ np.diag(w ** f) @ np.linalg.inv(V)).real


def loop_closure_error(transforms):
    """Compose a closed loop of similarities and measure deviation from identity.

    transforms: list [T_{0->1}, T_{1->2}, ..., T_{k->0}]. With drift-free links
    the product is identity; the returned dict quantifies residual scale, rotation
    (deg) and translation drift -- a direct scale-drift / loop-closure metric.
    """
    T = identity()
    for Ti in transforms:
        T = compose(Ti, T)
    s, R, t = T
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return {
        "scale_err": abs(np.log(s)),          # 0 == no scale drift
        "rot_deg": float(ang),
        "trans": float(np.linalg.norm(t)),
    }
