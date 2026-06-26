"""Vendored upstream reference for the parity check -- NOT our code.

These are the two numerical kernels FreeSplatter's `estimate_poses` calls, copied
from TencentARC/FreeSplatter `freesplatter/utils/recon_util.py`
(https://raw.githubusercontent.com/TencentARC/FreeSplatter/main/freesplatter/utils/recon_util.py,
fetched 2026-06-26). They exist here ONLY so `check_upstream_parity.py` can run
the original algorithm against our `focal.py`/`pnp.py` on real engine output.

  * `fast_pnp`  -- copied VERBATIM (it is cv2 + numpy only; runs as-is in nix).
  * `estimate_focal_np` -- a faithful numpy transcription of upstream
    `estimate_focal` (the original is torch; the math is identical -- integer
    `xy_grid` pixels, nan_to_num, mean/mean init, 10 IRLS reweights, the
    `focal_base` clip which is a no-op at the default min/max). Kept line-by-line
    so any divergence is a real algorithmic difference, not a re-derivation.

Do not "improve" or refactor this file; its value is being unmodified upstream.
"""
import cv2
import numpy as np


def xy_grid(W, H):
    """Upstream xy_grid (numpy branch): (H,W,2) int, [...,0]=col(x), [...,1]=row(y)."""
    tw, th = np.arange(W), np.arange(H)
    gx, gy = np.meshgrid(tw, th, indexing="xy")     # both (H,W)
    return np.stack([gx, gy], axis=-1)


def estimate_focal_np(pts3d, pp=None, mask=None, min_focal=0.0, max_focal=np.inf):
    """Faithful numpy transcription of upstream recon_util.estimate_focal."""
    H, W, THREE = pts3d.shape
    assert THREE == 3
    if pp is None:
        pp = np.array([W / 2.0, H / 2.0], dtype=float)
    pp = np.asarray(pp, float)

    pixels = xy_grid(W, H).reshape(-1, 2).astype(float) - pp.reshape(1, 2)   # (HW,2)
    pts = pts3d.reshape(H * W, 3).astype(float)

    if mask is not None:
        m = np.asarray(mask).ravel().astype(bool)
        assert len(m) == pts.shape[0]
        pts = pts[m]
        pixels = pixels[m]

    xy_over_z = pts[:, :2] / pts[:, 2:3]
    xy_over_z = np.nan_to_num(xy_over_z, posinf=0.0, neginf=0.0)

    dot_xy_px = (xy_over_z * pixels).sum(axis=-1)
    dot_xy_xy = (xy_over_z ** 2).sum(axis=-1)
    focal = dot_xy_px.mean() / dot_xy_xy.mean()

    for _ in range(10):
        dis = np.linalg.norm(pixels - focal * xy_over_z, axis=-1)
        w = 1.0 / np.clip(dis, 1e-8, None)
        focal = (w * dot_xy_px).mean() / (w * dot_xy_xy).mean()

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))
    focal = float(np.clip(focal, min_focal * focal_base, max_focal * focal_base))
    return focal


def fast_pnp(pts3d, mask, focal=None, pp=None, niter_PnP=10, k_dtype=np.float32):
    """VERBATIM copy of upstream recon_util.fast_pnp (cv2 + numpy).

    The default `k_dtype=np.float32` reproduces upstream byte-for-byte (upstream
    writes `K = np.float32(...)`). `check_upstream_parity.py` passes float64 to
    isolate the one precision choice that differs from our wrapper: on near-
    threshold-dense object data the float32 K nudges a few hundred points across
    the 5px inlier boundary, shifting the pose ~0.5deg. Matched precision -> exact."""
    H, W, _ = pts3d.shape
    pixels = np.mgrid[:W, :H].T.astype(float)

    if focal is None:
        S = max(W, H)
        tentative_focals = np.geomspace(S / 2, S * 3, 21)
    else:
        tentative_focals = [focal]

    if pp is None:
        pp = (W / 2, H / 2)

    best = 0,
    for focal in tentative_focals:
        K = np.array([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)], dtype=k_dtype)

        success, R, T, inliers = cv2.solvePnPRansac(
            pts3d[mask], pixels[mask], K, None,
            iterationsCount=niter_PnP, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        if not success:
            continue

        score = len(inliers)
        if success and score > best[0]:
            best = score, R, T, focal

    if not best[0]:
        return None

    _, R, T, best_focal = best
    R = cv2.Rodrigues(R)[0]      # world to cam
    world2cam = np.eye(4).astype(float)
    world2cam[:3, :3] = R
    world2cam[:3, 3] = T.reshape(3)
    cam2world = np.linalg.inv(world2cam)

    return best_focal, cam2world
