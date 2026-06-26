"""Shared focal-length estimation (Weiszfeld), faithful to FreeSplatter.

Upstream (freesplatter/models/model.py::estimate_focal, via DUSt3R) assumes a
single shared focal across views, centered principal point, square pixels. Per
view it solves

    f* = argmin_f  sum_ij || (i',j') - f * (X_ij0/X_ij2, X_ij1/X_ij2) ||

where (i',j') are pixel coords relative to the principal point and X is that
view's camera-frame point map (gaussian xyz). The robust minimizer is found by
the Weiszfeld algorithm (iteratively reweighted least squares, ~L1). Per-view
focals are then averaged into one shared value.

Pure numpy -- no cv2 needed for this stage.
"""
import numpy as np


def estimate_focal(pts3d_cam, pixels, pp, weiszfeld_iters=10):
    """Weiszfeld focal from camera-frame points and their pixels.

    pts3d_cam : (...,3) points in the view's OWN camera frame (Z>0 in front).
    pixels    : (...,2) pixel coordinates, same leading shape.
    pp        : (2,) principal point (image center).
    """
    X = np.asarray(pts3d_cam, float).reshape(-1, 3)
    P = np.asarray(pixels, float).reshape(-1, 2) - np.asarray(pp, float)
    # ray directions; Z==0 -> 0 exactly as upstream estimate_focal (nan_to_num)
    u = np.nan_to_num(X[:, :2] / X[:, 2:3], posinf=0.0, neginf=0.0)   # (N,2)

    pu = (P * u).sum(1)                            # per-point p . u
    uu = (u * u).sum(1)                            # per-point ||u||^2
    f = pu.sum() / uu.sum()                        # least-squares init
    for _ in range(weiszfeld_iters):
        r = np.linalg.norm(P - f * u, axis=1)
        w = 1.0 / np.maximum(r, 1e-8)
        f = (w * pu).sum() / (w * uu).sum()
    return float(f)


def estimate_shared_focal(views, pp, weiszfeld_iters=10):
    """Average per-view focals into one shared focal.

    views: list of (pts3d_cam, pixels) pairs. Mirrors upstream's
    `focals = focals.mean().repeat(N)`.
    """
    fs = [estimate_focal(p3, px, pp, weiszfeld_iters) for (p3, px) in views]
    return float(np.mean(fs)), fs
