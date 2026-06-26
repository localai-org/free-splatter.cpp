"""Verify our WHOLE pose orchestration against upstream, on REAL engine output.

`check_cv2_parity.py` already pinned our solver wrapper to cv2.solvePnPRansac on
synthetic ground truth. This goes one level up: it drives the *entire* upstream
`estimate_poses` recipe (FreeSplatter scene path: use_first_focal, opacity mask
for PnP, no focal mask, pnp_iter=10 -- see .cache/upstream/{model,runner}.py)
using the vendored-verbatim upstream kernels in `_upstream.py`, and compares it
to our `focal.py`/`pnp.py` on the gaussians the C++ engine actually produced.

Three levels, each isolating one stage so a divergence points at one file:
  1. focal kernel   -- focal.estimate_focal           vs _upstream.estimate_focal_np
  2. PnP solver     -- pnp.solve_pnp_cv2              vs _upstream.fast_pnp   (shared focal)
  3. full pipeline  -- pnp.estimate_poses             vs the upstream recipe replica
Plus the scene baseline-rescale (runner.run_freesplatter_scene) for reference.

    nix develop -c python3 pose/check_upstream_parity.py A_scn.f32   # scene (live case)

Defaults to the scratchpad dumps if no path is given. Opacity in the engine
output is ALREADY sigmoid-activated, so mask = (opacity > thr) reproduces
upstream's sigmoid(raw) > thr exactly.
"""
import os
import sys
import numpy as np
import cv2

import focal
import pnp
import _upstream as up

C, OPACITY = 23, 15
THR = 0.05
SEED = 0           # cv2.solvePnPRansac draws from the shared global cv::theRNG();
                   # reseed identically before each call so RANSAC picks the SAME
                   # minimal samples -> isolates algorithmic parity from RNG state.
SCRATCH = os.environ.get("EMP_DIR", "").rstrip("/")
_fails = []


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def load(path, H=512, W=512):
    a = np.fromfile(path, np.float32)
    N = a.size // (H * W * C)
    a = a.reshape(N, H, W, C)
    pts = [a[i, ..., 0:3].astype(np.float64) for i in range(N)]      # (H,W,3) each
    op = [a[i, ..., OPACITY].astype(np.float64) for i in range(N)]   # (H,W) each
    return N, H, W, pts, op


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"   ({detail})" if detail else ""))
    if not cond:
        _fails.append(name)


def upstream_estimate_poses(pts, op, H, W, pnp_iter=10):
    """Replicate FreeSplatter scene estimate_poses with vendored kernels.

    use_first_focal=True, masks=None: focal from view0 over ALL pixels; per-view
    PnP masked by opacity>THR. Returns (c2ws list, focal)."""
    pp = np.array([W / 2.0, H / 2.0])
    f = up.estimate_focal_np(pts[0], pp=pp, mask=None)              # view0, all pixels
    c2ws = []
    cv2.setRNGSeed(SEED)
    for i in range(len(pts)):
        mask = op[i] > THR                                          # (H,W) bool
        res = up.fast_pnp(pts[i], mask, focal=f, niter_PnP=pnp_iter,
                          k_dtype=np.float64)   # matched precision (our wrapper is f64)
        c2ws.append(res[1])
    return c2ws, f


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRATCH, "A_scn.f32")
    if not os.path.isabs(path) and SCRATCH and not os.path.exists(path):
        path = os.path.join(SCRATCH, path)
    N, H, W, pts, op = load(path)
    pp = np.array([W / 2.0, H / 2.0])
    print(f"=== upstream-parity on {os.path.basename(path)}  ({N} views) ===")
    for i in range(N):
        print(f"  view{i}: valid(opacity>{THR}) = {(op[i] > THR).mean()*100:.1f}%")

    # ---- Level 1: focal kernel (view0, all pixels) ----
    print("\n[1] focal kernel  focal.estimate_focal  vs  upstream.estimate_focal_np")
    grid = pnp.pixel_grid(H, W)                                     # our pixel convention
    f_up = up.estimate_focal_np(pts[0], pp=pp, mask=None)
    f_ours = focal.estimate_focal(pts[0].reshape(-1, 3), grid, pp)
    rel = abs(f_ours - f_up) / f_up
    check("focal matches upstream (all pixels, view0)", rel < 1e-6,
          f"ours {f_ours:.4f} vs up {f_up:.4f}  rel {rel:.2e}")

    # ---- Level 2: PnP solver, shared upstream focal ----
    print("\n[2] PnP solver  pnp.solve_pnp_cv2  vs  upstream.fast_pnp  (focal fixed)")
    K = np.array([[f_up, 0, pp[0]], [0, f_up, pp[1]], [0, 0, 1.0]])
    for i in range(N):
        mask = op[i] > THR
        Xw = pts[i].reshape(-1, 3)[mask.ravel()]
        px = grid[mask.ravel()]
        cv2.setRNGSeed(SEED)
        c2w_up = up.fast_pnp(pts[i], mask, focal=f_up, niter_PnP=10,
                             k_dtype=np.float64)[1]      # matched precision
        cv2.setRNGSeed(SEED)
        c2w_ours, _ = pnp.solve_pnp_cv2(Xw, px, K, iters=10, reproj=5.0)
        dR = rot_angle(c2w_ours[:3, :3] @ c2w_up[:3, :3].T)
        dt = float(np.linalg.norm(c2w_ours[:3, 3] - c2w_up[:3, 3]))
        check(f"view{i} cam2world matches upstream", dR < 1e-3 and dt < 1e-4,
              f"dR {dR:.2e} deg  dt {dt:.2e}")
        # diagnostic: upstream's native float32 K (precision-only effect)
        cv2.setRNGSeed(SEED)
        c2w_f32 = up.fast_pnp(pts[i], mask, focal=f_up, niter_PnP=10)[1]
        dR32 = rot_angle(c2w_ours[:3, :3] @ c2w_f32[:3, :3].T)
        print(f"        (upstream native float32 K: dR {dR32:.2e} deg -- "
              f"RANSAC inlier-boundary precision, not orchestration)")

    # ---- Level 3: full pipeline ----
    print("\n[3] full pipeline  pnp.estimate_poses  vs  upstream recipe")
    up_c2ws, f_up2 = upstream_estimate_poses(pts, op, H, W, pnp_iter=10)
    cv2.setRNGSeed(SEED)
    out = pnp.estimate_poses(pts, op, opacity_threshold=THR, backend="cv2",
                             normalize=False, cam_frame_points=pts)
    our_c2ws, f_our2 = out["cam2world"], out["focal"]
    relf = abs(f_our2 - f_up2) / f_up2
    check("pipeline focal matches", relf < 1e-6, f"ours {f_our2:.4f} vs up {f_up2:.4f}")
    # anchor-invariant relative pose view0->view1 (handles any re-anchor convention)
    def rel_pose(c):
        return np.linalg.inv(c[0]) @ c[1]
    Rup, Rours = rel_pose(up_c2ws), rel_pose(our_c2ws)
    dR = rot_angle(Rours[:3, :3] @ Rup[:3, :3].T)
    # compare translation DIRECTION (scale/anchor independent)
    tup, tour = Rup[:3, 3], Rours[:3, 3]
    cos_t = float(tup @ tour / (np.linalg.norm(tup) * np.linalg.norm(tour) + 1e-12))
    check("relative-pose rotation matches", dR < 1e-2, f"dR {dR:.2e} deg")
    check("relative-pose translation direction matches", cos_t > 1 - 1e-6,
          f"cos {cos_t:.8f}")

    # ---- scene baseline rescale (runner), reference only ----
    baseline = float(np.linalg.norm(up_c2ws[1][:3, 3] - up_c2ws[0][:3, 3])) + 1e-2
    print(f"\n[ref] scene baseline rescale: 1/baseline = {1.0/baseline:.4f} "
          f"(runner applies this to camera centers; our estimate_poses returns raw "
          f"poses, normalize=False)")

    print()
    if _fails:
        print(f"UPSTREAM PARITY FAILED ({len(_fails)}): " + ", ".join(_fails))
        raise SystemExit(1)
    print("UPSTREAM PARITY OK")


if __name__ == "__main__":
    main()
