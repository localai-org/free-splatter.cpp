# pose/ — camera recovery + cross-run alignment (downstream prototype)

**Status: TEMPORARY Python prototype — to be rewritten in C++ once proven, then
deleted.**

PnP pose recovery and cross-run alignment are now **in scope** (see CLAUDE.md).
The shipped implementation will be **C++**, reachable from `free_splatter-cli` and
the C API with **no Python dependency**. This directory is the throwaway research
prototype that proves the accumulating-reconstruction approach first: a
self-contained numpy + cv2 consumer of the engine's `[N, H, W, 23]` output,
**not wired into the CMake build or `ctest`**. Once the approach is proven it is
ported to C++ and this Python is removed — it is not a parallel implementation to
maintain.

## Why it's needed

Each `free_splatter_run` over a different photo pair produces gaussians in its
**own** coordinate system: positions are predicted in the first view's camera
frame, then the whole scene is rescaled by `s = 1/mean(‖point‖)` (see upstream
`estimate_poses`). To stitch successive splats into one world you need:

1. **PnP** — recover each view's camera from the per-pixel 2D↔3D correspondences
   the gaussians hand you for free.
2. **Cross-run alignment** — because two runs normalize by a *different* `1/d`,
   the shared content comes out at a different **scale**. Aligning run B into run
   A is therefore a **similarity** (rotation + translation + one uniform scale,
   7-DoF), not a rigid transform.

## Upstream recipe being mirrored (TencentARC/FreeSplatter)

`freesplatter/models/model.py::estimate_poses` → DUSt3R-derived `fast_pnp`
(`utils/recon_util.py`):

Verified against the cached upstream source (`.cache/upstream/{model,runner}.py`)
and the fetched `recon_util.py`, the **scene** recipe (the live use case) is:

| step | upstream (scene path) | here |
|---|---|---|
| correspondences | pixel ↔ gaussian center `X` (ch 0:3) | `pnp.pixel_grid` (integer px) + `points` |
| validity mask (PnP) | `sigmoid(opacity) > 0.05` | `estimate_poses(opacity_threshold=)` (engine output is already activated) |
| focal | Weiszfeld, **view 0 only** (`use_first_focal=True`), **all pixels** (`masks=None`) | `estimate_poses(use_first_focal=True)` → `focal.estimate_focal` |
| pose | `cv2.solvePnPRansac(reprojErr=5, SOLVEPNP_SQPNP, iters=10)`, `cam2world=inv(world2cam)` | `pnp.solve_pnp_cv2` (exact) / `solve_pnp_numpy` (reference) |
| anchor | none — view 0 is ~identity by construction (gaussians live in view 0's frame) | `estimate_poses` returns raw poses |
| scene scale | runner rescales **camera centers** so the view0→view1 **baseline = 1**: `s = 1/(‖t₁−t₀‖+1e-2)` | `pnp.rescale_to_baseline` / `estimate_poses(normalize=True)` |

> An earlier draft had three of these wrong (focal averaged over all views, an
> `inv(c2w[0])` re-anchor, and a `1/mean‖pts‖` scene scale). Driving the *actual*
> upstream `estimate_poses` on real engine output (see `check_upstream_parity.py`)
> surfaced and fixed all three — the focal one mattered: averaging a low-overlap
> second view dragged focal 507 vs the correct 596, a 15% error → 1.35° pose error.

`solve_pnp_cv2` is the byte-for-byte upstream call (needs `opencv`, in the CUDA
container or the nix shell). `solve_pnp_numpy` is a DLT + RANSAC reference with no
third-party deps, verified against analytic ground truth so the pipeline runs
anywhere numpy is.

## Files

- `focal.py` — Weiszfeld shared-focal estimation (mirrors upstream `estimate_focal`).
- `pnp.py` — DLT/RANSAC + cv2 PnP backends, `estimate_poses` orchestration, and
  `rescale_to_baseline` (the runner's scene normalization).
- `align.py` — Umeyama similarity fit, RANSAC variant, the **residual ladder**
  (`diagnose`: rigid→similarity→affine + depth-correlation) that detects whether a
  single uniform scale suffices or the mismatch is non-uniform, and similarity
  chaining / loop-closure metrics.
- `_upstream.py` — vendored upstream kernels (`fast_pnp` verbatim, `estimate_focal`
  numpy-transcribed) used ONLY by the parity check; not our code.
- `test_pose.py` — asset-free golden tests (no model, no fixtures, no cv2).
- `check_cv2_parity.py` — `solve_pnp_numpy` vs the exact `cv2.solvePnPRansac` on
  synthetic ground truth (needs cv2).
- `check_upstream_parity.py` — our whole orchestration vs the upstream recipe on
  REAL engine output (needs cv2 + a dumped `[N,H,W,23]` `.f32`).
- `empirical.py` — cross-run residual-ladder on real engine output (the scale test).

### Dense GT-posed control (is it us, or the data/model?)

- `tt_control.py` / `tt_experiment.py` — Tanks-and-Temples (NSVF) loader + engine
  vs GT-pose check. **Verdict: T&T is OUT OF DISTRIBUTION** for FreeSplatter-scene
  (narrow-FOV object orbits): opacity confident on only 8–17% of pixels, pose error
  28–145°. Kept as the harness + the negative result.
- `re10k_control.py` / `re10k_fetch.py` / `re10k_experiment.py` — RealEstate10K
  loader (parser + GT geometry), yt-dlp/ffmpeg frame fetch, and the engine vs
  GT-pose check. **In distribution → the control works**: relative pose recovered
  to **0.4–1.5°** vs GT, opacity confident on 68–75% of pixels.

## Run the tests

```sh
nix develop -c python3 pose/test_pose.py                      # golden, numpy only
nix develop -c python3 pose/check_cv2_parity.py               # solver vs cv2 (needs cv2)
EMP_DIR=/path/to/dumps nix develop -c python3 \
    pose/check_upstream_parity.py A_scn.f32                   # orchestration vs upstream
# dense control (needs the engine + a GGUF + network for re10k frames):
nix develop -c python3 pose/re10k_fetch.py CLIP.txt OUTDIR 10
FS_DEVICE=cpu nix develop -c python3 \
    pose/re10k_experiment.py CLIP.txt OUTDIR 40 20,40,80,160
```

## Status

- ✅ **Solver parity** — `solve_pnp_numpy` ≡ `cv2.solvePnPRansac` (~1e-7 clean).
- ✅ **Orchestration parity** — our `estimate_poses` is **bit-exact** to upstream's
  scene recipe on real engine output (focal/PnP/relative-pose all 0.00° at matched
  precision; the only residual is upstream's float32-K vs our float64, a RANSAC
  inlier-boundary effect of ≤0.5° on near-degenerate object data, 0° on scene).
- ✅ **Empirical scale test** — cross-run mismatch is a uniform-scale similarity
  (scene ~11% scale drift), no nonlinear warp; `diagnose` classifies it.
- ✅ **Dense GT-posed control** — pipeline validated against INDEPENDENT GT on
  in-distribution re10k (relative pose to 0.4–1.5°); T&T confirmed OOD. Finding:
  the model has a constant wide-FOV focal bias (recovers ~274 vs GT ~439) — benign
  for relative accumulation, off for metric scale.

## Not done yet (honest)

- **Cross-run consistency / accumulation test** on in-distribution data (the live
  question: clean accumulation vs needing consensus fusion).
- A **higher-motion clip** for the wide-baseline sweet-spot sweep (the sample clip
  is a slow pan, max ~6° over 160 frames).
- The live pipeline itself (sliding window, sim3 pose-graph to bound drift) and
  consensus fusion (the floater-removal step) — separate, later.
