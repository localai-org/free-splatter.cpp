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

## C++ port status (the ship target — `src/pose.{h,cpp}`)

The port has begun, dependency-free (only the self-contained `src/linalg.h`
Jacobi eigensolver — no Eigen/OpenCV), wired into the library and the asset-free
test tier (`tests/test_pose.cpp`, `ctest -LE model`):

- ✅ **`focal.py` → `estimate_focal`** — Weiszfeld; **bit-exact** to the numpy
  reference on a real scene dump (596.408591886 both).
- ✅ **`align.py` → Umeyama / RANSAC / residual-ladder / Sim(3) chaining /
  `sim_frac_power`** — `sim_frac_power` is a closed-form Sim(3) one-parameter
  subgroup (`A^f=s^f R^f`, translation `(A^f−I)(A−I)⁻¹t`), no complex eig needed.
  All 9 golden tests (the mirror of `test_pose.py`) pass under ASan/UBSan.
- ✅ **Robust PnP → `solve_pnp` (EPnP + Gauss-Newton)** — `solve_pnp_numpy` (DLT +
  RANSAC, via the 12×12 `AᵀA` nullspace) is kept as the asset-free golden-test
  reference, but on real scenes the DLT inherits the textbook **planar/mirror
  degeneracy** (seed-dependent: ~3/5 RANSAC seeds near cv2, ~2/5 a ~135–152°
  flip). The shipped real-data solver is `solve_pnp`: **EPnP** (planar-robust,
  non-iterative, uses all points so there is no random minimal-sample flip —
  reuses the 12×12 `MᵀM` Jacobi eigendecomposition) for the init, then a
  **Huber-robust Gauss-Newton** reprojection refine. On the real scene dump it is
  **deterministic** (identical across all RANSAC seeds) and lands at **0.73° of
  the upstream cv2/SQPNP** relative rotation (0.74° translation direction) — i.e.
  **cv2-parity on real data with no OpenCV dependency**, versus the numpy DLT's
  175° miss on the same dump. Golden tests cover exact recovery, a near-planar
  config (where DLT flips), and 15% gross-outlier rejection.
- ✅ **Accumulation chaining → `Accumulator` (accumulate.py)** — the live
  sliding-window loop: feed each consecutive pair's `[2,H,W,gc]` engine output,
  it recovers the pair's cameras (`estimate_poses`), fits the cross-run Sim(3) on
  the shared frame, composes a global chain, and drops every new frame's gaussians
  into one world (`cloud()`, `camera_path()`). Golden-tested (synthetic clip:
  trajectory ATE 7.6e-8 of extent, per-link scale to 3e-9). **Real-data parity**
  on the 13 cached `pair_*.f32` dumps vs the numpy/cv2 prototype: cloud size
  **bit-exact** (2,633,725) and per-link valid% identical (deterministic mask);
  per-link Sim(3) scale agrees to **mean 0.5%** (11/12 links <1%, worst on the
  documented low-inlier leg); trajectory within **6.6%** of the cv2 chain — the
  residual is the known RANSAC-RNG + EPnP-vs-SQPNP delta, not a port gap.
- ✅ **Loop closure → `sim4_invert` + `distribute_drift` (loop_closure.py)** — the
  even Sim(3) relaxation: from an accumulated drift `D` (the loop-closure
  measurement `P_n_loop · P_last⁻¹`), distribute `D^(k/n)` over the chain.
  `sim_frac_power` (the closed-form one-parameter subgroup) is **numerically
  identical to numpy's eig-based version (5e-10)** across f ∈ {−0.5 … 1.3};
  `sim4_invert` is an exact similarity inverse (1e-9); golden test recovers a
  known drifted loop to 4e-16. **Real-data parity** on the cached loopcache (13
  chain pairs + `close_0_260`): the recovered drift matches the prototype's loop
  error (scale 1.09 vs 1.12, **4.6° vs 4.4°**, README "4.4°"), deterministic
  valid% identical; the corrected-trajectory delta is the (already-characterized)
  EPnP-vs-cv2 PnP backend feeding `D`, since the distribution math itself is
  bit-identical to numpy. Confirms the prototype's finding: on this short clip the
  chain already closes, so loop closure is a near-identity correction.
- ⬜ Not yet ported: consensus fusion (composes the primitives above), and the
  CLI / C-API surface.

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
- `re10k_crossrun.py` — cross-run consistency vs baseline (the accumulation
  question): a shared frame's two reconstructions agree on **98% of pixels within
  10%** of scene extent at small baseline.

### Accumulation prototype (the live idea, assembled)

- `accumulate.py` — slide a window over a clip, recover each pair (PnP), chain a
  Sim(3) between consecutive runs via their shared-frame correspondences
  (`align.compose`), drop every frame's gaussians into one world, and measure the
  recovered camera trajectory vs GT. `render_ply.py` — pinhole projection of the
  colored cloud to a PNG (visual coherence check).
- **Result (13 pairs, stride 20):** per-link Sim(3) residual ~1.0–1.4%; trajectory
  ATE **11%** of extent with drift growing 7%→13% (monotone monocular scale drift,
  0.755 over 12 links); the 2.6M-point cloud renders as a **coherent room** matching
  the input. The accumulating-reconstruction idea is **proven** end-to-end.
- `find_loop.py` — search re10k poses for a clip that revisits its start (the loop
  substrate). `loop_closure.py` — chain open-loop, measure the loop error via a
  closing pair, then distribute it (even Sim(3) relaxation, `align.sim_frac_power`)
  and report ATE before/after. **Finding:** loop closure helps only when drift
  *accumulates* into a large loop error. On the real loop clip tested it did NOT
  help — the open-loop chain already closes (loop error 4.4° / scale 1.12 / 8%
  trans), so the ~34% ATE is per-link **odometry noise** (inlier% as low as 17%
  on fast legs) + the focal-bias warp, *self-consistent but distorted vs GT*, which
  loop closure can't fix. The correction machinery itself is verified to recover
  synthetic accumulated drift to ~0 (`test_pose.py::test_loop_correction`).
- `fuse.py` — **consensus fusion** (the edge-noise answer): voxelize the accumulated
  cloud at the consistency scale and keep only voxels corroborated by ≥K distinct
  frames, averaging the agreeing predictions. On the forward clip: **14% of points
  are single-frame floaters** that render as incoherent edge-haze, while the ≥2-frame
  consensus renders as a **clean, crisp room** with the haze gone. Tradeoff: dropping
  single-frame points also trims the single-view periphery (coverage vs cleanliness).

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
- ✅ **Accumulation prototype** — sliding-window Sim(3) chaining over a clip builds
  one coherent world; trajectory tracks GT to ~11% ATE with bounded-able drift. The
  idea is proven; per CLAUDE.md the next implementation step is the **C++ port**.
- ✅ **Loop closure** — implemented + machinery verified on synthetic drift (recovers
  to ~0). On a real loop clip it did NOT help: the chain already closes, so the error
  is odometry noise + focal warp, not accumulated drift. Lesson: better odometry
  (smaller baselines / fusion / focal) is the lever for short loops; loop closure
  pays off on long trajectories with consistent accumulated drift.
- ✅ **Consensus fusion** — gating the accumulated cloud on ≥2-frame agreement removes
  the 14% single-frame floaters (incoherent edge-haze) and yields a clean surface —
  the definitive "yes, accumulation removes the edge noise." Trades single-view
  periphery for a cleaner core.

## Not done yet (honest)

- The **focal bias** (recovered ~274 vs GT ~440+) — the remaining geometry-distortion
  source; worth understanding whether it's fixable.
- A **higher-motion clip** wide-baseline sweep, and a **long** trajectory where loop
  closure demonstrably pays off.
- **Fuse-then-align**: feed the consensus surface back into the odometry (lift the
  low-inlier links the loop-closure diagnosis flagged), not just the final cloud.
- The **C++ port** (CLI + C API, no Python) once the design is locked — then this
  prototype is deleted.
