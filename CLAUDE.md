# CLAUDE.md — working rules for free-splatter.cpp

## What this project is

A GGML/C++ engine for the **neural-network front half** of
TencentARC/FreeSplatter (image patch tokenizer → multi-view self-attention
transformer → per-pixel 3D-Gaussian parameter head) **plus the downstream
camera-pose recovery and cross-run registration that consume its output**. Given
N uncalibrated views it returns, per input pixel, the activated Gaussian
parameters; **PnP** then recovers each view's camera, and successive runs are
aligned into one accumulating world — the path toward live reconstruction from a
moving camera.

**Scope (updated):** the engine (pieces 1–3), **PnP pose recovery (now IN
scope)**, and the cross-run Sim(3) alignment / accumulation. Keep the seam at the
`[N, H, W, gaussian_channels]` tensor clean — it is the contract between the
engine and the pose consumer. Rendering itself stays in the demo viewer
(Vulkan/WebGL), not the engine.

Target checkpoint first: **`freesplatter-scene`** (`gaussian_channels=23`,
`sh_residual=true`, black background). The transformer backbone is identical
across all three variants, so object/object-2dgs are cheap follow-ons.

## Language & dependency policy

- **Everything ships in C++** (the engine, PnP, and the alignment/accumulation
  once proven). **Go only for the demo web server** — the purego layer that drives
  the C API → Vulkan + WebGL viewer.
- **The CLI and the C API must have NO Python dependency at runtime.** Every piece
  of current and future functionality is reachable from `free_splatter-cli` and
  `include/free_splatter.h` without invoking Python.
- **Python is confined to two places, neither shipped:**
  1. **Dev-time reference / conversion / validation** that runs in
     `docker/Dockerfile.cuda` (`scripts/hf_dump.py`, `convert.py`,
     `compare_taps.py`, …) — the only place torch runs; never a runtime dependency.
  2. **The `pose/` research prototype, TEMPORARILY.** It may continue in Python
     (numpy + cv2) **only until the accumulating-reconstruction approach is
     proven**. Once proven it is **rewritten in C++**, exposed via the CLI + C API,
     and **the Python is deleted.** `pose/` is not wired into CMake/ctest and is a
     throwaway prototype, not a parallel implementation to maintain.

## Validation is the backbone (non-negotiable)

This is a numerical port. Correctness means matching the PyTorch reference
**layer by layer**, not "it runs and looks plausible."

- **Every op is tapped.** Each meaningful intermediate gets a stable
  `ggml_set_name()` and is dumped (`free_splatter-cli --dump-taps DIR`) in the
  format `scripts/compare_taps.py` reads.
- **A piece is not "done" until its taps pass** `compare_taps.py` against the
  float64 PyTorch reference (`scripts/hf_dump.py`), in **topological order**,
  with the **first divergence fixed** — never paper over a downstream symptom of
  an upstream bug. compare_taps stops at the first failing tap on purpose.
- **New graph ops land with an asset-free golden-op test first**
  (`tests/test_graph_blocks.cpp`), pinned to hand-computed f64 references,
  **before** any fixture-based parity claim. A wrong op should fail with zero
  fixtures.
- **Gates:** per-row `cosine ≥ 0.99999` AND normalized `row-err ≤ 1e-2` on
  CPU-f32. Widen only via `compare_taps.py --scale N` for Vulkan/fp16, and write
  down why N.
- **Sensitive ops are computed in f64 in the reference** (attention softmax,
  GELU, norms): the engine targets the exact value, not a second noisy float
  path. Bring each stage up in **f32 first** to isolate algorithmic bugs, then
  switch weights to f16 and re-check the head error stays within gate.
- **Known silent-bug hotspots** (test these in isolation before trusting
  anything downstream): the patchify `(ic,kh,kw)` flatten order; the unpatchify
  `(p,q,c)` pixel-unshuffle order; attention precision (use
  `ggml_flash_attn_ext`/f32, never f16 softmax); GELU must be `ggml_gelu_erf`
  (exact erf), never `ggml_gelu` (tanh); the scale activation
  (`min + (max-min)·sigmoid`, not exp).

## Per-component discipline

Each component (`image`, `gguf_loader`, `backend`, `model`, the head, and — once
ported to C++ — **`pose` = PnP + focal + Sim(3) alignment**) has its own unit test
and is made independently green **before** cross-component parity. Component
boundaries match the file layout. Keep the seam at the
`[N,H,W,gaussian_channels]` tensor clean — that is the contract between the engine
and the pose/rendering consumers. The C++ `pose` component inherits the parity
discipline the Python prototype already established: **bit-exact to upstream
`estimate_poses`** (see `pose/check_upstream_parity.py`) and **validated against
independent ground-truth poses** (`pose/re10k_experiment.py`).

## Debugging philosophy (mandatory sequence)

1. **Collect information first.** On any bug, add/inspect taps, profile, trace
   shapes and dtypes, and survey the upstream FreeSplatter source / the relevant
   papers / other ggml ports **before** proposing a fix. Make the cause
   unambiguous first.
2. **Prefer admitting ignorance** and gathering more context over committing to a
   fix on a hunch. A wrong fix that masks a symptom is worse than no fix.
3. **Verify the fix did what you predicted.** After fixing, re-run the *specific*
   failing tap and confirm it now passes **for the reason you expected** — check
   that neighbouring taps did not shift, and that the gate which was failing is
   the gate now passing. Do not declare victory on a green aggregate.

## Trust boundary

- **The GGUF model file is TRUSTED** — not fuzzed, not hardened. Loading is for
  our own converted weights.
- **Image inputs are UNTRUSTED** — validated in `src/image.cpp`, fuzzed
  (`fuzz/fuzz_image.cpp`), and the test suite runs ASan/UBSan-clean. Keep this
  asymmetry intentional.

## Commands

- `nix develop` then `cmake --preset debug && cmake --build --preset debug` —
  ASan/UBSan, the verification build.
- `ctest --preset debug -LE model` — fast, asset-free tier (golden-op + unit).
- `FREE_SPLATTER_GGUF=…f32.gguf FREE_SPLATTER_FIXTURES=…/2view ctest -L model` —
  full per-layer + head parity (needs converted weights + fixtures;
  `FREE_SPLATTER_MAX_BLOCKS=1` for a fast piece-1 + block-0 subset).
- `cmake --preset vulkan` — Vulkan backend parity.
- `docker/Dockerfile.cuda` is the **only** place the upstream PyTorch model runs
  (reference taps + weight conversion). The engine never depends on torch/CUDA.

## README boundary

The README is for the **end user with the least interest in internals**. Keep
technical and validation detail HERE (and in script docstrings), not in the
README. See its top comment for the audience rule.
