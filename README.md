<!-- Audience: the end user turning images into Gaussians, and developers embedding
     the engine via its C API. Keep validation/internal detail in CLAUDE.md and the
     script docstrings — only what's needed to USE the tool (input layout, output
     shape, the public API). -->
# free-splatter.cpp

Turn a handful of plain photos of an object or scene into 3D Gaussians — no
camera poses, no GPU required, no Python. A small C/C++ program (built on
[ggml](https://github.com/ggml-org/ggml)) that runs the FreeSplatter neural
network on your CPU (or a Vulkan GPU).

You give it N images; it gives you, for every pixel, a 3D Gaussian (position,
colour, opacity, size, orientation) that a Gaussian-splatting viewer can render.

## Build

```sh
nix develop            # gets cmake, a compiler, and (optionally) Vulkan
cmake --preset release
cmake --build --preset release
```

No Nix? You just need CMake, a C++17 compiler, and the bundled `ggml` submodule
(`git submodule update --init`).

## Run

```sh
free_splatter-cli --device vulkan --splat scene.splat model.gguf view1.jpg view2.jpg
```

- `model.gguf` — the model weights, from
  [huggingface.co/LocalAI-io/free-splatter.cpp](https://huggingface.co/LocalAI-io/free-splatter.cpp)
  (e.g. `freesplatter-scene-f16.gguf`).
- the input views — ordinary image files (JPG/PNG/…); each is center-cropped and
  resized to 512×512. (Or pass one raw `.f32` file: views as 32-bit floats in
  `[0,1]`, view-major channels-then-rows-then-columns.)
- `--splat scene.splat` — write a gaussian-splat file for the `web/` viewer
  (`--max-splats N` caps the count, `--opacity-threshold T` prunes faint ones);
  `--out result.f32` saves the raw per-pixel gaussians instead.
- `--device vulkan` runs on a GPU (~seconds); the default CPU works everywhere.

See `web/README.md` to view the `.splat` in the browser.

## Web app & demo video

A small Go server (it binds the C API below via [purego](https://github.com/ebitengine/purego),
no cgo) turns photos dropped in your browser into a splat, and renders a
multi-scene **demo video** — several reconstructions placed in one 3D space with
the camera panning between them as each builds up. From the repo root:

```sh
nix develop -c scripts/serve.sh                       # serves http://localhost:8080
```

For the demo video, drop one folder of photos per scene/object under
`demo-photos/`, then bake and render:

```sh
SERVER=http://127.0.0.1:8080 scripts/demo/bake.sh     # reconstruct each folder (reuses unchanged)
# then open http://localhost:8080/demo.html and press Render (uses your GPU), or:
GPU=1 PORT=8080 RES=1920x1080 scripts/demo/make.sh    # render the MP4 offline (headless)
```

Pacing, zoom, vibrance, build-up curve and more are live URL knobs; full details
and the tuning reference are in [`server/README.md`](server/README.md).

## Embedding (C API)

The engine is a flat C library (`include/free_splatter.h`, build with
`-DFREE_SPLATTER_BUILD_SHARED=ON`, add `-DFREE_SPLATTER_VULKAN=ON` for GPU) shaped
for FFI — opaque handle, caller-owned buffers, no exceptions cross the boundary,
every free is NULL-safe. Failures are reported via `free_splatter_last_error`
(NULL = ok). A complete binding lives in the web server's `server/engine.go`.

```c
#include "free_splatter.h"

free_splatter_options *opts = free_splatter_options_new();
free_splatter_options_set_device(opts, "vulkan");      // "cpu" | "gpu" | "cuda" | "vulkan"[:N]
free_splatter_ctx *ctx = free_splatter_load("model.gguf", opts);
free_splatter_options_free(opts);
if (free_splatter_last_error(ctx)) { /* load failed — inspect, then free */ }

free_splatter_geometry geo;                            // size buffers without hardcoding constants
free_splatter_geometry_of(ctx, &geo);                  // in_channels=3, image_{height,width}=512, gaussian_channels=23

// images: n_views * in_channels * H * W float32 in [0,1], NCHW per view.
float *out; size_t n_out;
if (free_splatter_run(ctx, images, n_views, geo.image_height, geo.image_width, &out, &n_out) == 0) {
    // out: n_views * H * W * gaussian_channels float32 — activated, render-ready gaussians
    free_splatter_buf_free(out);
}
free_splatter_free(ctx);                               // NULL-safe
```

| function | purpose |
|---|---|
| `free_splatter_abi_version()` | ABI check against `FREE_SPLATTER_ABI_VERSION` (1) |
| `free_splatter_options_new` / `_free` + `_set_device` / `_set_threads` / `_set_dump_taps_dir` | configure (ABI-stable builder) |
| `free_splatter_load(gguf, opts)` → `ctx*` | load weights; non-NULL `last_error` means it failed |
| `free_splatter_geometry_of(ctx, *out)` | expected input dims + output channel count |
| `free_splatter_run(ctx, images, n_views, H, W, **out, *n_out)` | inference → malloc'd `[N, H, W, gaussian_channels]` |
| `free_splatter_buf_free` · `free_splatter_free` | release output buffer · context (both NULL-safe) |

The `[n_views, H, W, gaussian_channels]` output (xyz · SH · opacity · scale ·
rotation, per pixel) is the clean seam to any Gaussian-splatting rasterizer.

## License

Apache-2.0. Model weights are derived from
[TencentARC/FreeSplatter](https://github.com/TencentARC/FreeSplatter)
(Apache-2.0).
