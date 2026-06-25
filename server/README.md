# free-splatter web app

Drop a few photos of a scene in your browser and get back a 3D gaussian splat,
reconstructed in one feed-forward pass on the GPU.

This is a thin Go HTTP server that binds the `free_splatter` C API with
[purego](https://github.com/ebitengine/purego) (no cgo), runs inference on
**Vulkan**, and serves an embedded WebGL viewer. It is a *demo front-end* for the
engine — the numerical port, validation, and build details live in the repo root
`CLAUDE.md`.

## Run

```sh
# from the repo root (builds the Vulkan shared lib if needed, then serves):
nix develop -c scripts/serve.sh            # http://localhost:8080

# or directly, from this directory:
CGO_ENABLED=0 go run . \
  -lib    ../build/vulkan/libfree_splatter.so \
  -models scene=../.cache/freesplatter-scene-f16.gguf,object=../.cache/freesplatter-object-f16.gguf \
  -device vulkan -addr :8080
```

`CGO_ENABLED=0` is intentional: purego needs no C toolchain, and it gives
`net/http` the pure-Go resolver so the server builds on a bare `PATH`.

Flags: `-addr` `-lib` `-models` (comma list of `name=path.gguf`; the first is the
default) `-device` (`cpu`|`vulkan`|`vulkan:N`) `-max-splats` `-opacity-threshold`.

## Models

Several checkpoints can be loaded at once and switched per request (the uploader
shows a selector). All are the *same* transformer; they differ only in the head
config and what they were trained on:

| name | trained on | give it |
|---|---|---|
| `scene` | 2 views (BlendedMVS/ScanNet++/CO3D) | **2** overlapping photos of a scene |
| `object` | 4 views (Objaverse renders) | **3–4** photos around **one** object, plain background best |

Both are pose-free and reconstruct geometry purely from shared content, so the
input views **must overlap** — neither stitches non-overlapping photos into a
larger scene. Convert an object GGUF with
`scripts/convert.py ckpt.safetensors out.gguf --variant object`.

### Background removal (object path)

The object checkpoint wants the object isolated on white. Enable automatic
matting so users can drop normal photos:

```sh
nix develop -c scripts/setup_bgremove.sh   # one-time: venv + rembg (U2Net/onnxruntime)
```

`serve.sh` then auto-passes `-bgremove-cmd` and the uploader shows a **remove
background** toggle for the Object model. On a request with `remove_bg=1`, the
server runs the matting command (rembg batch, ~2 s/photo CPU), composites each
cutout on white, then reconstructs. It's a **subprocess**, not linked in — the
`-bgremove-cmd` template (`… {in} {out}` batch dirs) makes any matter pluggable,
and the feature simply stays hidden if no command is configured.

## How "build-up" works (it is not a diffusion step count)

The reconstruction is a single forward pass — there is **no iterative denoising
or step count**. What the viewer animates instead is a *reveal*: the engine sorts
the gaussians by importance (opacity × volume), so showing the top-K first makes
the scene assemble from its dominant structure to fine detail. The slider and the
▶ replay button scrub that reveal.

## REST API

| Method | Path | Body / params | Response |
|---|---|---|---|
| `GET`  | `/api/models` | — | `[{name,label,hint,views,size}, …]` (loaded checkpoints) |
| `POST` | `/api/reconstruct` | multipart `images` (1–8 photos) + optional `model` | `{id, model, n_views, n_splats, size, seconds}` |
| `GET`  | `/api/splat/{id}` | — | `application/octet-stream` antimatter15 `.splat` (32 B/splat, importance-sorted) |
| `GET`  | `/` | `?id=&yaw=&pitch=&dist=&reveal=` | embedded viewer (deep-link `?id=` skips the uploader) |

Photos are decoded and preprocessed (center-crop → 512² → CHW, `[0,1]`) in
memory-safe Go, so untrusted uploads never reach the C/stb path. The `.splat`
already carries the OpenCV→OpenGL orientation flip, so it renders upright in any
antimatter15-style viewer.

## Demo video (`/demo.html`)

A separate storyboard page renders a promo video: several baked scenes/objects
laid out in one shared 3D space, the camera panning between them with a slight
zoom (to read as 3D), each one opening on its source photos which then shrink to
the upper-right as the splats build up. It is **offline and deterministic** —
the page steps a fixed frame schedule (camera/reveal/billboard state is a pure
function of frame index), reads each frame back, and streams the PNGs to the
server, which assembles them with ffmpeg (same recipe as `~/c/LocalVQE/demo`).

**Add your own photos:** drop one folder per scene/object under `demo-photos/`
(e.g. `demo-photos/kitchen/` with 2 overlapping photos → scene model, or
`demo-photos/mug/` with 3–4 photos around one object → object model), then re-run
the bake and reload the page — each folder becomes a station, in alphabetical
order. Optional files in a folder: `model.txt`, `label.txt`, `nobg`. With no
folders dropped, the built-in sample scenes are baked. Preview the plan with
`DISCOVER_ONLY=1 scripts/demo/bake.sh`.

```sh
# 1. bake content: each demo-photos/ folder (or the samples) -> .splat + manifest.json
SERVER=http://127.0.0.1:4001 scripts/demo/bake.sh        # server must be up
# 2a. render on the GPU (recommended, fast): open the page in a real browser and
#     press "Render":   http://<host>:4001/demo.html
# 2b. render headless (offline/CI; needs bun + chromium):
RES=1280x720 scripts/demo/make.sh            # SwiftShader (slow)
GPU=1 RES=1920x1080 scripts/demo/make.sh     # real GPU via ANGLE/Vulkan (fast)
```

`scripts/demo/render.js` (bun) drives chromium over CDP to render the frames — a
plain headless load never runs the page and `--screenshot` exits after one frame,
so the dedicated driver is what makes offline capture work. Frame state is a pure
function of frame index, so the output is identical on GPU and software.

Edit the `STATIONS` list in `scripts/demo/bake.sh` to change what the video
features and `timeline`/`offset` in the emitted `manifest.json` for pacing/layout.
Output resolution is chosen in the page (1080p/720p/1:1/9:16 for different feeds);
`?t=<sec>` renders a single deterministic still (used for thumbnails). The page
reuses the viewer's EWA shaders, including the **vibrance** boost (`?sat=`), so
the video matches what you see live.

## Files

- `engine.go` — purego bindings to the flat C API (`include/free_splatter.h`)
- `convert.go` — photo → model input; gaussians → `.splat`
- `main.go` — HTTP routes + `go:embed web`
- `demo.go` — demo-video capture/encode endpoints (`/api/demo/*`, `/demo-assets/`)
- `web/index.html` — drag-drop uploader + EWA WebGL viewer + reveal slider + vibrance
- `web/demo.html` — multi-scene storyboard renderer (shared-world camera, image
  billboards, deterministic frame capture)
- `../scripts/demo/{bake,make}.sh`, `render.js` — bake content, drive the offline
  render (bun+CDP), assemble the MP4
