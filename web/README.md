# Demo viewer

A self-contained WebGL2 gaussian-splat viewer (`index.html`). It loads a
`.splat` **progressively** — splats are pre-sorted by importance, so the scene
builds up (dominant structure first) as the file streams in — with depth-sorted
alpha blending, orbit/zoom/WASD camera, and an HTML/CSS status panel.

The heavy step (photos → gaussians) runs **off the browser** (CPU or Vulkan,
seconds on a GPU); the browser only renders the precomputed `.splat`.

## Run

The viewer must be **served** (the sort runs in a Web Worker; `file://` won't
work):

```sh
# 1. make a demo .splat from two scene photos (inside the nix shell)
nix develop -c scripts/make_demo.sh web/scene.splat \
    .cache/scenes/scannetpp_1/001.jpg .cache/scenes/scannetpp_1/002.jpg

# 2. serve and open
python3 -m http.server -d web 8000      # then open http://localhost:8000
```

Load a different file with `?splat=other.splat`.

## Accumulating-reconstruction demo (`accumulate.html`)

A second viewer that plays the **growing world** the pose pipeline builds: the
cloud reconstructed from 2 photos, then 3, then 4, … as each new view is folded
in (`free_splatter-cli --accumulate` → cross-run Sim(3) alignment). The input
photos appear one-per-step in the top-right filmstrip, newest highlighted; the
WebGL renderer is the same EWA splatting as `index.html`.

```sh
# bake the demo (engine pass over a photo stream -> a self-contained demo dir)
nix develop -c scripts/make_accumulate_demo.sh .cache/demo/accumulate \
    frames/f0000.png frames/f0020.png frames/f0040.png frames/f0060.png ...

# serve it (the bake copies accumulate.html in as index.html)
python3 -m http.server -d .cache/demo/accumulate 8080   # open http://localhost:8080
```

URL params: `?ms=2600` step interval, `?start=N` begin at step N, `?auto=0` no
auto-advance, `?spin=0` no auto-orbit. The cloud shown is the raw accumulated
cloud; `--fuse` also writes a consensus-fused `acc_fused.splat` (the edge-haze
floaters removed).

## Sanity-checking the data separately

The `.splat` uses the antimatter15 layout, so you can verify the *data* (apart
from this renderer) by dragging it onto <https://antimatter15.com/splat/>. If it
looks right there but not here, the bug is in this viewer's camera/conventions.
