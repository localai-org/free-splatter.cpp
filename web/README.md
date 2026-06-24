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

## Sanity-checking the data separately

The `.splat` uses the antimatter15 layout, so you can verify the *data* (apart
from this renderer) by dragging it onto <https://antimatter15.com/splat/>. If it
looks right there but not here, the bug is in this viewer's camera/conventions.
