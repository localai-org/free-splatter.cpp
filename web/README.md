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
auto-advance, `?spin=0` no auto-orbit. The bake also writes a consensus-fused
`acc_fused.splat` (single-view floaters removed — only voxels seen by ≥ K frames
survive), shown as a final step. `--fuse-mode kept` keeps every raw gaussian in a
consensus voxel (dense); `best` keeps only the most-confident frame per voxel
(dense AND de-ghosted); `averaged` collapses each voxel to one denoised point
(cleaner but sparse — only worthwhile with many overlapping frames).

**De-ghosting.** Pairwise chaining overlays each frame's reconstruction in world
space, so where two frames disagree about a surface you see doubled ("ghosted")
copies. The fused step removes them by **best-frame selection** (`--fuse-mode
best`): within each voxel it keeps only the single most-confident frame's
gaussians, so disagreeing copies aren't stacked — one frame represents each
region. This de-ghosts by *selection* (no averaging → no blur), at the cost of
visible seams where adjacent voxels pick different frames. The two artifact-free
alternatives are more expensive: fixing the registration (only helps when the
disagreement is a *rigid* pose drift — usually it isn't, it's per-pair stereo
depth disagreement baked into the network output, which no rigid alignment can
undo) or photometric 3DGS re-optimization. A gaussian-level averaging pass
(`--refine`, **off** by default) exists but is **not recommended**: it averages
points that already share a voxel — blurring the in-register surface — while
leaving ghosts that are more than a voxel apart untouched, so it trades sharpness
for no real de-ghosting.

**Pick frames with lateral baseline.** Two-view depth needs the camera to
*translate sideways* — a pure forward dolly (the camera moving along the direction
it faces) gives near-zero parallax and reconstructs blurry. Frames a few apart
from an orbiting / strafing clip work far better than tightly-spaced frames of a
walk-forward clip. Rule of thumb: the per-pair lateral baseline should be a few %
of scene depth or more.

**…or let the gate pick them (`MIN_PARALLAX`).** The bake passes
`--min-parallax DEG` (default 8°, `MIN_PARALLAX=0` to disable): a candidate frame
is folded in only if its triangulation angle vs the last *kept* frame clears the
threshold — otherwise its depth is ill-conditioned and the model invents it.
Skipped frames re-anchor against the last kept one (keyframe selection), so you can
feed a long, dense frame stream and let the gate curate the well-conditioned
subset. The threshold is the engine's *after-inference* angle
(`free_splatter-cli --parallax …`), which over-reports, so keep it well above
COLMAP's 1–2°. `scripts/parallax_ref.py` (cv2, dev-shell only) is the independent
geometric cross-check — agreement on a pair validates it; a large gap (geometric
angle tiny, model angle confident) flags the model hallucinating depth.

## Sanity-checking the data separately

The `.splat` uses the antimatter15 layout, so you can verify the *data* (apart
from this renderer) by dragging it onto <https://antimatter15.com/splat/>. If it
looks right there but not here, the bug is in this viewer's camera/conventions.
