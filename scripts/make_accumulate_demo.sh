#!/usr/bin/env bash
# Bake the accumulating-reconstruction demo: run the engine over a stream of
# photos, fold them into one growing world (free_splatter-cli --accumulate), and
# lay out a self-contained web demo — the cloud reconstructed from 2 photos, then
# 3, then 4, …, plus the input thumbnails and the accumulate.html viewer.
#
#   MODEL=.cache/freesplatter-scene-f16.gguf \
#     scripts/make_accumulate_demo.sh OUT_DIR FRAME0 FRAME1 FRAME2 ...
#
# Then serve it:  cd OUT_DIR && python3 -m http.server 8080   (open index.html)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
MODEL=${MODEL:-.cache/freesplatter-scene-f16.gguf}
MAXSPLATS=${MAXSPLATS:-500000}
DEVICE=${DEVICE:-cpu}
CLI=${CLI:-$ROOT/build/release/bin/free_splatter-cli}

[ "$#" -ge 3 ] || { echo "usage: $0 OUT_DIR FRAME0 FRAME1 [FRAME2 ...]  (>=2 frames)" >&2; exit 1; }
OUT=$1; shift
FRAMES=("$@")
mkdir -p "$OUT"

# 1) input thumbnails: center-crop square + resize, mirroring the engine's own
#    preprocessing so the strip shows exactly the view that was reconstructed.
for i in "${!FRAMES[@]}"; do
  ffmpeg -y -loglevel error -i "${FRAMES[$i]}" \
    -vf "crop='min(iw,ih)':'min(iw,ih)',scale=360:360" "$OUT/view_$i.jpg"
done

# 2) accumulate: one engine pass over the stream writes acc_2.splat, acc_3.splat,
#    ... (the cloud after each photo is folded in) + acc_fused.splat (the
#    consensus surface: voxels seen by >= K frames, single-view floaters removed).
"$CLI" --device "$DEVICE" --accumulate --fuse --splat-prefix "$OUT/acc" \
  --max-splats "$MAXSPLATS" "$MODEL" "${FRAMES[@]}"

# 3) manifest.json: one step per added photo (acc_n.splat + the n thumbnails, the
#    last being the one just added), then a final consensus-fused step.
n=${#FRAMES[@]}
allimgs=""
for ((j=0;j<n;j++)); do allimgs+="\"view_$j.jpg\""; [ $j -lt $((n-1)) ] && allimgs+=", "; done
{
  echo '{ "steps": ['
  for ((k=2;k<=n;k++)); do
    imgs=""
    for ((j=0;j<k;j++)); do imgs+="\"view_$j.jpg\""; [ $j -lt $((k-1)) ] && imgs+=", "; done
    printf '    {"splat":"acc_%d.splat","images":[%s],"n":%d},\n' "$k" "$imgs" "$k"
  done
  printf '    {"splat":"acc_fused.splat","images":[%s],"n":%d,"label":"consensus-fused — single-view floaters removed"}\n' "$allimgs" "$n"
  echo '  ] }'
} > "$OUT/manifest.json"

# 4) the viewer, self-contained next to its assets
cp "$ROOT/web/accumulate.html" "$OUT/index.html"
echo "demo baked -> $OUT"
echo "serve: (cd $OUT && python3 -m http.server 8080)  then open http://localhost:8080/"
