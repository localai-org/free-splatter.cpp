#!/usr/bin/env bash
# End-to-end demo asset: scene images -> .splat for the WebGL viewer.
#   nix develop -c scripts/make_demo.sh [out.splat] img1 img2 [img3 ...]
# Env: FREE_SPLATTER_GGUF, FREE_SPLATTER_CLI, FREE_SPLATTER_DEVICE, PY.
set -euo pipefail
cd "$(dirname "$0")/.."

GGUF=${FREE_SPLATTER_GGUF:-.cache/freesplatter-scene-f32.gguf}
DEVICE=${FREE_SPLATTER_DEVICE:-vulkan}
CLI=${FREE_SPLATTER_CLI:-build/vulkan/free_splatter-cli}
PY=${PY:-.venv-torch/bin/python}

out="${1:-web/scene.splat}"; shift || true
[ "$#" -ge 2 ] || { echo "usage: make_demo.sh [out.splat] img1 img2 [...]" >&2; exit 2; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$PY" scripts/preprocess_scene.py "$tmp/scene.f32" "$@"
"$CLI" --device "$DEVICE" --out "$tmp/g.f32" "$GGUF" "$tmp/scene.f32"
"$PY" scripts/to_splat.py "$tmp/g.f32" "$out"
echo "demo asset -> $out  (serve web/ and open index.html)"
