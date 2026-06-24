#!/usr/bin/env bash
# Build the WASM module (CPU backend only) and stage it under web/vendor/.
# Run inside the Emscripten dev shell:  nix develop .#wasm && web/build-wasm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD=build/wasm
emcmake cmake -S . -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DFREE_SPLATTER_BUILD_TOOLS=OFF \
    -DFREE_SPLATTER_BUILD_TESTS=OFF
emmake cmake --build "$BUILD" --target free_splatter_wasm -j

# Keep Emscripten's names: the .js has its .wasm filename baked in, so the pair
# must stay together.
mkdir -p web/vendor
cp "$BUILD"/free_splatter_wasm.js   web/vendor/
cp "$BUILD"/free_splatter_wasm.wasm web/vendor/
echo "staged web/vendor/free_splatter_wasm.{js,wasm}"
