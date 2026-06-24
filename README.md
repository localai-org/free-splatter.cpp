<!-- Audience: the end user who just wants to turn images into Gaussians and has
     the least possible interest in internals. Keep technical/validation detail
     in CLAUDE.md and the script docstrings, not here. Only include detail that
     is needed to USE the tool (input layout, output shape). -->
# free-splatter.cpp

Turn a handful of plain photos of an object or scene into 3D Gaussians — no
camera poses, no GPU required, no Python. A small C/C++ program (built on
[ggml](https://github.com/ggml-org/ggml)) that runs the FreeSplatter neural
network on your CPU (or a Vulkan GPU).

You give it N images; it gives you, for every pixel, a 3D Gaussian (position,
colour, opacity, size, orientation) that a Gaussian-splatting viewer can render.

> **Status:** under construction. The model runs are being brought up and
> validated one layer at a time; see `CLAUDE.md` for the engineering approach.

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
free_splatter-cli model.gguf images.f32
```

- `model.gguf` — the model weights (see Releases).
- `images.f32` — your input views as raw 32-bit floats in `[0,1]`, laid out one
  view after another, each view as channels-then-rows-then-columns (RGB,
  512×512). The number of views is detected automatically.

The output is one Gaussian per pixel per view. Use `--out result.f32` to save it,
`--device vulkan` to run on a GPU.

## License

Apache-2.0. Model weights are derived from
[TencentARC/FreeSplatter](https://github.com/TencentARC/FreeSplatter)
(Apache-2.0).
