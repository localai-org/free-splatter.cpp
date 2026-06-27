# Fuzzing

libFuzzer harnesses for the **untrusted** input surfaces, built with
`-fsanitize=fuzzer,address,undefined`.

```sh
nix develop .#fuzz
cmake --preset fuzz && cmake --build --preset fuzz
mkdir -p corpus_img    && ./build/fuzz/fuzz_image   -max_total_time=60 corpus_img
mkdir -p corpus_opt    && ./build/fuzz/fuzz_options -max_total_time=60 corpus_opt
mkdir -p corpus_pose   && ./build/fuzz/fuzz_pose    -max_total_time=60 corpus_pose
mkdir -p corpus_decode && ./build/fuzz/fuzz_decode  -max_total_time=60 corpus_decode
```

## Targets

- **`fuzz_image`** — `free_splatter::ingest_images`, the boundary all
  caller-supplied pixels cross. Drives arbitrary view counts, resolutions and
  pixel bit-patterns (NaN/Inf/denormals) and asserts the validator never
  crashes, reads OOB, or returns a populated buffer on a rejected input.
- **`fuzz_options`** — the options-builder / device-string setters with
  arbitrary byte strings; asserts no crash and NULL-safe frees.
- **`fuzz_pose`** — the public pose C-API: `free_splatter_estimate_poses` and the
  accumulator (`add_pair` → `cloud` / `fuse` / `camera_path`) driven with
  arbitrary float gaussian buffers (NaN/Inf/denormals) and fuzz-chosen geometry.
  Caught a real SIGFPE (the RANSAC sampler's `% N` with N=0 valid
  correspondences) and motivated the non-finite guards in `consensus_fuse`
  (the float→int voxel-coordinate cast).
- **`fuzz_decode`** — the untrusted image-FILE path: arbitrary bytes →
  `stb_image` → center-crop + resize → CHW (the surface a user photo crosses in
  the CLI / demo). stb is vendored THIRD-PARTY; per CLAUDE.md we fuzz the boundary
  and would guard (not patch) any stb-internal sanitizer trip — none seen so far.

## Trust boundary (intentional)

The **GGUF model file is TRUSTED** and is deliberately **not** fuzzed: it is our
own converter's output, loaded by us. Image inputs (encoded files and decoded
pixels) and the public C-API arguments are **untrusted** and fuzzed. Keep this
asymmetry — see `CLAUDE.md`.
