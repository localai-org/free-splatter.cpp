# Fuzzing

libFuzzer harnesses for the **untrusted** input surfaces, built with
`-fsanitize=fuzzer,address,undefined`.

```sh
nix develop .#fuzz
cmake --preset fuzz && cmake --build --preset fuzz
mkdir -p corpus_img && ./build/fuzz/fuzz_image -max_total_time=60 corpus_img
mkdir -p corpus_opt && ./build/fuzz/fuzz_options -max_total_time=60 corpus_opt
```

## Targets

- **`fuzz_image`** — `free_splatter::ingest_images`, the boundary all
  caller-supplied pixels cross. Drives arbitrary view counts, resolutions and
  pixel bit-patterns (NaN/Inf/denormals) and asserts the validator never
  crashes, reads OOB, or returns a populated buffer on a rejected input.
- **`fuzz_options`** — the options-builder / device-string setters with
  arbitrary byte strings; asserts no crash and NULL-safe frees.

## Trust boundary (intentional)

The **GGUF model file is TRUSTED** and is deliberately **not** fuzzed: it is our
own converter's output, loaded by us. Image inputs and the public C-API string
arguments are **untrusted** and fuzzed. Keep this asymmetry — see `CLAUDE.md`.
