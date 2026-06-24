{
  description = "free-splatter.cpp — GGML inference for FreeSplatter's NN front half (CPU + Vulkan)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      # numpy only: the scripts that also need `gguf`/`torch` (convert.py,
      # hf_dump.py) run in docker/Dockerfile.cuda alongside the GPU model. The
      # M0 gate synthesizes its test GGUF in C++ (tests/test_loader.cpp), so the
      # build + fast test tier need no Python at all.
      pyEnv = pkgs.python3.withPackages (ps: with ps; [ numpy ]);
    in
    {
      devShells.${system} = {
        # CPU + Vulkan build/test. Configure with the CMake presets:
        #   cmake --preset debug      (ASan/UBSan; the verification preset)
        #   cmake --preset vulkan     (-DFREE_SPLATTER_VULKAN=ON)
        default = pkgs.mkShell {
          packages = [
            pkgs.cmake
            pkgs.ninja
            pkgs.gcc
            pkgs.pkg-config
            pkgs.git
            pyEnv
            # Vulkan backend (used by the `vulkan` preset)
            pkgs.vulkan-loader
            pkgs.vulkan-headers
            pkgs.shaderc
            # headless browser for screenshotting the WebGL demo (web/)
            pkgs.chromium
          ];
          # Let pip-installed PyTorch/numpy wheels (used by scripts/, in a
          # .venv-torch) find their shared libs. The wheels are manylinux and
          # link libstdc++ / libgcc_s / libz / libgomp not on the pure path.
          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib pkgs.zlib
            ]}:''${LD_LIBRARY_PATH:-}"
          '';
        };

        # Fuzzing: clang + libFuzzer + ASan/UBSan. Use:
        #   nix develop .#fuzz && cmake --preset fuzz && cmake --build --preset fuzz
        fuzz = pkgs.mkShell {
          packages = [ pkgs.cmake pkgs.ninja pkgs.clang pkgs.llvm pkgs.pkg-config ];
        };

        # WASM: Emscripten toolchain for the in-browser build. Emscripten's Nix
        # package ships a read-only cache; EM_CACHE must point at a writable copy
        # or the first sysroot/port build fails.
        wasm = pkgs.mkShell {
          packages = [ pkgs.cmake pkgs.ninja pkgs.emscripten pkgs.nodejs pyEnv ];
          shellHook = ''
            export EM_CACHE="''${EM_CACHE:-$PWD/.em-cache}"
            if [ ! -d "$EM_CACHE" ]; then
              mkdir -p "$EM_CACHE"
              cp -r ${pkgs.emscripten}/share/emscripten/cache/. "$EM_CACHE/" 2>/dev/null || true
              chmod -R u+w "$EM_CACHE"
            fi
          '';
        };
      };
    };
}
