cmake_minimum_required(VERSION 3.21)
project(free-splatter C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Backends are opt-in. CPU is always built. Vulkan/CUDA are off by default so a
# plain `cmake` configures on any machine; enable via the presets or -D flags.
option(FREE_SPLATTER_VULKAN "Enable the ggml Vulkan backend" OFF)
option(FREE_SPLATTER_CUDA   "Enable the ggml CUDA backend"   OFF)
option(FREE_SPLATTER_FUZZ   "Build libFuzzer targets (clang only)" OFF)
# OFF lets a downstream consumer add_subdirectory() this project and link the
# free_splatter lib without pulling in the cli/bench/test/fuzz tree.
option(FREE_SPLATTER_BUILD_TOOLS  "Build free_splatter-cli and free_splatter-bench" ON)
option(FREE_SPLATTER_BUILD_TESTS  "Build the test suite" ON)
option(FREE_SPLATTER_BUILD_SHARED "Build libfree_splatter as a shared library (FFI/WASM)" OFF)
set(FREE_SPLATTER_FUZZ_SANITIZERS "fuzzer,address,undefined" CACHE STRING
    "Sanitizers for fuzz builds")

if (FREE_SPLATTER_VULKAN)
    set(GGML_VULKAN ON)
endif()
if (FREE_SPLATTER_CUDA)
    set(GGML_CUDA ON)
    # ggml's VMM pool uses the CUDA *driver* API (cuMem*), linking libcuda.so.1 —
    # present only at runtime via the GPU driver. A GPU-less build container then
    # cannot link libggml-cuda.so. We never depend on VMM, so drop it.
    set(GGML_CUDA_NO_VMM ON)
endif()
set(GGML_BUILD_TESTS    OFF)
set(GGML_BUILD_EXAMPLES OFF)

# tinyBLAS (llamafile) cache-blocked GEMM for the CPU backend. ggml ships it OFF
# by default, leaving mul_mat on the per-row ggml_vec_dot fallback; turning it on
# gives a cache-blocked F16/F32 GEMM that runs the transformer Linears ~1.6x
# faster (53%->32% of CPU time at S=8192). It is numerically a no-op (same f32
# accumulation) and is parity-verified layer-by-layer against the f64 reference
# at the strict CPU-f32 gate. Set as the ggml *default* (not forced) so it covers
# every preset yet still yields to an explicit -DGGML_LLAMAFILE=OFF.
set(GGML_LLAMAFILE_DEFAULT ON)

# Portable, fast CPU build by DEFAULT -- including a preset-less `cmake -B build`.
# Build every ggml CPU ISA variant (SSE..AVX-512) as runtime-dispatched dynamic
# backends and pick the best at load time, instead of a single -march=native
# target. Native is fragile across machines and, under Nix, NIX_ENFORCE_NO_NATIVE
# strips it -- silently leaving a ~13x slower SCALAR build. The presets used to be
# the only portable path; this makes the bare build "just work" too.
#
# Scope: only a plain CPU Release/RelWithDebInfo (or the typeless bare build).
# Skipped for GPU/fuzz/debug builds (kept simple and quick to compile), and each
# knob is guarded so an explicit -D or a preset cacheVariable still wins:
#   - release-native (GGML_NATIVE=ON) keeps its single native target -- the
#     `AND NOT GGML_NATIVE` guards avoid ggml's NATIVE-vs-BACKEND_DL conflict;
#   - release / release-portable already set these ON (DEFINED -> left as-is).
# (GGML_BACKEND_DL's BUILD_SHARED_LIBS requirement is met by ggml's own
# default-ON shared-libs option on Linux.)
if ((NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        AND NOT FREE_SPLATTER_VULKAN AND NOT FREE_SPLATTER_CUDA AND NOT FREE_SPLATTER_FUZZ)
    # CACHE (no FORCE): visible/overridable, and an explicit -D or preset value
    # already in the cache wins. ggml's own option() then honors these.
    set(GGML_NATIVE OFF CACHE BOOL "ggml: optimize for the build CPU (off => portable multi-variant build)")
    if (NOT GGML_NATIVE)
        set(GGML_BACKEND_DL       ON CACHE BOOL "ggml: build backends as dynamic libs (enables runtime CPU ISA dispatch)")
        set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "ggml: build every CPU ISA variant; best is picked at load time")
    endif()
endif()

# Co-locate our executables with ggml's dynamically-loaded backend .so files. A
# GGML_BACKEND_DL build emits libggml-*.so to <build>/bin and loads them by path
# at runtime; ggml sets CMAKE_RUNTIME_OUTPUT_DIRECTORY for that, but only in its
# own (subdirectory) scope -- so without this our cli/bench land in the build
# root, away from the backends, and startup fails with "CPU backend init failed".
# The presets set this explicitly; an explicit value still wins (if-not-set).
# Gated on GGML_BACKEND_DL: only a dynamic-backend build needs the co-location,
# so static builds (vulkan/debug/native presets) keep their default binary
# location and existing script paths (e.g. bench.sh's build/vulkan/...) hold.
if (GGML_BACKEND_DL AND NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# --- Optional custom-op patch slot (LocalVQE pattern) -----------------------
# If pieces 1-3 ever need a ggml op not upstream, drop a patch in patches/ and it
# is applied to the submodule working tree at configure time, idempotently.
file(GLOB FREE_SPLATTER_GGML_PATCHES "${CMAKE_SOURCE_DIR}/patches/ggml-*.patch")
foreach(patch ${FREE_SPLATTER_GGML_PATCHES})
    execute_process(
        COMMAND git -C ${CMAKE_SOURCE_DIR}/ggml apply --check --reverse ${patch}
        RESULT_VARIABLE already_applied OUTPUT_QUIET ERROR_QUIET)
    if (NOT already_applied EQUAL 0)
        message(STATUS "Applying ggml patch: ${patch}")
        execute_process(COMMAND git -C ${CMAKE_SOURCE_DIR}/ggml apply ${patch})
    endif()
endforeach()

if (NOT EXISTS "${CMAKE_SOURCE_DIR}/ggml/include/ggml.h")
    message(FATAL_ERROR
        "ggml submodule missing. Run: git submodule update --init --recursive")
endif()
add_subdirectory(ggml)

if (FREE_SPLATTER_FUZZ)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "FREE_SPLATTER_FUZZ requires clang (libFuzzer)")
    endif()
    # Instrument the whole tree for coverage (fuzzer-no-link adds NO main) plus
    # ASan/UBSan. Only the fuzz executables link libFuzzer's main (-fsanitize=
    # fuzzer, added per-target below) -- applying it globally would inject a
    # second main into the cli/bench/tests and fail to link.
    add_compile_options(-fsanitize=fuzzer-no-link,address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

if (FREE_SPLATTER_BUILD_SHARED)
    set(_fs_libtype SHARED)
else()
    set(_fs_libtype STATIC)
endif()

add_library(free_splatter ${_fs_libtype}
    src/backend.cpp
    src/gguf_loader.cpp
    src/model.cpp
    src/image.cpp
    src/pose.cpp
    src/free_splatter.cpp
)
target_include_directories(free_splatter PUBLIC include)
target_link_libraries(free_splatter PUBLIC ggml)
target_compile_options(free_splatter PRIVATE -Wall -Wextra)
set_target_properties(free_splatter PROPERTIES C_VISIBILITY_PRESET hidden
                                               CXX_VISIBILITY_PRESET hidden)

if (FREE_SPLATTER_BUILD_TOOLS AND NOT EMSCRIPTEN)
    add_executable(free_splatter-cli tools/free_splatter-cli.cpp tools/stb_impl.cpp)
    target_link_libraries(free_splatter-cli PRIVATE free_splatter)
    target_include_directories(free_splatter-cli PRIVATE src ${CMAKE_SOURCE_DIR}/third_party/stb)
    target_compile_options(free_splatter-cli PRIVATE -Wall -Wextra)
    # the vendored stb headers are warning-noisy; build that TU quietly
    set_source_files_properties(tools/stb_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")

    add_executable(free_splatter-bench bench/free_splatter-bench.cpp)
    target_link_libraries(free_splatter-bench PRIVATE free_splatter)
    target_include_directories(free_splatter-bench PRIVATE src)
endif()

if (FREE_SPLATTER_FUZZ)
    foreach(target fuzz_image fuzz_options)
        add_executable(${target} fuzz/${target}.cpp)
        target_link_libraries(${target} PRIVATE free_splatter)
        target_include_directories(${target} PRIVATE src)
        target_link_options(${target} PRIVATE -fsanitize=fuzzer)  # libFuzzer main
    endforeach()
endif()

# --- WASM (Emscripten) -------------------------------------------------------
# CPU backend only; exports the flat C API for purego-style ccall/cwrap in JS.
if (EMSCRIPTEN)
    add_executable(free_splatter_wasm src/wasm_glue.cpp)
    target_link_libraries(free_splatter_wasm PRIVATE free_splatter)
    target_link_options(free_splatter_wasm PRIVATE
        "-sMODULARIZE=1" "-sEXPORT_NAME=createFreeSplatterModule"
        "-sALLOW_MEMORY_GROWTH=1" "--no-entry"
        "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF32']"
        "-sEXPORTED_FUNCTIONS=['_free_splatter_abi_version','_free_splatter_load','_free_splatter_free','_free_splatter_run','_free_splatter_last_error','_free_splatter_geometry_of','_free_splatter_buf_free','_free_splatter_options_new','_free_splatter_options_free','_free_splatter_options_set_device','_free_splatter_options_set_threads','_free_splatter_options_set_dump_taps_dir','_malloc','_free']")
endif()

if (FREE_SPLATTER_BUILD_TESTS AND NOT EMSCRIPTEN)
    enable_testing()
    add_subdirectory(tests)
endif()
