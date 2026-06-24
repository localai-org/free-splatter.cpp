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
