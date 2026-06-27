# Tests needing model assets carry the "model" label and skip (exit 77) when
# FREE_SPLATTER_GGUF_DIR is unset. Fast gate (asset-free): ctest -LE model.
function(free_splatter_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE free_splatter ggml)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

# Asset-free fast tier (no "model" label).
free_splatter_add_test(test_loader)        # GGUF KV/hparams round-trip
free_splatter_add_test(test_graph_blocks)  # golden-op pins (ggml ops we depend on)
free_splatter_add_test(test_image)         # untrusted image-input validation
free_splatter_add_test(test_pose)          # pose math golden tests (focal/align/PnP)

# Per-layer parity vs the float64 PyTorch reference. Labelled "model": runs only
# under `ctest -L model` and self-skips (77) unless FREE_SPLATTER_GGUF and
# FREE_SPLATTER_FIXTURES are set. Needs a converted gguf + scripts/hf_dump.py
# fixtures (see CLAUDE.md). Full-depth is heavy (~15 GB / minutes); set
# FREE_SPLATTER_MAX_BLOCKS for a fast piece-1 + block-0 subset.
free_splatter_add_test(test_parity)
set_tests_properties(test_parity PROPERTIES LABELS "model")
