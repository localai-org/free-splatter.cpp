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

# Per-layer parity vs PyTorch lands in M1 (needs converted weights + fixtures):
#   free_splatter_add_test(test_parity)
#   set_tests_properties(test_parity PROPERTIES LABELS "model"
#       ENVIRONMENT "FREE_SPLATTER_FIXTURES=${CMAKE_SOURCE_DIR}/tests/fixtures/hf")
