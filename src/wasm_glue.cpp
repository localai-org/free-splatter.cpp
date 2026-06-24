// WASM translation unit for the free_splatter_wasm target. The public C API
// (include/free_splatter.h, implemented in free_splatter.cpp) is exported to
// JavaScript via the -sEXPORTED_FUNCTIONS list in CMakeLists.txt; Emscripten
// needs a TU for the executable target, and this keeps the export surface in
// one obvious place. No extra glue is required: callers use ccall/cwrap over the
// flat C functions, passing image data through HEAPF32.
//
// (If a richer JS ergonomics layer is ever wanted, add embind bindings here.)
