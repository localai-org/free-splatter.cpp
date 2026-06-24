// libFuzzer harness for the options-builder / device-string surface. Arbitrary
// (possibly non-NUL-terminated -> we copy) byte strings flow into the setters;
// asserts the builder never crashes and frees cleanly. (Full device resolution
// happens in free_splatter_load against a trusted model, not fuzzed here.)
#include "free_splatter.h"

#include <cstdint>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    const std::string s((const char *) data, size);   // bounded copy

    free_splatter_options * o = free_splatter_options_new();
    free_splatter_options_set_device(o, s.c_str());
    free_splatter_options_set_dump_taps_dir(o, s.c_str());
    free_splatter_options_set_threads(o, size ? (int) data[0] - 64 : 0);
    free_splatter_options_set_device(o, nullptr);  // NULL-safety
    free_splatter_options_free(o);
    free_splatter_options_free(nullptr);            // NULL-safe free
    return 0;
}
