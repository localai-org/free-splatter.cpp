// Persistent ggml backend + graph allocator (parakeet.cpp pattern).
//
// One backend + one gallocr live for the model's lifetime; each forward builds
// its graph in a no_alloc=true context, allocates via the persistent gallocr
// (compute buffer stays alive across calls), writes input data AFTER alloc,
// computes, reads back.
#pragma once

#include <ggml-backend.h>

#include <string>

namespace free_splatter {

struct engine_backend {
    ggml_backend_t be     = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::string    device;   // resolved name
    std::string    error;

    // device_req: "" | "cpu" | "gpu" | "cuda" | "vulkan", optionally ":N" to pick
    // the Nth matching GPU. "gpu" = first GPU of whichever backend is built.
    bool init(const std::string & device_req, int n_threads);
    void release();
    ~engine_backend() { release(); }

    bool is_cpu() const;
};

} // namespace free_splatter
