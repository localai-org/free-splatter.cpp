// libFuzzer harness for the UNTRUSTED image-input surface (src/image.cpp).
// ingest_images is the boundary all caller-supplied pixels cross; this hammers
// it with arbitrary view counts, resolutions and pixel bit-patterns (NaN/Inf
// included) and asserts it never crashes, reads OOB, or leaves `out` populated
// on a rejected input. The GGUF model file is trusted and NOT fuzzed.
#include "gguf_loader.h"
#include "image.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    if (size < 4) return 0;

    free_splatter::hparams hp;
    hp.image_size  = 512;
    hp.in_channels = 3;
    hp.patch_size  = 8;

    // Drive shape from the input so the validator's reject paths and the one
    // accept path (512x512) are all exercised, including the n_views cap.
    const int32_t n_views = (int32_t) (data[0] % 70) - 2;          // -2..67 (neg/zero/over-cap)
    const int32_t hw_sel  = data[1] % 3;
    const int32_t H = hw_sel == 0 ? 512 : (hw_sel == 1 ? 256 : (int32_t) (data[2] * 4));
    const int32_t W = hw_sel == 0 ? 512 : 511;

    // Build a correctly-sized buffer (the API contract) filled from the fuzz
    // bytes reinterpreted as floats, so NaN/Inf/denormals all appear.
    std::vector<float> img;
    if (n_views > 0 && n_views <= 64 && H == 512 && W == 512) {
        const size_t want = (size_t) n_views * hp.in_channels * H * W;
        img.resize(want);
        std::memcpy(img.data(), data, std::min(want * sizeof(float), size));
        for (size_t i = (size > 4 ? size : 4); i < want; i++) {
            // tile remaining from the input bytes (cheap, deterministic)
            ((uint8_t *) img.data())[i] = data[i % size];
        }
    } else {
        // reject-path inputs don't need a full buffer; a small one suffices since
        // ingest_images checks shape before touching pixels.
        img.assign(16, 0.0f);
    }

    std::vector<float> out;
    std::string err;
    const bool ok = free_splatter::ingest_images(hp, img.data(), n_views, H, W, out, err);
    // Invariants: on reject, no output and a message; on accept, exact size.
    if (!ok) {
        if (!out.empty() || err.empty()) __builtin_trap();
    } else {
        if (out.size() != (size_t) n_views * hp.in_channels * H * W) __builtin_trap();
    }
    return 0;
}
