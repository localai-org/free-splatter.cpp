// libFuzzer harness for the UNTRUSTED image-FILE decode path: arbitrary bytes ->
// stb_image -> the CLI's center-crop + resize -> [0,1] CHW. This is the actual
// surface a user photo crosses in `free_splatter-cli` / the demo. stb_image is a
// vendored THIRD-PARTY decoder; per CLAUDE.md we fuzz the boundary and, where stb
// itself trips a sanitizer on malformed input, GUARD it rather than patch stb.
#include "stb_image.h"
#include "stb_image_resize2.h"

#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    int w = 0, h = 0, n = 0;
    // Decode arbitrary bytes as an image, forcing RGB (the CLI's contract).
    unsigned char * px = stbi_load_from_memory(data, (int) size, &w, &h, &n, 3);
    if (!px) return 0;                      // rejected -> nothing to do

    // Guard against a decoded size that would overflow the crop/resize math (stb
    // can report large dims on crafted headers); the CLI works on sane photos.
    if ((int64_t) w * h > 64ll * 1024 * 1024 || w <= 0 || h <= 0) { stbi_image_free(px); return 0; }

    // center-crop to a square, then resize to a small size (mirror the CLI).
    const int s = w < h ? w : h, left = (w - s) / 2, top = (h - s) / 2;
    std::vector<unsigned char> sq((size_t) s * s * 3);
    for (int y = 0; y < s; y++)
        std::memcpy(&sq[(size_t) y * s * 3], &px[((size_t)(top + y) * w + left) * 3], (size_t) s * 3);
    stbi_image_free(px);

    const int outsz = 32;
    std::vector<unsigned char> rz((size_t) outsz * outsz * 3);
    stbir_resize_uint8_linear(sq.data(), s, s, 0, rz.data(), outsz, outsz, 0, STBIR_RGB);

    std::vector<float> chw((size_t) 3 * outsz * outsz);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < outsz * outsz; i++)
            chw[(size_t) c * outsz * outsz + i] = rz[(size_t) i * 3 + c] / 255.0f;
    return 0;
}
