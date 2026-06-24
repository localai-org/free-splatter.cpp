// Asset-free: the untrusted image-input validator. These cases double as the
// unit guard for the same code the fuzzer (fuzz/fuzz_image.cpp) hammers.
#include "gguf_loader.h"
#include "image.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

int main() {
    free_splatter::hparams hp;
    hp.image_size  = 512;
    hp.in_channels = 3;
    hp.patch_size  = 8;

    const int H = 512, W = 512, C = 3;
    const size_t per_view = (size_t) C * H * W;

    // valid: 2 views of mid-grey
    {
        std::vector<float> img(2 * per_view, 0.5f);
        std::vector<float> out;
        std::string err;
        CHECK(free_splatter::ingest_images(hp, img.data(), 2, H, W, out, err));
        CHECK(err.empty());
        CHECK(out.size() == 2 * per_view);
    }
    // wrong resolution
    {
        std::vector<float> img((size_t) C * 256 * 256, 0.0f);
        std::vector<float> out; std::string err;
        CHECK(!free_splatter::ingest_images(hp, img.data(), 1, 256, 256, out, err));
        CHECK(!err.empty());
    }
    // non-finite value
    {
        std::vector<float> img(per_view, 0.0f);
        img[123] = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> out; std::string err;
        CHECK(!free_splatter::ingest_images(hp, img.data(), 1, H, W, out, err));
    }
    // infinity too
    {
        std::vector<float> img(per_view, 0.0f);
        img[per_view - 1] = std::numeric_limits<float>::infinity();
        std::vector<float> out; std::string err;
        CHECK(!free_splatter::ingest_images(hp, img.data(), 1, H, W, out, err));
    }
    // zero / negative views
    {
        std::vector<float> img(per_view, 0.0f);
        std::vector<float> out; std::string err;
        CHECK(!free_splatter::ingest_images(hp, img.data(), 0, H, W, out, err));
    }
    // null pointer
    {
        std::vector<float> out; std::string err;
        CHECK(!free_splatter::ingest_images(hp, nullptr, 1, H, W, out, err));
    }

    std::printf(failures ? "test_image: %d FAILURES\n" : "test_image: ok\n", failures);
    return failures ? 1 : 0;
}
