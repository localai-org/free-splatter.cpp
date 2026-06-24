// free_splatter-cli: load a model, run inference over N views, optionally dump
// per-layer taps for parity checking against the PyTorch reference.
//
//   free_splatter-cli [--device DEV] [--dump-taps DIR] [--out OUT.f32]
//                     MODEL.gguf [IMAGES.f32]
//
// IMAGES.f32 is raw little-endian float32, view-major NCHW, [0,1]. The view
// count is inferred from the file size and the model geometry.
#include "free_splatter.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [--device DEV] [--dump-taps DIR] [--out OUT.f32] MODEL.gguf [IMAGES.f32]\n",
        argv0);
    return 2;
}

int main(int argc, char ** argv) {
    const char * device    = nullptr;
    const char * taps_dir  = nullptr;
    const char * out_path  = nullptr;
    const char * model     = nullptr;
    const char * images    = nullptr;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--device" && i + 1 < argc)        device   = argv[++i];
        else if (a == "--dump-taps" && i + 1 < argc) taps_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc)       out_path = argv[++i];
        else if (a == "-h" || a == "--help")         return usage(argv[0]);
        else if (!model)                             model    = argv[i];
        else if (!images)                            images   = argv[i];
        else                                         return usage(argv[0]);
    }
    if (!model) return usage(argv[0]);

    free_splatter_options * opts = free_splatter_options_new();
    if (device)   free_splatter_options_set_device(opts, device);
    if (taps_dir) free_splatter_options_set_dump_taps_dir(opts, taps_dir);

    free_splatter_ctx * ctx = free_splatter_load(model, opts);
    free_splatter_options_free(opts);
    if (!ctx) { std::fprintf(stderr, "load: out of memory\n"); return 1; }
    if (const char * err = free_splatter_last_error(ctx)) {
        std::fprintf(stderr, "load failed: %s\n", err);
        free_splatter_free(ctx);
        return 1;
    }

    free_splatter_geometry geo;
    free_splatter_geometry_of(ctx, &geo);
    std::printf("model: %dx%d, in=%d, gaussian_channels=%d\n",
                geo.image_width, geo.image_height, geo.in_channels, geo.gaussian_channels);

    if (!images) { free_splatter_free(ctx); return 0; }

    std::ifstream f(images, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", images); free_splatter_free(ctx); return 1; }
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<float> buf(bytes / sizeof(float));
    f.read((char *) buf.data(), bytes);

    const int64_t per_view = (int64_t) geo.in_channels * geo.image_height * geo.image_width;
    if (per_view == 0 || buf.size() % per_view != 0) {
        std::fprintf(stderr, "image file size is not a whole number of views\n");
        free_splatter_free(ctx);
        return 1;
    }
    const int32_t n_views = (int32_t) (buf.size() / per_view);

    float * out = nullptr;
    size_t  n_out = 0;
    if (free_splatter_run(ctx, buf.data(), n_views, geo.image_height, geo.image_width,
                          &out, &n_out) != 0) {
        std::fprintf(stderr, "run failed: %s\n", free_splatter_last_error(ctx));
        free_splatter_free(ctx);
        return 1;
    }
    std::printf("ran %d views -> %zu gaussian floats\n", n_views, n_out);

    if (out_path) {
        std::ofstream o(out_path, std::ios::binary);
        o.write((const char *) out, (std::streamsize) n_out * sizeof(float));
    }
    free_splatter_buf_free(out);
    free_splatter_free(ctx);
    return 0;
}
