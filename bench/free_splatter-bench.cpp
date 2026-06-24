// free_splatter-bench: load a model and time end-to-end forward passes.
// (Forward is built incrementally across M1-M3; until then this reports load +
// geometry so the target links and the harness exists.)
#include "free_splatter.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [--device DEV]\n", argv[0]);
        return 2;
    }
    const char * device = nullptr;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--device" && i + 1 < argc) device = argv[++i];
    }

    free_splatter_options * opts = free_splatter_options_new();
    if (device) free_splatter_options_set_device(opts, device);
    free_splatter_ctx * ctx = free_splatter_load(argv[1], opts);
    free_splatter_options_free(opts);

    if (!ctx || free_splatter_last_error(ctx)) {
        std::fprintf(stderr, "load failed: %s\n",
                     ctx ? free_splatter_last_error(ctx) : "oom");
        free_splatter_free(ctx);
        return 1;
    }
    free_splatter_geometry geo;
    free_splatter_geometry_of(ctx, &geo);
    std::printf("loaded: %dx%d gaussian_channels=%d\n",
                geo.image_width, geo.image_height, geo.gaussian_channels);
    free_splatter_free(ctx);
    return 0;
}
