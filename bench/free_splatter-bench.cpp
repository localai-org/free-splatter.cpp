// free_splatter-bench: time end-to-end forward passes of the engine.
//
// Loads a model, builds an N-view input (a deterministic synthetic batch, or a
// raw view-major NCHW .f32 file via --input), then runs free_splatter_run a few
// times and reports per-iteration latency (min/median/mean/max) and throughput.
// Load time is reported separately and excluded from the forward timings.
//
//   free_splatter-bench [--device DEV] [--views N] [--input FILE.f32]
//                       [--iters K] [--warmup W] [--threads T] MODEL.gguf
//
// The last line is machine-parseable (scripts/bench.sh reads it):
//   RESULT engine device=DEV views=N gc=23 load_ms=.. min_ms=.. median_ms=.. \
//          mean_ms=.. max_ms=.. views_per_s=..
#include "free_splatter.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static int usage(const char * a0) {
    std::fprintf(stderr,
        "usage: %s [--device DEV] [--views N] [--input FILE.f32]\n"
        "          [--iters K] [--warmup W] [--threads T] MODEL.gguf\n", a0);
    return 2;
}

int main(int argc, char ** argv) {
    const char * device = nullptr, * model = nullptr, * input = nullptr;
    int views = 2, iters = 10, warmup = 2, threads = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--device" && i+1 < argc)  device  = argv[++i];
        else if (a == "--views"  && i+1 < argc)  views   = std::atoi(argv[++i]);
        else if (a == "--input"  && i+1 < argc)  input   = argv[++i];
        else if (a == "--iters"  && i+1 < argc)  iters   = std::atoi(argv[++i]);
        else if (a == "--warmup" && i+1 < argc)  warmup  = std::atoi(argv[++i]);
        else if (a == "--threads" && i+1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help")     return usage(argv[0]);
        else if (!model)                         model   = argv[i];
        else                                     return usage(argv[0]);
    }
    if (!model || iters < 1 || views < 1) return usage(argv[0]);

    free_splatter_options * opts = free_splatter_options_new();
    if (device)      free_splatter_options_set_device(opts, device);
    if (threads > 0) free_splatter_options_set_threads(opts, threads);

    const auto t_load0 = clk::now();
    free_splatter_ctx * ctx = free_splatter_load(model, opts);
    const double load_ms = ms_since(t_load0);
    free_splatter_options_free(opts);
    if (!ctx) { std::fprintf(stderr, "load: out of memory\n"); return 1; }
    if (const char * err = free_splatter_last_error(ctx)) {
        std::fprintf(stderr, "load failed: %s\n", err); free_splatter_free(ctx); return 1;
    }

    free_splatter_geometry geo;
    free_splatter_geometry_of(ctx, &geo);
    const int64_t per_view = (int64_t) geo.in_channels * geo.image_height * geo.image_width;

    // Build the input: a raw .f32 (its size fixes the view count), or a
    // deterministic synthetic batch in [0,1]. Values don't affect timing; an
    // LCG keeps it reproducible and free of denormals.
    std::vector<float> buf;
    if (input) {
        std::ifstream f(input, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", input); free_splatter_free(ctx); return 1; }
        const std::streamsize bytes = f.tellg(); f.seekg(0);
        buf.resize(bytes / sizeof(float)); f.read((char *) buf.data(), bytes);
        if (per_view == 0 || buf.size() % per_view != 0) {
            std::fprintf(stderr, "input size is not a whole number of views\n"); free_splatter_free(ctx); return 1; }
        views = (int) (buf.size() / per_view);
    } else {
        buf.resize((size_t) views * per_view);
        uint32_t s = 0x9e3779b9u;
        for (float & v : buf) { s = s * 1664525u + 1013904223u; v = (s >> 8) * (1.0f / 16777216.0f); }
    }

    std::printf("model: %s  device=%s  %dx%d  in=%d  gc=%d  views=%d  S=%lld  load=%.1f ms\n",
                model, device ? device : "cpu", geo.image_width, geo.image_height,
                geo.in_channels, geo.gaussian_channels, views,
                (long long) ((int64_t) views * (geo.image_height/8) * (geo.image_width/8)), load_ms);

    auto once = [&](double * out_ms) -> bool {
        float * out = nullptr; size_t n_out = 0;
        const auto t0 = clk::now();
        const int rc = free_splatter_run(ctx, buf.data(), (int32_t) views,
                                         geo.image_height, geo.image_width, &out, &n_out);
        if (out_ms) *out_ms = ms_since(t0);
        if (rc != 0) { std::fprintf(stderr, "run failed: %s\n", free_splatter_last_error(ctx)); return false; }
        free_splatter_buf_free(out);
        return true;
    };

    for (int i = 0; i < warmup; i++) { if (!once(nullptr)) { free_splatter_free(ctx); return 1; } }

    std::vector<double> t(iters);
    for (int i = 0; i < iters; i++) {
        if (!once(&t[i])) { free_splatter_free(ctx); return 1; }
        std::printf("  iter %2d: %8.1f ms\n", i, t[i]);
    }
    free_splatter_free(ctx);

    std::vector<double> sorted = t;
    std::sort(sorted.begin(), sorted.end());
    const double mn = sorted.front(), mx = sorted.back();
    const double med = sorted[iters / 2];
    double sum = 0; for (double v : t) sum += v;
    const double mean = sum / iters;
    const double vps = views / (med / 1000.0);

    std::printf("\nsummary: min=%.1f  median=%.1f  mean=%.1f  max=%.1f ms   "
                "(%.2f views/s, %.2f scenes/s)\n", mn, med, mean, mx, vps, 1000.0 / med);
    std::printf("RESULT engine device=%s views=%d gc=%d load_ms=%.1f min_ms=%.1f "
                "median_ms=%.1f mean_ms=%.1f max_ms=%.1f views_per_s=%.3f\n",
                device ? device : "cpu", views, geo.gaussian_channels, load_ms,
                mn, med, mean, mx, vps);
    return 0;
}
