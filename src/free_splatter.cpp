#include "free_splatter.h"

#include "image.h"
#include "model.h"
#include "options.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

// ---- opaque types ----------------------------------------------------------

struct free_splatter_options {
    free_splatter::options o;
};

struct free_splatter_ctx {
    free_splatter::model m;
    free_splatter::options opts;
    std::string error;
};

namespace {

// Streams taps to <dir>/<name>.f32 (raw little-endian) as forward emits them,
// and on finish() writes <dir>/meta.json ({shape:[rows,cols], dtype:"f32"}). The
// model never holds all activations in host memory. Used as the tap_sink.
struct tap_writer {
    std::string dir;
    std::string meta = "{\n  \"taps\": {\n";
    bool        first = true;
    bool        ok = true;
    std::string err;

    explicit tap_writer(const std::string & d) : dir(d) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) { ok = false; err = "could not create tap dir: " + dir; }
    }

    void operator()(const std::string & name, const float * data, int64_t rows, int64_t cols) {
        if (!ok) return;
        std::ofstream f(std::filesystem::path(dir) / (name + ".f32"), std::ios::binary);
        if (!f) { ok = false; err = "could not write tap: " + name; return; }
        f.write((const char *) data, (std::streamsize) rows * cols * (int64_t) sizeof(float));
        if (!first) meta += ",\n";
        first = false;
        meta += "    \"" + name + "\": {\"shape\": [" + std::to_string(rows) + ", " +
                std::to_string(cols) + "], \"dtype\": \"f32\"}";
    }

    bool finish() {
        if (!ok) return false;
        meta += "\n  }\n}\n";
        std::ofstream m(std::filesystem::path(dir) / "meta.json");
        if (!m) { ok = false; err = "could not write meta.json in " + dir; return false; }
        m << meta;
        return true;
    }
};

} // namespace

// ---- ABI -------------------------------------------------------------------

int free_splatter_abi_version(void) { return FREE_SPLATTER_ABI_VERSION; }

// ---- options builder -------------------------------------------------------

free_splatter_options * free_splatter_options_new(void) { return new free_splatter_options(); }
void free_splatter_options_free(free_splatter_options * opts) { delete opts; }

void free_splatter_options_set_device(free_splatter_options * opts, const char * device) {
    if (opts) opts->o.device = device ? device : "";
}
void free_splatter_options_set_threads(free_splatter_options * opts, int n_threads) {
    if (opts) opts->o.n_threads = n_threads;
}
void free_splatter_options_set_dump_taps_dir(free_splatter_options * opts, const char * dir) {
    if (opts) opts->o.dump_taps_dir = dir ? dir : "";
}

// ---- lifecycle -------------------------------------------------------------

free_splatter_ctx * free_splatter_load(const char * gguf_path, const free_splatter_options * opts) {
    if (!gguf_path) return nullptr;
    auto * ctx = new (std::nothrow) free_splatter_ctx();
    if (!ctx) return nullptr;
    if (opts) ctx->opts = opts->o;

    if (!ctx->m.load(gguf_path, ctx->opts.device.empty() ? "cpu" : ctx->opts.device,
                     ctx->opts.n_threads)) {
        ctx->error = ctx->m.error;
    }
    return ctx;
}

void free_splatter_free(free_splatter_ctx * ctx) { delete ctx; }

const char * free_splatter_last_error(const free_splatter_ctx * ctx) {
    if (!ctx) return "NULL context";
    return ctx->error.empty() ? nullptr : ctx->error.c_str();
}

int free_splatter_geometry_of(const free_splatter_ctx * ctx, free_splatter_geometry * out) {
    if (!ctx || !out || !ctx->error.empty()) return -1;
    const free_splatter::hparams & hp = ctx->m.hp();
    out->in_channels       = hp.in_channels;
    out->image_height      = hp.image_size;
    out->image_width       = hp.image_size;
    out->gaussian_channels = hp.gaussian_channels;
    return 0;
}

// ---- inference -------------------------------------------------------------

int free_splatter_run(free_splatter_ctx * ctx, const float * images, int32_t n_views,
                      int32_t height, int32_t width, float ** out, size_t * n_out) {
    if (out)   *out   = nullptr;
    if (n_out) *n_out = 0;
    if (!ctx || !images || !out || !n_out) return -1;
    if (!ctx->error.empty()) return -1;

    std::vector<float> clean;
    if (!free_splatter::ingest_images(ctx->m.hp(), images, n_views, height, width,
                                      clean, ctx->error)) {
        return -1;
    }

    const bool want_taps = !ctx->opts.dump_taps_dir.empty();
    std::vector<float> result;

    if (want_taps) {
        tap_writer writer(ctx->opts.dump_taps_dir);
        if (!writer.ok) { ctx->error = writer.err; return -1; }
        const bool fwd = ctx->m.forward(clean.data(), n_views, result,
            [&writer](const std::string & n, const float * d, int64_t r, int64_t c) {
                writer(n, d, r, c);
            });
        if (!fwd)            { ctx->error = ctx->m.error; return -1; }
        if (!writer.finish()) ctx->error = writer.err;  // surface; result still valid
    } else {
        if (!ctx->m.forward(clean.data(), n_views, result)) {
            ctx->error = ctx->m.error;
            return -1;
        }
    }

    *out = (float *) malloc(result.size() * sizeof(float));
    if (!*out) { ctx->error = "output allocation failed"; return -1; }
    std::memcpy(*out, result.data(), result.size() * sizeof(float));
    *n_out = result.size();
    return 0;
}

void free_splatter_buf_free(void * buf) { free(buf); }
