#include "free_splatter.h"

#include "image.h"
#include "model.h"
#include "options.h"
#include "pose.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

// ---- downstream pose recovery + accumulation -------------------------------

namespace {

// De-interleave per-view contiguous points (3*P) + opacity (P) from a
// [n_views,H,W,gc] engine buffer. Returns per-view base pointers into `store`.
void deinterleave(const float * g, int nv, int P, int gc,
                  std::vector<std::vector<float>> & pts,
                  std::vector<std::vector<float>> & ops,
                  std::vector<const float *> & pptr,
                  std::vector<const float *> & optr) {
    pts.assign(nv, {}); ops.assign(nv, {});
    pptr.resize(nv); optr.resize(nv);
    for (int v = 0; v < nv; v++) {
        pts[v].resize((size_t) 3 * P); ops[v].resize(P);
        for (int i = 0; i < P; i++) {
            const float * a = &g[(size_t) (v * P + i) * gc];
            pts[v][3*i] = a[0]; pts[v][3*i+1] = a[1]; pts[v][3*i+2] = a[2];
            ops[v][i] = a[15];
        }
        pptr[v] = pts[v].data(); optr[v] = ops[v].data();
    }
}

// Copy a pose vector's cloud into a malloc'd C array (ABI-stable layout).
int emit_points(const std::vector<free_splatter::pose::AccumPoint> & src,
                free_splatter_point ** out, size_t * n_out) {
    *out = (free_splatter_point *) malloc(src.size() * sizeof(free_splatter_point));
    if (!*out && !src.empty()) return -1;
    for (size_t i = 0; i < src.size(); i++) {
        (*out)[i].x = src[i].x; (*out)[i].y = src[i].y; (*out)[i].z = src[i].z;
        (*out)[i].r = src[i].r; (*out)[i].g = src[i].g; (*out)[i].b = src[i].b;
        (*out)[i].frame = src[i].frame;
    }
    *n_out = src.size();
    return 0;
}

} // namespace

struct free_splatter_accumulator {
    free_splatter::pose::Accumulator acc;
    int gc = 0;
    free_splatter_accumulator(int H, int W, double thr) : acc(H, W, thr) {}
};

int free_splatter_estimate_poses(const float * gaussians, int32_t n_views, int32_t height,
                                 int32_t width, int32_t gaussian_channels,
                                 float opacity_threshold, float * cam2world_out, float * focal_out) {
    if (!gaussians || !cam2world_out || n_views < 1 || height < 1 || width < 1 || gaussian_channels < 16)
        return -1;
    const int P = height * width;
    std::vector<std::vector<float>> pts, ops;
    std::vector<const float *> pptr, optr;
    deinterleave(gaussians, n_views, P, gaussian_channels, pts, ops, pptr, optr);
    free_splatter::pose::PoseResult pr =
        free_splatter::pose::estimate_poses(pptr, optr, height, width, opacity_threshold);
    for (int v = 0; v < n_views; v++)
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
            cam2world_out[v*16 + i*4 + j] = (float) pr.cam2world[v](i, j);
    if (focal_out) *focal_out = (float) pr.focal;
    return 0;
}

free_splatter_accumulator * free_splatter_accumulator_new(int32_t height, int32_t width,
                                                          float opacity_threshold) {
    if (height < 1 || width < 1) return nullptr;
    return new (std::nothrow) free_splatter_accumulator(height, width, opacity_threshold);
}

void free_splatter_accumulator_free(free_splatter_accumulator * acc) { delete acc; }

int free_splatter_accumulator_add_pair(free_splatter_accumulator * acc, const float * gaussians,
                                       int32_t gaussian_channels) {
    if (!acc || !gaussians || gaussian_channels < 16) return -1;
    acc->gc = gaussian_channels;
    acc->acc.add_pair(gaussians, gaussian_channels);
    return 0;
}

int free_splatter_accumulator_frame_count(const free_splatter_accumulator * acc) {
    return acc ? acc->acc.frame_count() : 0;
}

int free_splatter_accumulator_cloud(const free_splatter_accumulator * acc,
                                    free_splatter_point ** out, size_t * n_out) {
    if (out) *out = nullptr;
    if (n_out) *n_out = 0;
    if (!acc || !out || !n_out) return -1;
    return emit_points(acc->acc.cloud(), out, n_out);
}

int free_splatter_accumulator_fuse(const free_splatter_accumulator * acc, float voxel_frac,
                                   int32_t k, free_splatter_point ** out, size_t * n_out) {
    if (out) *out = nullptr;
    if (n_out) *n_out = 0;
    if (!acc || !out || !n_out || k < 1 || !(voxel_frac > 0)) return -1;
    std::vector<free_splatter::pose::AccumPoint> fused;
    free_splatter::pose::consensus_fuse(acc->acc.cloud(), voxel_frac, k, fused);
    return emit_points(fused, out, n_out);
}

int free_splatter_accumulator_camera_path(const free_splatter_accumulator * acc,
                                          float ** out, int32_t * n_frames) {
    if (out) *out = nullptr;
    if (n_frames) *n_frames = 0;
    if (!acc || !out || !n_frames) return -1;
    std::vector<fsla::Mat4> path = acc->acc.camera_path();
    *out = (float *) malloc(path.size() * 16 * sizeof(float));
    if (!*out && !path.empty()) return -1;
    for (size_t f = 0; f < path.size(); f++)
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
            (*out)[f*16 + i*4 + j] = (float) path[f](i, j);
    *n_frames = (int32_t) path.size();
    return 0;
}
