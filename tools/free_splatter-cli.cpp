// free_splatter-cli: load a model, run inference over N views, and write the
// gaussians as raw floats (--out) and/or an antimatter15 .splat (--splat) for
// the web viewer. Inputs are either ONE raw .f32 file (view-major NCHW, [0,1])
// or several image files (jpg/png/...), each decoded, center-cropped to a
// square, and resized to the model resolution.
//
//   free_splatter-cli [--device DEV] [--splat OUT.splat] [--out OUT.f32]
//                     [--opacity-threshold T] [--max-splats N] [--dump-taps DIR]
//                     MODEL.gguf  (IMAGES... | INPUT.f32)
#include "free_splatter.h"

#include "stb_image.h"          // implementation in tools/stb_impl.cpp
#include "stb_image_resize2.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool ends_with(const std::string & s, const char * suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// Decode an image and preprocess like FreeSplatter's scene path: center-crop to
// a square, resize to size x size, scale to [0,1], lay out CHW. Appends to `out`.
static bool load_image_chw(const char * path, int size, std::vector<float> & out) {
    int w, h, n;
    unsigned char * px = stbi_load(path, &w, &h, &n, 3);   // force RGB
    if (!px) { std::fprintf(stderr, "decode failed: %s (%s)\n", path, stbi_failure_reason()); return false; }

    const int s = std::min(w, h), left = (w - s) / 2, top = (h - s) / 2;
    std::vector<unsigned char> sq((size_t) s * s * 3);
    for (int y = 0; y < s; y++)
        std::memcpy(&sq[(size_t) y * s * 3], &px[((size_t)(top + y) * w + left) * 3], (size_t) s * 3);
    stbi_image_free(px);

    std::vector<unsigned char> rz((size_t) size * size * 3);
    stbir_resize_uint8_linear(sq.data(), s, s, 0, rz.data(), size, size, 0, STBIR_RGB);

    const size_t base = out.size();
    out.resize(base + (size_t) 3 * size * size);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < size * size; i++)
            out[base + (size_t) c * size * size + i] = rz[(size_t) i * 3 + c] / 255.0f;
    return true;
}

// Convert the engine's activated gaussians [n*gc] to an antimatter15 .splat:
// 32 bytes/splat = pos(3 f32), scale(3 f32), rgba(4 u8), rot(4 u8: w,x,y,z).
// Prunes opacity <= threshold, sorts by importance (opacity*volume), caps to
// max_splats (0 = all).
static bool write_splat(const float * g, size_t n, int gc, float opac_thr,
                        size_t max_splats, const char * path) {
    const double C0 = 0.28209479177387814;
    std::vector<std::pair<float, size_t>> keep;   // (importance, index)
    keep.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const float op = g[i * gc + 15];
        if (op <= opac_thr) continue;
        const float vol = std::max(g[i*gc+16],1e-9f) * std::max(g[i*gc+17],1e-9f) * std::max(g[i*gc+18],1e-9f);
        keep.push_back({ op * vol, i });
    }
    std::sort(keep.begin(), keep.end(), [](auto & a, auto & b) { return a.first > b.first; });
    const size_t m = (max_splats > 0) ? std::min(max_splats, keep.size()) : keep.size();

    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path); return false; }
    auto u8 = [](float v) -> unsigned char { float t = v < 0 ? 0 : v > 255 ? 255 : v; return (unsigned char) t; };
    for (size_t k = 0; k < m; k++) {
        const float * x = &g[keep[k].second * gc];
        // FreeSplatter's reference frame is OpenCV (y down, z forward); convert
        // to the viewer's OpenGL convention (y up) via a 180deg rotation about X
        // = diag(1,-1,-1): position.yz negate, quaternion (w,x,y,z)->(-x,w,-z,y).
        float pos[3]   = { x[0], -x[1], -x[2] };
        float scale[3] = { x[16], x[17], x[18] };
        unsigned char rgba[4], rot[4];
        for (int c = 0; c < 3; c++) { float v = 0.5f + (float) C0 * x[3+c]; rgba[c] = u8((v<0?0:v>1?1:v) * 255.0f); }
        rgba[3] = u8(std::min(std::max(x[15], 0.0f), 1.0f) * 255.0f);
        float q[4] = { -x[20], x[19], -x[22], x[21] };
        float nrm = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]) + 1e-12f;
        for (int c = 0; c < 4; c++) rot[c] = u8(q[c]/nrm * 128.0f + 128.0f);
        f.write((const char *) pos, 12);
        f.write((const char *) scale, 12);
        f.write((const char *) rgba, 4);
        f.write((const char *) rot, 4);
    }
    std::printf("wrote %s: %zu splats (pruned/cap of %zu kept)\n", path, m, keep.size());
    return true;
}

// Write an accumulated point cloud (xyz + rgb, no per-point covariance) as an
// antimatter15 .splat, each point a small isotropic gaussian. Caps to max_splats
// by stride-subsampling. scale_frac sets the splat radius as a fraction of the
// cloud's rms extent.
static bool write_cloud_splat(const free_splatter_point * pts, size_t n, size_t max_splats,
                              float scale_frac, const char * path) {
    // rms extent about the centroid -> per-point isotropic scale
    double m[3] = {0,0,0};
    for (size_t i = 0; i < n; i++) { m[0]+=pts[i].x; m[1]+=pts[i].y; m[2]+=pts[i].z; }
    if (n) { m[0]/=n; m[1]/=n; m[2]/=n; }
    double ss = 0;
    for (size_t i = 0; i < n; i++) { const double dx=pts[i].x-m[0], dy=pts[i].y-m[1], dz=pts[i].z-m[2]; ss += dx*dx+dy*dy+dz*dz; }
    const float ext = (float) std::sqrt(ss / (n ? n : 1));
    const float scale = std::max(scale_frac * ext, 1e-6f);

    const size_t stride = (max_splats > 0 && n > max_splats) ? (n + max_splats - 1) / max_splats : 1;
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path); return false; }
    auto u8 = [](float v) -> unsigned char { float t = v < 0 ? 0 : v > 255 ? 255 : v; return (unsigned char) t; };
    size_t written = 0;
    for (size_t i = 0; i < n; i += stride) {
        // OpenCV (y down, z forward) -> viewer OpenGL (y up): negate y,z.
        float pos[3]   = { pts[i].x, -pts[i].y, -pts[i].z };
        float sc[3]    = { scale, scale, scale };
        unsigned char rgba[4] = { u8(pts[i].r*255.0f), u8(pts[i].g*255.0f), u8(pts[i].b*255.0f), 255 };
        unsigned char rot[4]  = { 255, 128, 128, 128 };   // identity quaternion (w,x,y,z)
        f.write((const char *) pos, 12);
        f.write((const char *) sc, 12);
        f.write((const char *) rgba, 4);
        f.write((const char *) rot, 4);
        written++;
    }
    std::printf("wrote %s: %zu splats (of %zu cloud points)\n", path, written, n);
    return true;
}

static int usage(const char * a0) {
    std::fprintf(stderr,
        "usage: %s [--device DEV] [--splat OUT.splat] [--out OUT.f32]\n"
        "          [--opacity-threshold T] [--max-splats N] [--dump-taps DIR]\n"
        "          MODEL.gguf (IMAGES... | INPUT.f32)\n"
        "\n"
        "accumulate mode (>=2 images -> one world from the photo stream):\n"
        "  %s --accumulate [--splat-prefix P] [--fuse] [--voxel V] [--fuse-k K]\n"
        "     [--splat-scale S] [--max-splats N] MODEL.gguf IMG0 IMG1 IMG2 ...\n"
        "  runs the engine on each consecutive pair, chains the runs (Sim(3)) into\n"
        "  one accumulating cloud; writes P_<nframes>.splat after each pair (the\n"
        "  evolving reconstruction) and, with --fuse, a consensus-fused P_fused.splat.\n",
        a0, a0);
    return 2;
}

int main(int argc, char ** argv) {
    const char * device = nullptr, * taps_dir = nullptr, * out_path = nullptr, * splat_path = nullptr;
    const char * model = nullptr, * splat_prefix = nullptr;
    float opac_thr = 5e-3f;
    long  max_splats = 0;
    bool  accumulate = false, fuse = false;
    float voxel = 0.02f, splat_scale = 0.0015f;
    int   fuse_k = 2;
    std::vector<std::string> inputs;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--device" && i+1 < argc)             device = argv[++i];
        else if (a == "--dump-taps" && i+1 < argc)      taps_dir = argv[++i];
        else if (a == "--out" && i+1 < argc)            out_path = argv[++i];
        else if (a == "--splat" && i+1 < argc)          splat_path = argv[++i];
        else if (a == "--opacity-threshold" && i+1<argc) opac_thr = (float) atof(argv[++i]);
        else if (a == "--max-splats" && i+1 < argc)     max_splats = atol(argv[++i]);
        else if (a == "--accumulate")                   accumulate = true;
        else if (a == "--splat-prefix" && i+1 < argc)   splat_prefix = argv[++i];
        else if (a == "--fuse")                         fuse = true;
        else if (a == "--voxel" && i+1 < argc)          voxel = (float) atof(argv[++i]);
        else if (a == "--fuse-k" && i+1 < argc)         fuse_k = atoi(argv[++i]);
        else if (a == "--splat-scale" && i+1 < argc)    splat_scale = (float) atof(argv[++i]);
        else if (a == "-h" || a == "--help")            return usage(argv[0]);
        else if (!model)                                model = argv[i];
        else                                            inputs.push_back(argv[i]);
    }
    if (!model) return usage(argv[0]);

    free_splatter_options * opts = free_splatter_options_new();
    if (device)   free_splatter_options_set_device(opts, device);
    if (taps_dir) free_splatter_options_set_dump_taps_dir(opts, taps_dir);
    free_splatter_ctx * ctx = free_splatter_load(model, opts);
    free_splatter_options_free(opts);
    if (!ctx) { std::fprintf(stderr, "load: out of memory\n"); return 1; }
    if (const char * err = free_splatter_last_error(ctx)) {
        std::fprintf(stderr, "load failed: %s\n", err); free_splatter_free(ctx); return 1;
    }

    free_splatter_geometry geo;
    free_splatter_geometry_of(ctx, &geo);
    std::printf("model: %dx%d, in=%d, gaussian_channels=%d\n",
                geo.image_width, geo.image_height, geo.in_channels, geo.gaussian_channels);
    if (inputs.empty()) { free_splatter_free(ctx); return 0; }

    // ---- accumulate mode: chain consecutive image pairs into one world --------
    if (accumulate) {
        if (inputs.size() < 2) { std::fprintf(stderr, "--accumulate needs >=2 images\n"); free_splatter_free(ctx); return 2; }
        const int sz = geo.image_width;
        // decode every frame once (CHW per view)
        std::vector<std::vector<float>> frames;
        for (const std::string & p : inputs) {
            std::vector<float> img;
            if (!load_image_chw(p.c_str(), sz, img)) { free_splatter_free(ctx); return 1; }
            frames.push_back(std::move(img));
        }
        free_splatter_accumulator * acc = free_splatter_accumulator_new(sz, sz, opac_thr);
        if (!acc) { std::fprintf(stderr, "accumulator alloc failed\n"); free_splatter_free(ctx); return 1; }

        const int64_t per_view = (int64_t) geo.in_channels * sz * sz;
        for (size_t k = 0; k + 1 < frames.size(); k++) {
            std::vector<float> pair(2 * per_view);
            std::memcpy(&pair[0],        frames[k].data(),   per_view * sizeof(float));
            std::memcpy(&pair[per_view], frames[k+1].data(), per_view * sizeof(float));
            float * g = nullptr; size_t ng = 0;
            if (free_splatter_run(ctx, pair.data(), 2, sz, sz, &g, &ng) != 0) {
                std::fprintf(stderr, "run pair %zu failed: %s\n", k, free_splatter_last_error(ctx));
                free_splatter_accumulator_free(acc); free_splatter_free(ctx); return 1;
            }
            free_splatter_accumulator_add_pair(acc, g, geo.gaussian_channels);
            free_splatter_buf_free(g);
            const int nframes = free_splatter_accumulator_frame_count(acc);
            std::printf("pair %zu (%s, %s) -> %d frames\n", k,
                        inputs[k].c_str(), inputs[k+1].c_str(), nframes);
            if (splat_prefix) {
                free_splatter_point * cloud = nullptr; size_t nc = 0;
                free_splatter_accumulator_cloud(acc, &cloud, &nc);
                char path[1024];
                std::snprintf(path, sizeof path, "%s_%d.splat", splat_prefix, nframes);
                write_cloud_splat(cloud, nc, (size_t) max_splats, splat_scale, path);
                free_splatter_buf_free(cloud);
            }
        }
        if (fuse && splat_prefix) {
            free_splatter_point * fc = nullptr; size_t nf = 0;
            free_splatter_accumulator_fuse(acc, voxel, fuse_k, &fc, &nf);
            char path[1024];
            std::snprintf(path, sizeof path, "%s_fused.splat", splat_prefix);
            write_cloud_splat(fc, nf, (size_t) max_splats, splat_scale, path);
            free_splatter_buf_free(fc);
        }
        free_splatter_accumulator_free(acc);
        free_splatter_free(ctx);
        return 0;
    }

    // assemble input: one raw .f32, or several decoded images
    std::vector<float> buf;
    int32_t n_views = 0;
    const int64_t per_view = (int64_t) geo.in_channels * geo.image_height * geo.image_width;
    if (inputs.size() == 1 && ends_with(inputs[0], ".f32")) {
        std::ifstream f(inputs[0], std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", inputs[0].c_str()); free_splatter_free(ctx); return 1; }
        const std::streamsize bytes = f.tellg(); f.seekg(0);
        buf.resize(bytes / sizeof(float)); f.read((char *) buf.data(), bytes);
        if (per_view == 0 || buf.size() % per_view != 0) {
            std::fprintf(stderr, "input size is not a whole number of views\n"); free_splatter_free(ctx); return 1; }
        n_views = (int32_t) (buf.size() / per_view);
    } else {
        for (const std::string & p : inputs)
            if (!load_image_chw(p.c_str(), geo.image_width, buf)) { free_splatter_free(ctx); return 1; }
        n_views = (int32_t) inputs.size();
    }

    float * out = nullptr; size_t n_out = 0;
    if (free_splatter_run(ctx, buf.data(), n_views, geo.image_height, geo.image_width, &out, &n_out) != 0) {
        std::fprintf(stderr, "run failed: %s\n", free_splatter_last_error(ctx)); free_splatter_free(ctx); return 1;
    }
    std::printf("ran %d views -> %zu gaussian floats\n", n_views, n_out);

    if (out_path) { std::ofstream o(out_path, std::ios::binary); o.write((const char *) out, (std::streamsize) n_out * sizeof(float)); }
    if (splat_path) write_splat(out, n_out / geo.gaussian_channels, geo.gaussian_channels, opac_thr, (size_t) max_splats, splat_path);

    free_splatter_buf_free(out);
    free_splatter_free(ctx);
    return 0;
}
