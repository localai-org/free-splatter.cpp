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
#include "splat.h"              // the single shared .splat record encoder

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
    for (size_t k = 0; k < m; k++) {
        const float * x = &g[keep[k].second * gc];
        const float pos[3]   = { x[0], x[1], x[2] };           // OpenCV; the encoder flips y,z
        const float scale[3] = { x[16], x[17], x[18] };
        const float quat[4]  = { x[19], x[20], x[21], x[22] };  // (w,x,y,z)
        const float rgb[3]   = { 0.5f + (float) C0 * x[3], 0.5f + (float) C0 * x[4], 0.5f + (float) C0 * x[5] };
        unsigned char rec[32];
        free_splatter::encode_splat_record(rec, pos, scale, quat, rgb, x[15]);
        f.write((const char *) rec, 32);
    }
    std::printf("wrote %s: %zu splats (pruned/cap of %zu kept)\n", path, m, keep.size());
    return true;
}

// Write an accumulated gaussian cloud as an antimatter15 .splat, emitting each
// point's true anisotropic scale + rotation (carried through the Sim(3)) AND its
// activated opacity as the splat alpha — so it renders exactly like the single-run
// write_splat (proper alpha blending), not a fully-opaque, swirling, blurry soup.
// scale_mult scales the radii (1.0 = as-predicted). When capping, keep the most
// important splats (opacity * volume), matching write_splat (not a uniform stride).
static bool write_cloud_splat(const free_splatter_point * pts, size_t n, size_t max_splats,
                              float scale_mult, const char * path) {
    auto importance = [&](size_t i) -> double {
        const free_splatter_point & p = pts[i];
        const double vol = (double) std::max(p.sx,1e-9f) * std::max(p.sy,1e-9f) * std::max(p.sz,1e-9f);
        return (double) std::max(p.opacity, 0.0f) * vol;
    };
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = i;
    if (max_splats > 0 && n > max_splats) {
        std::partial_sort(idx.begin(), idx.begin() + max_splats, idx.end(),
                          [&](size_t a, size_t b){ return importance(a) > importance(b); });
        idx.resize(max_splats);
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path); return false; }
    for (size_t i : idx) {
        const free_splatter_point & p = pts[i];
        const float pos[3]   = { p.x, p.y, p.z };               // OpenCV; the encoder flips y,z
        const float scale[3] = { scale_mult*p.sx, scale_mult*p.sy, scale_mult*p.sz };
        const float quat[4]  = { p.qw, p.qx, p.qy, p.qz };      // (w,x,y,z)
        const float rgb[3]   = { p.r, p.g, p.b };
        unsigned char rec[32];
        free_splatter::encode_splat_record(rec, pos, scale, quat, rgb, p.opacity);
        f.write((const char *) rec, 32);
    }
    std::printf("wrote %s: %zu splats (of %zu cloud points)\n", path, idx.size(), n);
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
    bool  accumulate = false, fuse = false, fuse_keep = true;  // dense (kept) by default
    float voxel = 0.02f, splat_scale = 1.0f;   // multiplier on the predicted gaussian radii
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
        else if (a == "--fuse-mode" && i+1 < argc)      fuse_keep = std::string(argv[++i]) != "averaged";
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

    // ---- accumulate mode: chain a photo stream into one world -----------------
    // Two input forms: a stream of IMAGES (the engine runs on each consecutive
    // pair), or pre-computed `.f32` pair dumps (each a [2,H,W,gc] engine output —
    // the engine is skipped, for fast re-bakes / fusion sweeps off cached runs).
    if (accumulate) {
        const int sz = geo.image_width, gc = geo.gaussian_channels;
        const bool dump_mode = std::all_of(inputs.begin(), inputs.end(),
                                           [](const std::string & s){ return ends_with(s, ".f32"); });
        if (!dump_mode && inputs.size() < 2) {
            std::fprintf(stderr, "--accumulate needs >=2 images (or N .f32 pair dumps)\n"); free_splatter_free(ctx); return 2; }
        const size_t npairs = dump_mode ? inputs.size() : inputs.size() - 1;
        const int64_t per_view = (int64_t) geo.in_channels * sz * sz;
        const size_t  pair_floats = (size_t) 2 * sz * sz * gc;

        // image mode: decode every frame once (CHW per view)
        std::vector<std::vector<float>> frames;
        if (!dump_mode)
            for (const std::string & p : inputs) {
                std::vector<float> img;
                if (!load_image_chw(p.c_str(), sz, img)) { free_splatter_free(ctx); return 1; }
                frames.push_back(std::move(img));
            }

        free_splatter_accumulator * acc = free_splatter_accumulator_new(sz, sz, opac_thr);
        if (!acc) { std::fprintf(stderr, "accumulator alloc failed\n"); free_splatter_free(ctx); return 1; }

        for (size_t k = 0; k < npairs; k++) {
            const float * g = nullptr;
            std::vector<float> dumpbuf;
            float * runout = nullptr;
            if (dump_mode) {
                std::ifstream f(inputs[k], std::ios::binary | std::ios::ate);
                if (!f) { std::fprintf(stderr, "cannot open %s\n", inputs[k].c_str()); free_splatter_accumulator_free(acc); free_splatter_free(ctx); return 1; }
                const std::streamsize bytes = f.tellg(); f.seekg(0);
                if ((size_t)(bytes / sizeof(float)) != pair_floats) {
                    std::fprintf(stderr, "dump %s: %zu floats, expected 2*%d*%d*%d\n", inputs[k].c_str(),
                                 (size_t)(bytes/sizeof(float)), sz, sz, gc);
                    free_splatter_accumulator_free(acc); free_splatter_free(ctx); return 1; }
                dumpbuf.resize(pair_floats); f.read((char *) dumpbuf.data(), bytes);
                g = dumpbuf.data();
            } else {
                std::vector<float> pair(2 * per_view);
                std::memcpy(&pair[0],        frames[k].data(),   per_view * sizeof(float));
                std::memcpy(&pair[per_view], frames[k+1].data(), per_view * sizeof(float));
                size_t ng = 0;
                if (free_splatter_run(ctx, pair.data(), 2, sz, sz, &runout, &ng) != 0) {
                    std::fprintf(stderr, "run pair %zu failed: %s\n", k, free_splatter_last_error(ctx));
                    free_splatter_accumulator_free(acc); free_splatter_free(ctx); return 1; }
                g = runout;
            }
            free_splatter_accumulator_add_pair(acc, g, gc);
            if (runout) free_splatter_buf_free(runout);
            const int nframes = free_splatter_accumulator_frame_count(acc);
            std::printf("pair %zu -> %d frames\n", k, nframes);
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
            free_splatter_accumulator_fuse(acc, voxel, fuse_k, fuse_keep ? 1 : 0, &fc, &nf);
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
