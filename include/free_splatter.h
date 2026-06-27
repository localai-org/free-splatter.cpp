// free-splatter.cpp — a GGML inference engine for the neural-network front half
// of TencentARC/FreeSplatter: image patch tokenizer -> multi-view self-attention
// transformer -> per-pixel 3D-Gaussian parameter head.
//
// Scope: pieces 1-3 only. Given N uncalibrated views it returns, per input pixel,
// the activated Gaussian parameters (xyz, SH, opacity, scale, rotation). The
// rasterizer and the PnP pose solver are out of scope (a downstream consumer).
//
// Flat C API: no exceptions cross this boundary, every pointer-returning function
// reports failure via free_splatter_last_error, every free is NULL-safe. Shaped
// for FFI (purego, ctypes, cgo, WASM): opaque handle, caller-owned flat buffers.
#ifndef FREE_SPLATTER_H
#define FREE_SPLATTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Exported symbol marker. The library is built with -fvisibility=hidden, so only
// functions marked FREE_SPLATTER_API are exported across the ABI boundary.
#if defined(_WIN32)
  #define FREE_SPLATTER_API __declspec(dllexport)
#else
  #define FREE_SPLATTER_API __attribute__((visibility("default")))
#endif

#define FREE_SPLATTER_ABI_VERSION 1
FREE_SPLATTER_API int free_splatter_abi_version(void);

typedef struct free_splatter_ctx free_splatter_ctx;

// ---- options builder ----
// ABI-stable configuration: add setters without changing struct layout, so a
// compiled binding never breaks when new knobs appear.
typedef struct free_splatter_options free_splatter_options;
FREE_SPLATTER_API free_splatter_options * free_splatter_options_new(void);
FREE_SPLATTER_API void free_splatter_options_free(free_splatter_options * opts); // NULL-safe
// device: NULL or "cpu", "gpu", "cuda", "vulkan" (optionally ":N" to pick the Nth
// matching GPU). "gpu" = first GPU of whichever backend was compiled in.
FREE_SPLATTER_API void free_splatter_options_set_device(free_splatter_options * opts, const char * device);
// n_threads <= 0 picks a default (CPU only).
FREE_SPLATTER_API void free_splatter_options_set_threads(free_splatter_options * opts, int n_threads);
// If non-NULL, every named intermediate tensor (tap) is written to this dir on
// the next free_splatter_run, as <name>.f32/.i32 + meta.json — the format
// scripts/compare_taps.py reads. NULL disables tap dumping (the fast path).
FREE_SPLATTER_API void free_splatter_options_set_dump_taps_dir(free_splatter_options * opts, const char * dir);

// ---- lifecycle ----
// Returns NULL only on allocation failure; a non-NULL handle whose
// free_splatter_last_error is non-NULL means load failed (inspect, then free).
FREE_SPLATTER_API free_splatter_ctx * free_splatter_load(const char * gguf_path, const free_splatter_options * opts);
FREE_SPLATTER_API void                free_splatter_free(free_splatter_ctx * ctx);            // NULL-safe
FREE_SPLATTER_API const char *        free_splatter_last_error(const free_splatter_ctx * ctx); // NULL if no error

// ---- model geometry ----
// Lets a binding size buffers without hardcoding model constants.
typedef struct {
    int32_t in_channels;        // image channels the model expects (3, RGB)
    int32_t image_height;       // expected input height  (e.g. 512)
    int32_t image_width;        // expected input width   (e.g. 512)
    int32_t gaussian_channels;  // per-pixel output channels (e.g. 23)
} free_splatter_geometry;
FREE_SPLATTER_API int free_splatter_geometry_of(const free_splatter_ctx * ctx, free_splatter_geometry * out);

// ---- inference ----
// images: caller-owned, n_views * in_channels * height * width float32, range
//   [0,1], laid out NCHW per view (view-major, then channel, then row, then col).
// On success *out is malloc'd: n_views * height * width * gaussian_channels
//   float32 (the activated, render-ready Gaussian params); *n_out is its element
//   count. Free with free_splatter_buf_free. Returns 0 on success, -1 on failure
//   (see free_splatter_last_error). If a dump-taps dir was set on the options,
//   taps are written there as a side effect.
FREE_SPLATTER_API int  free_splatter_run(free_splatter_ctx * ctx, const float * images, int32_t n_views,
                       int32_t height, int32_t width, float ** out, size_t * n_out);
FREE_SPLATTER_API void free_splatter_buf_free(void * buf);  // NULL-safe

// ---- downstream pose recovery + accumulation ----
// The pose consumer of the engine seam: given the engine's [N,H,W,gaussian_channels]
// output it recovers each view's camera (PnP) and stitches successive runs into one
// accumulating world (cross-run Sim(3)). Dependency-free (no Python, no OpenCV).

// Recover each view's camera from an engine output buffer. gaussians:
// n_views*height*width*gaussian_channels float32 (the layout free_splatter_run
// returns). cam2world_out: caller-owned n_views*16 float32, filled with a
// row-major 4x4 cam2world per view. If focal_out is non-NULL it receives the
// estimated shared focal. opacity_threshold gates valid pixels (engine output is
// already activated). Returns 0 on success, -1 on failure.
FREE_SPLATTER_API int free_splatter_estimate_poses(
    const float * gaussians, int32_t n_views, int32_t height, int32_t width,
    int32_t gaussian_channels, float opacity_threshold,
    float * cam2world_out, float * focal_out);

// One accumulated 3D gaussian in the global frame: position, color in [0,1], the
// anisotropic scale and rotation quaternion (w,x,y,z) — transformed through the
// cross-run Sim(3) so the cloud renders as oriented splats — and the source frame
// it came from (for consensus fusion).
typedef struct {
    float x, y, z;
    float r, g, b, opacity;
    float sx, sy, sz;
    float qw, qx, qy, qz;
    int32_t frame;
} free_splatter_point;

// Sliding-window accumulator: feed each consecutive pair's engine output and it
// chains the runs into one world. Opaque handle, not thread-safe.
typedef struct free_splatter_accumulator free_splatter_accumulator;
FREE_SPLATTER_API free_splatter_accumulator * free_splatter_accumulator_new(
    int32_t height, int32_t width, float opacity_threshold);
FREE_SPLATTER_API void free_splatter_accumulator_free(free_splatter_accumulator * acc); // NULL-safe

// Add one pair's engine output: 2*height*width*gaussian_channels float32, where
// view 0 shares a frame with the previous pair's view 1. Returns 0 on success.
FREE_SPLATTER_API int free_splatter_accumulator_add_pair(
    free_splatter_accumulator * acc, const float * gaussians, int32_t gaussian_channels);

// Number of frames accumulated so far (== pairs_added + 1, or 0 before any pair).
FREE_SPLATTER_API int free_splatter_accumulator_frame_count(const free_splatter_accumulator * acc);

// Copy the current accumulated cloud: *out is malloc'd free_splatter_point[*n_out]
// (free with free_splatter_buf_free). Returns 0 on success, -1 on failure.
FREE_SPLATTER_API int free_splatter_accumulator_cloud(
    const free_splatter_accumulator * acc, free_splatter_point ** out, size_t * n_out);

// Consensus-fuse the current cloud: keep only voxels (size voxel_frac * extent)
// corroborated by >= k distinct source frames (single-frame floaters dropped).
// keep_raw=0 averages each consensus voxel to one point (denoised but decimated);
// keep_raw=1 keeps every raw gaussian in a consensus voxel (dense, nothing averaged
// away — better when there are few frames). *out malloc'd as above. Returns 0 ok.
FREE_SPLATTER_API int free_splatter_accumulator_fuse(
    const free_splatter_accumulator * acc, float voxel_frac, int32_t k, int32_t keep_raw,
    free_splatter_point ** out, size_t * n_out);

// Copy the global camera trajectory: *out malloc'd (*n_frames)*16 float32, a
// row-major 4x4 cam2world (similarity) per frame. Returns 0 on success.
FREE_SPLATTER_API int free_splatter_accumulator_camera_path(
    const free_splatter_accumulator * acc, float ** out, int32_t * n_frames);

#ifdef __cplusplus
}
#endif

#endif // FREE_SPLATTER_H
