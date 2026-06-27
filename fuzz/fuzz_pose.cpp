// libFuzzer harness for the public pose C-API surface (free_splatter_estimate_poses
// + the accumulator). These now take caller-supplied float buffers across the ABI,
// so a binding that feeds NaN/Inf/degenerate geometry must not crash, read OOB, or
// trip UBSan (e.g. a non-finite float -> int voxel-coord cast). The gaussian buffer
// is engine output, but the C-API is a public boundary; keep it robust.
#include "free_splatter.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    if (size < 8) return 0;

    // small, fuzz-driven geometry so many inputs are explored cheaply
    const int gc = 23;
    const int H = 4 + (data[0] % 5) * 4;          // 4,8,12,16,20
    const int W = 4 + (data[1] % 5) * 4;
    const int P = H * W;

    auto fill = [&](std::vector<float> & buf, size_t off) {
        // reinterpret the fuzz bytes as floats (NaN/Inf/denormals all appear),
        // tiling deterministically to fill the whole byte range.
        uint8_t * b = (uint8_t *) buf.data();
        const size_t nbytes = buf.size() * sizeof(float);
        for (size_t i = 0; i < nbytes; i++) b[i] = data[(off + i) % size];
    };

    // 1) estimate_poses on a 2-view buffer
    {
        std::vector<float> g((size_t) 2 * P * gc);
        fill(g, 2);
        std::vector<float> c2w((size_t) 2 * 16);
        float focal = 0;
        free_splatter_estimate_poses(g.data(), 2, H, W, gc, 0.05f, c2w.data(), &focal);
    }

    // 2) accumulator: add a few pairs, then cloud / fuse / camera_path
    free_splatter_accumulator * acc = free_splatter_accumulator_new(H, W, 0.05f);
    if (acc) {
        const int npairs = 1 + (data[2] % 3);     // 1..3 pairs
        std::vector<float> g((size_t) 2 * P * gc);
        for (int k = 0; k < npairs; k++) {
            fill(g, (size_t) 3 + k);
            free_splatter_accumulator_add_pair(acc, g.data(), gc);
        }
        free_splatter_point * cloud = nullptr; size_t nc = 0;
        if (free_splatter_accumulator_cloud(acc, &cloud, &nc) == 0) free_splatter_buf_free(cloud);

        const float voxel = 0.005f + (data[3] % 16) * 0.01f;   // 0.005..0.155
        const int k = 1 + (data[4] % 4);                       // 1..4
        const int keep_raw = data[5] & 1;                      // both fuse modes
        free_splatter_point * fused = nullptr; size_t nf = 0;
        if (free_splatter_accumulator_fuse(acc, voxel, k, keep_raw, &fused, &nf) == 0) free_splatter_buf_free(fused);

        float * path = nullptr; int32_t nfr = 0;
        if (free_splatter_accumulator_camera_path(acc, &path, &nfr) == 0) free_splatter_buf_free(path);

        free_splatter_accumulator_free(acc);
    }
    return 0;
}
