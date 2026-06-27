// pose.h — downstream camera-pose recovery + cross-run alignment (C++ port).
//
// This is the C++ port of the proven `pose/` research prototype (focal.py +
// align.py + pnp.py): given the engine's per-pixel gaussians it recovers each
// view's camera (PnP), and aligns successive runs into one accumulating world
// (Sim(3)). It is dependency-free — only the self-contained linalg.h — so it
// ships from the CLI / C API with no Python and no Eigen/OpenCV (see CLAUDE.md).
//
// Faithful to FreeSplatter's scene recipe (estimate_poses -> DUSt3R fast_pnp):
// the numpy reference solver, verified ~1e-7 against cv2 and bit-exact to upstream
// estimate_poses on real engine output (pose/check_upstream_parity.py).
//
// Conventions: f64 throughout; a similarity acts as x -> s*(R@x)+t; gaussian
// channel layout (scene, 23ch) is xyz[0:3] SH[3:15] opacity[15] scale[16:19]
// rotation[19:23], opacity already activated (sigmoid) in [0,1].
#ifndef FREE_SPLATTER_POSE_H
#define FREE_SPLATTER_POSE_H

#include "linalg.h"

#include <string>
#include <vector>

namespace free_splatter {
namespace pose {

using fsla::Vec3;
using fsla::Mat3;
using fsla::Mat4;

// ---- focal (Weiszfeld shared-focal, mirrors pose/focal.py) -----------------
// Robust (L1) focal from camera-frame points and their pixels, principal point
// pp. N points, pts3d is N*3 (x,y,z), pixels is N*2 (col,row), all row-major.
double estimate_focal(const double * pts3d, const double * pixels, int N,
                      double ppx, double ppy, int weiszfeld_iters = 10);

// ---- similarity alignment (mirrors pose/align.py) --------------------------
struct Sim3 { double s; Mat3 R; Vec3 t; };           // x -> s*(R@x)+t

Sim3 sim_identity();
Vec3 sim_apply(const Sim3 & T, const Vec3 & x);
Sim3 sim_compose(const Sim3 & T2, const Sim3 & T1);  // apply T1 then T2
Sim3 sim_invert(const Sim3 & T);
Mat4 sim_matrix(const Sim3 & T);                     // 4x4 [[sR,t],[0,1]]

// Umeyama closed-form (s,R,t) minimizing ||sRX+t - Y||^2. with_scale=false -> rigid.
// X, Y are N*3 row-major.
Sim3 fit_similarity(const double * X, const double * Y, int N, bool with_scale = true);

// RANSAC similarity robust to gross outliers; fills `inliers` (size N, 0/1).
Sim3 fit_similarity_ransac(const double * X, const double * Y, int N, double thresh,
                           int iters, std::vector<char> & inliers,
                           bool with_scale = true, uint64_t seed = 0);

// Residual ladder (rigid -> similarity -> affine) + depth correlation, and the
// verdict that classifies the A<->B mismatch. Mirrors align.diagnose.
struct Ladder {
    double scene, rigid, similarity, affine, scale, depth_corr, aff_gain;
    bool structured;
    std::string verdict;   // rigid_ok | needs_scale | needs_affine | nonlinear | similarity_plus_noise
};
Ladder diagnose(const double * X, const double * Y, int N,
                double tol = 1e-3, double corr_tol = 0.3);

// Loop-closure metrics for a closed chain of similarities (deviation from I).
struct LoopError { double scale_err, rot_deg, trans; };
LoopError loop_closure_error(const std::vector<Sim3> & links);

// Fractional power M^f of a Sim(3) 4x4 (even loop-closure relaxation). Real part
// of the eigendecomposition; valid while rotation < 180deg. Mirrors sim_frac_power.
Mat4 sim_frac_power(const Mat4 & M, double f);

// ---- PnP -------------------------------------------------------------------
// RANSAC DLT PnP with known intrinsics K (mirrors pose/pnp.py numpy backend).
// Xw is N*3 world points, pixels is N*2 (col,row). Returns cam2world (4x4) and
// fills `inliers` (size N, 0/1). Kept as the asset-free golden-test reference; on
// real scenes the DLT inherits the planar/mirror degeneracy — use solve_pnp.
Mat4 solve_pnp_numpy(const double * Xw, const double * pixels, int N, const Mat3 & K,
                     std::vector<char> & inliers,
                     double thresh_px = 5.0, int iters = 100, uint64_t seed = 0);

// Robust PnP: EPnP (planar-robust, deterministic — uses all points, no random
// minimal samples) for the initial pose, then a Gauss-Newton / Huber reprojection
// refine. Reaches cv2/SQPNP-grade poses on real scenes with no OpenCV dependency.
// Fills `inliers` (reprojection < thresh_px). This is what estimate_poses uses.
Mat4 solve_pnp(const double * Xw, const double * pixels, int N, const Mat3 & K,
               std::vector<char> & inliers, double thresh_px = 5.0, int gn_iters = 10);

// EPnP-only (no refine) — exposed for testing the planar-robust initialization.
Mat4 solve_pnp_epnp(const double * Xw, const double * pixels, int N, const Mat3 & K);

// Integer pixel grid (col,row), row-major over H*W — matches upstream xy_grid
// (no half-pixel offset). Fills `out` with 2*H*W doubles.
void pixel_grid(int H, int W, std::vector<double> & out);

// ---- estimate_poses orchestration (scene recipe, mirrors pnp.estimate_poses) -
struct PoseResult {
    std::vector<Mat4> cam2world;
    double focal;
    double scale;
};
// points: N views, each H*W*3 (gaussian centers, view-0 frame). opacities: N
// views, each H*W (activated; pass nullptr entries for no mask). focal<=0 ->
// estimate from view 0 over all pixels (use_first_focal). normalize -> the
// runner's 1/baseline camera rescale.
PoseResult estimate_poses(const std::vector<const float *> & points,
                          const std::vector<const float *> & opacities,
                          int H, int W,
                          double opacity_threshold = 0.05,
                          double focal = -1.0, int pnp_iter = 100,
                          bool normalize = false, uint64_t seed = 0);

} // namespace pose
} // namespace free_splatter

#endif // FREE_SPLATTER_POSE_H
