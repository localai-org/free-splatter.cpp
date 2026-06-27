// pose.cpp — C++ port of the pose/ prototype (focal.py + align.py + pnp.py).
//
// Faithful to FreeSplatter's scene recipe; the dependency-free numpy reference
// solver is what ships here (verified ~1e-7 vs cv2, bit-exact vs upstream
// estimate_poses on real engine output). All linear algebra goes through the
// self-contained Jacobi eigensolver in linalg.h — no Eigen, no OpenCV.
#include "pose.h"

#include <cmath>
#include <cstring>

namespace free_splatter {
namespace pose {

using fsla::smallest_eigenvector;
using fsla::svd3;
using fsla::det3;
using fsla::inv3;
using fsla::mat3_mul;
using fsla::mat3_transpose;
using fsla::mat3_apply;
using fsla::mat3_identity;
using fsla::mat4_identity;
using fsla::inv_rigid4;

namespace {

// Deterministic SplitMix64 — so RANSAC is reproducible across platforms (we don't
// need numpy's PCG64 bitstream, only a fixed inlier-converging sampler).
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    int bounded(int n) { return (int) (next() % (uint64_t) n); }
};

void sample_distinct(Rng & rng, int N, int k, int * out) {
    for (int i = 0; i < k; i++) {
        int v;
        bool dup;
        do {
            v = rng.bounded(N);
            dup = false;
            for (int j = 0; j < i; j++) if (out[j] == v) { dup = true; break; }
        } while (dup);
        out[i] = v;
    }
}

double rms3(const double * r, int N) {            // sqrt(mean over rows of ||row||^2)
    double s = 0;
    for (int i = 0; i < N; i++) s += r[3*i]*r[3*i] + r[3*i+1]*r[3*i+1] + r[3*i+2]*r[3*i+2];
    return std::sqrt(s / (double) N);
}

void mean3(const double * X, int N, double * m) {
    m[0] = m[1] = m[2] = 0;
    for (int i = 0; i < N; i++) { m[0] += X[3*i]; m[1] += X[3*i+1]; m[2] += X[3*i+2]; }
    m[0] /= N; m[1] /= N; m[2] /= N;
}

// Solve A X = B in place (A n x n row-major, B n x m row-major) by Gaussian
// elimination with partial pivoting. Result overwrites B. Used for the affine
// normal equations in the residual ladder.
void solve_lin(double * A, int n, double * B, int m) {
    for (int col = 0; col < n; col++) {
        int piv = col;
        for (int r = col + 1; r < n; r++)
            if (std::fabs(A[r*n+col]) > std::fabs(A[piv*n+col])) piv = r;
        if (piv != col) {
            for (int c = 0; c < n; c++) std::swap(A[col*n+c], A[piv*n+c]);
            for (int c = 0; c < m; c++) std::swap(B[col*m+c], B[piv*m+c]);
        }
        const double d = A[col*n+col];
        if (std::fabs(d) < 1e-300) continue;
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            const double f = A[r*n+col] / d;
            if (f == 0) continue;
            for (int c = col; c < n; c++) A[r*n+c] -= f * A[col*n+c];
            for (int c = 0; c < m; c++)   B[r*m+c] -= f * B[col*m+c];
        }
    }
    for (int col = 0; col < n; col++) {
        const double d = A[col*n+col];
        if (std::fabs(d) > 1e-300) for (int c = 0; c < m; c++) B[col*m+c] /= d;
    }
}

Mat3 rodrigues(const Vec3 & axis, double phi) {     // rotation about unit axis by phi
    const double cphi = std::cos(phi), sphi = std::sin(phi), v = 1.0 - cphi;
    const double x = axis[0], y = axis[1], z = axis[2];
    Mat3 R{};
    R(0,0) = cphi + x*x*v;   R(0,1) = x*y*v - z*sphi; R(0,2) = x*z*v + y*sphi;
    R(1,0) = y*x*v + z*sphi; R(1,1) = cphi + y*y*v;   R(1,2) = y*z*v - x*sphi;
    R(2,0) = z*x*v - y*sphi; R(2,1) = z*y*v + x*sphi; R(2,2) = cphi + z*z*v;
    return R;
}

} // namespace

// ---- focal ----------------------------------------------------------------

double estimate_focal(const double * pts3d, const double * pixels, int N,
                      double ppx, double ppy, int weiszfeld_iters) {
    std::vector<double> P(2 * N), U(2 * N), pu(N), uu(N);
    for (int i = 0; i < N; i++) {
        const double X = pts3d[3*i], Y = pts3d[3*i+1], Z = pts3d[3*i+2];
        double ux = (Z != 0.0) ? X / Z : 0.0;
        double uy = (Z != 0.0) ? Y / Z : 0.0;
        if (!std::isfinite(ux)) ux = 0.0;       // mirror np.nan_to_num(posinf/neginf->0)
        if (!std::isfinite(uy)) uy = 0.0;
        const double Px = pixels[2*i]   - ppx;
        const double Py = pixels[2*i+1] - ppy;
        P[2*i] = Px; P[2*i+1] = Py;
        U[2*i] = ux; U[2*i+1] = uy;
        pu[i] = Px*ux + Py*uy;
        uu[i] = ux*ux + uy*uy;
    }
    double spu = 0, suu = 0;
    for (int i = 0; i < N; i++) { spu += pu[i]; suu += uu[i]; }
    double f = (suu != 0.0) ? spu / suu : 0.0;
    for (int it = 0; it < weiszfeld_iters; it++) {
        double num = 0, den = 0;
        for (int i = 0; i < N; i++) {
            const double dx = P[2*i] - f*U[2*i], dy = P[2*i+1] - f*U[2*i+1];
            const double r = std::sqrt(dx*dx + dy*dy);
            const double w = 1.0 / std::max(r, 1e-8);
            num += w * pu[i];
            den += w * uu[i];
        }
        f = (den != 0.0) ? num / den : f;
    }
    return f;
}

// ---- similarity -----------------------------------------------------------

Sim3 sim_identity() { return { 1.0, mat3_identity(), {0,0,0} }; }

Vec3 sim_apply(const Sim3 & T, const Vec3 & x) {
    Vec3 Rx = mat3_apply(T.R, x);
    return { T.s*Rx[0] + T.t[0], T.s*Rx[1] + T.t[1], T.s*Rx[2] + T.t[2] };
}

Sim3 sim_compose(const Sim3 & T2, const Sim3 & T1) {   // apply T1 then T2
    Sim3 r;
    r.s = T2.s * T1.s;
    r.R = mat3_mul(T2.R, T1.R);
    Vec3 R2t1 = mat3_apply(T2.R, T1.t);
    r.t = { T2.s*R2t1[0] + T2.t[0], T2.s*R2t1[1] + T2.t[1], T2.s*R2t1[2] + T2.t[2] };
    return r;
}

Sim3 sim_invert(const Sim3 & T) {
    Sim3 r;
    r.s = 1.0 / T.s;
    r.R = mat3_transpose(T.R);
    Vec3 Rit = mat3_apply(r.R, T.t);
    r.t = { -r.s*Rit[0], -r.s*Rit[1], -r.s*Rit[2] };
    return r;
}

Mat4 sim_matrix(const Sim3 & T) {
    Mat4 M = mat4_identity();
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) M(i, j) = T.s * T.R(i, j);
    for (int i = 0; i < 3; i++) M(i, 3) = T.t[i];
    return M;
}

Sim3 fit_similarity(const double * X, const double * Y, int N, bool with_scale) {
    double mx[3], my[3];
    mean3(X, N, mx);
    mean3(Y, N, my);
    // Sigma = (Ycᵀ Xc)/n, cross-covariance mapping X -> Y; var_x = mean ||Xc||^2.
    Mat3 Sigma{};
    double var_x = 0;
    for (int i = 0; i < N; i++) {
        const double xc[3] = { X[3*i]-mx[0], X[3*i+1]-mx[1], X[3*i+2]-mx[2] };
        const double yc[3] = { Y[3*i]-my[0], Y[3*i+1]-my[1], Y[3*i+2]-my[2] };
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) Sigma(r, c) += yc[r]*xc[c];
        var_x += xc[0]*xc[0] + xc[1]*xc[1] + xc[2]*xc[2];
    }
    for (int i = 0; i < 9; i++) Sigma.a[i] /= N;
    var_x /= N;

    Mat3 U, V; Vec3 D;
    svd3(Sigma, U, D, V);                          // Sigma = U diag(D) Vᵀ
    double sgn = (det3(U) * det3(V) < 0) ? -1.0 : 1.0;
    // R = U diag(1,1,sgn) Vᵀ
    Mat3 US = U;
    for (int r = 0; r < 3; r++) US(r, 2) *= sgn;
    Mat3 R = mat3_mul(US, mat3_transpose(V));

    double s = 1.0;
    if (with_scale && var_x > 0)
        s = (D[0] + D[1] + sgn * D[2]) / var_x;

    Vec3 mxv = { mx[0], mx[1], mx[2] };
    Vec3 Rmx = mat3_apply(R, mxv);
    Sim3 T;
    T.s = s; T.R = R;
    T.t = { my[0] - s*Rmx[0], my[1] - s*Rmx[1], my[2] - s*Rmx[2] };
    return T;
}

Sim3 fit_similarity_ransac(const double * X, const double * Y, int N, double thresh,
                           int iters, std::vector<char> & inliers,
                           bool with_scale, uint64_t seed) {
    Rng rng(seed);
    std::vector<char> best(N, 0);
    int best_cnt = 0;
    std::vector<double> xs(9), ys(9);              // 3-point minimal samples
    for (int it = 0; it < iters; it++) {
        int idx[3];
        sample_distinct(rng, N, 3, idx);
        for (int j = 0; j < 3; j++)
            for (int c = 0; c < 3; c++) { xs[3*j+c] = X[3*idx[j]+c]; ys[3*j+c] = Y[3*idx[j]+c]; }
        Sim3 T = fit_similarity(xs.data(), ys.data(), 3, with_scale);
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            Vec3 xi = { X[3*i], X[3*i+1], X[3*i+2] };
            Vec3 p = sim_apply(T, xi);
            const double dx = p[0]-Y[3*i], dy = p[1]-Y[3*i+1], dz = p[2]-Y[3*i+2];
            if (std::sqrt(dx*dx + dy*dy + dz*dz) < thresh) cnt++;
        }
        if (cnt > best_cnt) { best_cnt = cnt; best.assign(N, 0);
            for (int i = 0; i < N; i++) {
                Vec3 xi = { X[3*i], X[3*i+1], X[3*i+2] };
                Vec3 p = sim_apply(T, xi);
                const double dx = p[0]-Y[3*i], dy = p[1]-Y[3*i+1], dz = p[2]-Y[3*i+2];
                best[i] = (std::sqrt(dx*dx + dy*dy + dz*dz) < thresh) ? 1 : 0;
            }
        }
    }
    if (best_cnt < 3) std::fill(best.begin(), best.end(), 1);
    // refit on the inlier set
    std::vector<double> xi, yi;
    for (int i = 0; i < N; i++) if (best[i]) {
        for (int c = 0; c < 3; c++) { xi.push_back(X[3*i+c]); yi.push_back(Y[3*i+c]); }
    }
    inliers = best;
    return fit_similarity(xi.data(), yi.data(), (int) (xi.size()/3), with_scale);
}

// affine Y ~= A X + b (12-DoF), returns prediction rms only via caller; here we
// fill `pred` (N*3). Normal equations on Xh=[X,1].
static double affine_residual(const double * X, const double * Y, int N) {
    double G[16] = {0};       // XhᵀXh (4x4)
    double B[12] = {0};       // XhᵀY  (4x3)
    for (int i = 0; i < N; i++) {
        const double xh[4] = { X[3*i], X[3*i+1], X[3*i+2], 1.0 };
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) G[r*4+c] += xh[r]*xh[c];
            for (int c = 0; c < 3; c++) B[r*3+c] += xh[r]*Y[3*i+c];
        }
    }
    solve_lin(G, 4, B, 3);    // B now holds M (4x3): rows are coeffs
    double s = 0;
    for (int i = 0; i < N; i++) {
        const double xh[4] = { X[3*i], X[3*i+1], X[3*i+2], 1.0 };
        for (int c = 0; c < 3; c++) {
            double p = 0;
            for (int r = 0; r < 4; r++) p += xh[r]*B[r*3+c];
            const double d = p - Y[3*i+c];
            s += d*d;
        }
    }
    return std::sqrt(s / (double) N);
}

Ladder diagnose(const double * X, const double * Y, int N, double tol, double corr_tol) {
    Ladder L{};
    double my[3]; mean3(Y, N, my);
    std::vector<double> yc(3 * N);
    for (int i = 0; i < N; i++) { yc[3*i]=Y[3*i]-my[0]; yc[3*i+1]=Y[3*i+1]-my[1]; yc[3*i+2]=Y[3*i+2]-my[2]; }
    L.scene = rms3(yc.data(), N);

    Sim3 Tr = fit_similarity(X, Y, N, /*with_scale=*/false);
    Sim3 Ts = fit_similarity(X, Y, N, /*with_scale=*/true);
    std::vector<double> res_r(3*N), res_s(3*N);
    for (int i = 0; i < N; i++) {
        Vec3 xi = { X[3*i], X[3*i+1], X[3*i+2] };
        Vec3 pr = sim_apply(Tr, xi), ps = sim_apply(Ts, xi);
        for (int c = 0; c < 3; c++) { res_r[3*i+c] = pr[c]-Y[3*i+c]; res_s[3*i+c] = ps[c]-Y[3*i+c]; }
    }
    L.rigid = rms3(res_r.data(), N);
    L.similarity = rms3(res_s.data(), N);
    L.affine = affine_residual(X, Y, N);
    L.scale = Ts.s;

    // depth_corr: Pearson(||X-mean||, ||similarity residual||)
    double mx[3]; mean3(X, N, mx);
    std::vector<double> depth(N), rmag(N);
    for (int i = 0; i < N; i++) {
        const double dx = X[3*i]-mx[0], dy = X[3*i+1]-mx[1], dz = X[3*i+2]-mx[2];
        depth[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        rmag[i] = std::sqrt(res_s[3*i]*res_s[3*i] + res_s[3*i+1]*res_s[3*i+1] + res_s[3*i+2]*res_s[3*i+2]);
    }
    double md = 0, mr = 0;
    for (int i = 0; i < N; i++) { md += depth[i]; mr += rmag[i]; }
    md /= N; mr /= N;
    double cov = 0, vd = 0, vr = 0;
    for (int i = 0; i < N; i++) { cov += (depth[i]-md)*(rmag[i]-mr); vd += (depth[i]-md)*(depth[i]-md); vr += (rmag[i]-mr)*(rmag[i]-mr); }
    L.depth_corr = (vr > 1e-24) ? cov / std::sqrt(vd*vr) : 0.0;

    const double sc = (L.scene > 0) ? L.scene : 1.0;
    const double rr = L.rigid/sc, rs = L.similarity/sc, ra = L.affine/sc;
    L.aff_gain = (L.similarity > 0) ? (L.similarity - L.affine)/L.similarity : 0.0;
    L.structured = std::fabs(L.depth_corr) > corr_tol || L.aff_gain > 0.25;
    if (rr < tol)         L.verdict = "rigid_ok";
    else if (rs < tol)    L.verdict = "needs_scale";
    else if (ra < tol)    L.verdict = "needs_affine";
    else if (L.structured) L.verdict = "nonlinear";
    else                  L.verdict = "similarity_plus_noise";
    return L;
}

LoopError loop_closure_error(const std::vector<Sim3> & links) {
    Sim3 T = sim_identity();
    for (const Sim3 & Ti : links) T = sim_compose(Ti, T);
    LoopError e;
    e.scale_err = std::fabs(std::log(T.s));
    const double tr = T.R(0,0) + T.R(1,1) + T.R(2,2);
    e.rot_deg = std::acos(std::max(-1.0, std::min(1.0, (tr - 1.0)/2.0))) * 180.0 / M_PI;
    e.trans = std::sqrt(T.t[0]*T.t[0] + T.t[1]*T.t[1] + T.t[2]*T.t[2]);
    return e;
}

// M^f for a Sim(3) 4x4 [[sR,t],[0,1]], closed form via the group's one-parameter
// subgroup: A^f = s^f R^f (R^f = same axis, f*angle), translation by
// (A^f - I)(A - I)^{-1} t (the analytic continuation of the integer-power
// geometric series). Equals numpy's eig-based principal value while angle < 180.
Mat4 sim_frac_power(const Mat4 & M, double f) {
    Mat3 A{};
    Vec3 t{};
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) A(i, j) = M(i, j); t[i] = M(i, 3); }
    const double s = std::cbrt(det3(A));
    Mat3 R = A;
    for (int i = 0; i < 9; i++) R.a[i] /= s;
    const double tr = R(0,0) + R(1,1) + R(2,2);
    const double ang = std::acos(std::max(-1.0, std::min(1.0, (tr - 1.0)/2.0)));

    Mat3 Rf;
    if (ang < 1e-12) {
        Rf = mat3_identity();
    } else {
        const double sa = std::sin(ang);
        Vec3 axis = { (R(2,1)-R(1,2))/(2*sa), (R(0,2)-R(2,0))/(2*sa), (R(1,0)-R(0,1))/(2*sa) };
        Rf = rodrigues(axis, f * ang);
    }
    const double sf = std::pow(s, f);
    Mat3 Af = Rf;
    for (int i = 0; i < 9; i++) Af.a[i] *= sf;

    Mat4 out = mat4_identity();
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) out(i, j) = Af(i, j);

    // translation: (Af - I)(A - I)^{-1} t, with the A->I limit handled as f*t.
    Mat3 AmI = A;
    for (int i = 0; i < 3; i++) AmI(i, i) -= 1.0;
    if (std::fabs(det3(AmI)) < 1e-12) {
        for (int i = 0; i < 3; i++) out(i, 3) = f * t[i];
    } else {
        Mat3 AfmI = Af;
        for (int i = 0; i < 3; i++) AfmI(i, i) -= 1.0;
        Mat3 K = mat3_mul(AfmI, inv3(AmI));
        Vec3 tf = mat3_apply(K, t);
        for (int i = 0; i < 3; i++) out(i, 3) = tf[i];
    }
    return out;
}

// ---- PnP ------------------------------------------------------------------

namespace {

// DLT: smallest right singular vector of the 2N-row system, via the smallest
// eigenvector of AᵀA (12x12). idx selects the k points used. Returns the 3x4
// matrix p (row-major) up to sign/scale.
std::array<double,12> dlt(const double * Xw, const double * xn, const int * idx, int k) {
    double AtA[144] = {0};
    for (int j = 0; j < k; j++) {
        const int i = idx[j];
        const double X = Xw[3*i], Y = Xw[3*i+1], Z = Xw[3*i+2];
        const double u = xn[2*i], v = xn[2*i+1];
        const double ra[12] = { X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u };
        const double rb[12] = { 0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v };
        for (int r = 0; r < 12; r++)
            for (int c = 0; c < 12; c++) AtA[r*12+c] += ra[r]*ra[c] + rb[r]*rb[c];
    }
    std::vector<double> v = smallest_eigenvector(AtA, 12);
    std::array<double,12> p;
    for (int i = 0; i < 12; i++) p[i] = v[i];
    return p;
}

struct RT { Mat3 R; Vec3 t; double err; bool ok; };

// Resolve the DLT sign and recover a proper (R,t) world2cam, picking the sign with
// lower reprojection error among cheirality-valid candidates (points in front).
RT decode(const std::array<double,12> & p, const double * Xw, const double * xn,
          const int * idx, int k) {
    RT best; best.ok = false; best.err = 1e300;
    for (double sgn : { 1.0, -1.0 }) {
        Mat3 M{};
        Vec3 tau{};
        for (int r = 0; r < 3; r++) { for (int c = 0; c < 3; c++) M(r, c) = sgn*p[r*4+c]; tau[r] = sgn*p[r*4+3]; }
        Mat3 U, V; Vec3 Sv;
        svd3(M, U, Sv, V);
        const double d = (det3(U) * det3(V) < 0) ? -1.0 : 1.0;
        Mat3 US = U;
        for (int r = 0; r < 3; r++) US(r, 2) *= d;
        Mat3 R = mat3_mul(US, mat3_transpose(V));      // nearest proper rotation
        const double lam = (Sv[0] + Sv[1] + Sv[2]) / 3.0;
        if (lam < 1e-12) continue;
        Vec3 t = { tau[0]/lam, tau[1]/lam, tau[2]/lam };

        double err = 0; bool behind = false;
        for (int j = 0; j < k; j++) {
            const int i = idx[j];
            Vec3 xi = { Xw[3*i], Xw[3*i+1], Xw[3*i+2] };
            Vec3 Xc = mat3_apply(R, xi);
            Xc[0] += t[0]; Xc[1] += t[1]; Xc[2] += t[2];
            if (Xc[2] <= 0) { behind = true; break; }
            const double px = Xc[0]/Xc[2], py = Xc[1]/Xc[2];
            const double dx = px - xn[2*i], dy = py - xn[2*i+1];
            err += std::sqrt(dx*dx + dy*dy);
        }
        if (behind) continue;
        err /= k;
        if (err < best.err) { best.err = err; best.R = R; best.t = t; best.ok = true; }
    }
    return best;
}

} // namespace

void pixel_grid(int H, int W, std::vector<double> & out) {
    out.resize((size_t) 2 * H * W);
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            const size_t k = (size_t) r * W + c;
            out[2*k] = c; out[2*k+1] = r;
        }
}

Mat4 solve_pnp_numpy(const double * Xw, const double * pixels, int N, const Mat3 & K,
                     std::vector<char> & inliers, double thresh_px, int iters, uint64_t seed) {
    const Mat3 Kinv = inv3(K);
    std::vector<double> xn((size_t) 2 * N);
    for (int i = 0; i < N; i++) {
        Vec3 uv = { pixels[2*i], pixels[2*i+1], 1.0 };
        Vec3 n = mat3_apply(Kinv, uv);
        xn[2*i] = n[0]; xn[2*i+1] = n[1];
    }
    const double thresh = thresh_px / K(0, 0);

    Rng rng(seed);
    std::vector<char> best(N, 0);
    int best_cnt = 0;
    for (int it = 0; it < iters; it++) {
        int idx[6];
        sample_distinct(rng, N, 6, idx);
        RT rt = decode(dlt(Xw, xn.data(), idx, 6), Xw, xn.data(), idx, 6);
        if (!rt.ok) continue;
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            Vec3 xi = { Xw[3*i], Xw[3*i+1], Xw[3*i+2] };
            Vec3 Xc = mat3_apply(rt.R, xi);
            Xc[2] += rt.t[2];
            if (Xc[2] <= 0) continue;
            Xc[0] += rt.t[0]; Xc[1] += rt.t[1];
            const double dx = Xc[0]/Xc[2] - xn[2*i], dy = Xc[1]/Xc[2] - xn[2*i+1];
            if (std::sqrt(dx*dx + dy*dy) < thresh) cnt++;
        }
        if (cnt > best_cnt) {
            best_cnt = cnt;
            best.assign(N, 0);
            for (int i = 0; i < N; i++) {
                Vec3 xi = { Xw[3*i], Xw[3*i+1], Xw[3*i+2] };
                Vec3 Xc = mat3_apply(rt.R, xi);
                Xc[0] += rt.t[0]; Xc[1] += rt.t[1]; Xc[2] += rt.t[2];
                if (Xc[2] <= 0) continue;
                const double dx = Xc[0]/Xc[2] - xn[2*i], dy = Xc[1]/Xc[2] - xn[2*i+1];
                best[i] = (std::sqrt(dx*dx + dy*dy) < thresh) ? 1 : 0;
            }
        }
    }
    std::vector<int> sel;
    for (int i = 0; i < N; i++) if (best[i]) sel.push_back(i);
    if ((int) sel.size() < 6) { sel.clear(); for (int i = 0; i < N; i++) sel.push_back(i); }
    RT rt = decode(dlt(Xw, xn.data(), sel.data(), (int) sel.size()),
                   Xw, xn.data(), sel.data(), (int) sel.size());

    Mat4 world2cam = mat4_identity();
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) world2cam(i, j) = rt.R(i, j); world2cam(i, 3) = rt.t[i]; }
    inliers = best;
    return inv_rigid4(world2cam);
}

// ---- estimate_poses orchestration -----------------------------------------

PoseResult estimate_poses(const std::vector<const float *> & points,
                          const std::vector<const float *> & opacities,
                          int H, int W, double opacity_threshold,
                          double focal, int pnp_iter, bool normalize, uint64_t seed) {
    const int Nv = (int) points.size();
    const int P = H * W;
    const double ppx = W / 2.0, ppy = H / 2.0;
    std::vector<double> grid;
    pixel_grid(H, W, grid);

    // per-view opacity masks
    std::vector<std::vector<char>> masks(Nv);
    for (int v = 0; v < Nv; v++) {
        masks[v].assign(P, 1);
        if (v < (int) opacities.size() && opacities[v]) {
            for (int i = 0; i < P; i++) masks[v][i] = (opacities[v][i] > opacity_threshold) ? 1 : 0;
        }
    }

    // focal: view 0 only, ALL pixels (use_first_focal, scene recipe)
    if (focal <= 0.0) {
        std::vector<double> pts(3 * P), pix(2 * P);
        for (int i = 0; i < P; i++) {
            pts[3*i] = points[0][3*i]; pts[3*i+1] = points[0][3*i+1]; pts[3*i+2] = points[0][3*i+2];
            pix[2*i] = grid[2*i]; pix[2*i+1] = grid[2*i+1];
        }
        focal = estimate_focal(pts.data(), pix.data(), P, ppx, ppy);
    }
    Mat3 K = mat3_identity();
    K(0,0) = focal; K(1,1) = focal; K(0,2) = ppx; K(1,2) = ppy;

    PoseResult res;
    res.focal = focal;
    res.scale = 1.0;
    for (int v = 0; v < Nv; v++) {
        std::vector<double> Xw, px;
        for (int i = 0; i < P; i++) if (masks[v][i]) {
            Xw.push_back(points[v][3*i]); Xw.push_back(points[v][3*i+1]); Xw.push_back(points[v][3*i+2]);
            px.push_back(grid[2*i]); px.push_back(grid[2*i+1]);
        }
        std::vector<char> inl;
        Mat4 c2w = solve_pnp_numpy(Xw.data(), px.data(), (int) (Xw.size()/3), K, inl, 5.0, pnp_iter, seed);
        res.cam2world.push_back(c2w);
    }

    if (normalize && Nv >= 2) {
        Vec3 t0 = { res.cam2world.front()(0,3), res.cam2world.front()(1,3), res.cam2world.front()(2,3) };
        Vec3 tl = { res.cam2world.back()(0,3),  res.cam2world.back()(1,3),  res.cam2world.back()(2,3) };
        const double baseline = std::sqrt((tl[0]-t0[0])*(tl[0]-t0[0]) + (tl[1]-t0[1])*(tl[1]-t0[1]) + (tl[2]-t0[2])*(tl[2]-t0[2])) + 1e-2;
        res.scale = 1.0 / baseline;
        for (auto & c : res.cam2world) for (int i = 0; i < 3; i++) c(i, 3) *= res.scale;
    }
    return res;
}

} // namespace pose
} // namespace free_splatter
