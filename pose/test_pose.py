"""Asset-free golden tests for the pose / coordinate-normalization prototype.

No model, no fixtures, no cv2: synthetic geometry with KNOWN ground truth, the
way CLAUDE.md wants new ops verified ("pinned to hand-computed references... a
wrong op should fail with zero fixtures"). Run:

    nix develop -c python3 pose/test_pose.py     # from repo root
    cd pose && nix develop -c python3 test_pose.py

Each test prints PASS/FAIL with the numbers it checked; exit code is nonzero on
any failure.
"""
import numpy as np

import align
import focal
import pnp

RNG = np.random.default_rng(12345)
_fails = []


def check(name, cond, detail: object = ""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _fails.append(name)


def rand_rotation(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def rot_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


# ===========================================================================
# Coordinate-system normalization (the scale question)
# ===========================================================================

def test_similarity_roundtrip():
    """Recover a KNOWN (s,R,t) exactly -- the fundamental correctness golden."""
    print("test_similarity_roundtrip")
    X = RNG.normal(size=(200, 3)) * np.array([3, 2, 5])
    s_t, R_t, t_t = 1.7, rand_rotation(RNG), np.array([4.0, -1.0, 2.0])
    Y = align.apply_sim((s_t, R_t, t_t), X)
    s, R, t = align.fit_similarity(X, Y)
    check("scale", abs(s - s_t) < 1e-9, f"{s:.12f} vs {s_t}")
    check("rotation", rot_angle(R @ R_t.T) < 1e-7, f"{rot_angle(R @ R_t.T):.2e} deg")
    check("translation", np.linalg.norm(t - t_t) < 1e-7)
    check("residual~0", align._rms(align.apply_sim((s, R, t), X) - Y) < 1e-9)


def test_scale_detection():
    """B differs from A by PURE uniform scale: rigid fails, similarity nails it.
    This is the detector for 'you need the scale DoF' -- the expected cross-run case."""
    print("test_scale_detection")
    X = RNG.normal(size=(300, 3)) * 2.0
    s_t = 2.5
    Y = align.apply_sim((s_t, rand_rotation(RNG), np.array([1.0, 2.0, -3.0])), X)
    L = align.diagnose(X, Y)
    check("recovered scale", abs(L["scale"] - s_t) < 1e-8, f"{L['scale']:.10f}")
    check("rigid residual large", L["rigid"] / L["scene"] > 0.1, f"{L['rigid']/L['scene']:.3f}")
    check("similarity residual ~0", L["similarity"] / L["scene"] < 1e-9)
    check("verdict needs_scale", L["verdict"] == "needs_scale", L["verdict"])


def test_nonlinear_detection():
    """Affine and genuinely-nonlinear warps are distinguished from pure scale."""
    print("test_nonlinear_detection")
    X = RNG.normal(size=(400, 3)) * 2.0 + np.array([0, 0, 6.0])

    # (a) anisotropic (affine) warp: stretch axes unequally -> not a similarity,
    #     but affine fits exactly.
    A = np.diag([1.0, 1.6, 0.7]) @ rand_rotation(RNG)
    Yaff = (A @ X.T).T + np.array([1.0, 0.0, 2.0])
    L = align.diagnose(X, Yaff)
    check("affine: similarity fails", L["similarity"] / L["scene"] > 1e-3,
          f"{L['similarity']/L['scene']:.3f}")
    check("affine: affine fits", L["affine"] / L["scene"] < 1e-9)
    check("affine: verdict needs_affine", L["verdict"] == "needs_affine", L["verdict"])

    # (b) nonlinear warp: depth-quadratic stretch (focal-error fingerprint) ->
    #     even affine cannot remove it, and the residual correlates with depth.
    Z = X[:, 2]
    Ynl = X.copy()
    Ynl[:, 0] *= (1.0 + 0.15 * (Z - Z.mean()) ** 2 / Z.var())
    L2 = align.diagnose(X, Ynl)
    check("nonlinear: affine fails", L2["affine"] / L2["scene"] > 1e-3,
          f"{L2['affine']/L2['scene']:.3f}")
    check("nonlinear: verdict nonlinear", L2["verdict"] == "nonlinear", L2["verdict"])
    check("nonlinear: depth-correlated residual", L2["structured"],
          f"corr={L2['depth_corr']:.2f}")


def test_outlier_robustness():
    """RANSAC similarity survives 30% gross floaters in the overlap."""
    print("test_outlier_robustness")
    n = 400
    X = RNG.normal(size=(n, 3)) * 2.0
    s_t, R_t, t_t = 1.3, rand_rotation(RNG), np.array([0.5, -2.0, 1.0])
    Y = align.apply_sim((s_t, R_t, t_t), X)
    n_out = int(0.3 * n)
    out_idx = RNG.choice(n, n_out, replace=False)
    Y[out_idx] += RNG.normal(size=(n_out, 3)) * 10.0           # floaters
    s, R, t, inl = align.fit_similarity_ransac(X, Y, thresh=0.05)
    check("scale recovered", abs(s - s_t) < 1e-3, f"{s:.6f} vs {s_t}")
    check("rotation recovered", rot_angle(R @ R_t.T) < 0.1)
    check("translation recovered", np.linalg.norm(t - t_t) < 1e-2)
    check("outliers excluded", (~inl[out_idx]).mean() > 0.9,
          f"{(~inl[out_idx]).mean()*100:.0f}% of outliers rejected")


def test_loop_closure():
    """A closed loop of consistent links ~= identity; injected scale drift shows."""
    print("test_loop_closure")
    links = []
    for _ in range(4):
        links.append((RNG.uniform(0.8, 1.2), rand_rotation(RNG), RNG.normal(size=3)))
    # close the loop exactly: last link = inverse of the composed first three
    T = align.identity()
    for Ti in links[:3]:
        T = align.compose(Ti, T)
    links_closed = links[:3] + [align.invert(T)]
    e = align.loop_closure_error(links_closed)
    check("closed loop ~ identity", e["scale_err"] < 1e-9 and e["rot_deg"] < 1e-6
          and e["trans"] < 1e-9, f"scale_err={e['scale_err']:.1e}")
    # now inject a 12% scale error into one link
    bad = links_closed[:]
    bad[0] = (bad[0][0] * 1.12, bad[0][1], bad[0][2])
    e2 = align.loop_closure_error(bad)
    check("scale drift detected", abs(e2["scale_err"] - np.log(1.12)) < 1e-9,
          f"scale_err={e2['scale_err']:.4f} (expected {np.log(1.12):.4f})")


# ===========================================================================
# PnP (verified against analytic ground truth, numpy backend)
# ===========================================================================

def make_intrinsics(f=400.0, H=512, W=512):
    pp = np.array([W / 2.0, H / 2.0])
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    return K, pp, f, H, W


def project(Pw, R, t, K):
    Xc = (R @ Pw.T).T + t
    proj = (K @ Xc.T).T
    return proj[:, :2] / proj[:, 2:3], Xc[:, 2]


def test_focal_recovery():
    """Weiszfeld recovers a known focal from exact (point, pixel) pairs."""
    print("test_focal_recovery")
    K, pp, f, H, W = make_intrinsics(f=437.0)
    Xc = RNG.uniform([-2, -2, 4], [2, 2, 9], size=(500, 3))   # camera-frame, Z>0
    px = f * Xc[:, :2] / Xc[:, 2:3] + pp
    f_est = focal.estimate_focal(Xc, px, pp)
    check("exact focal", abs(f_est - f) < 1e-6, f"{f_est:.9f} vs {f}")
    px_noisy = px + RNG.normal(size=px.shape) * 0.5
    f_n = focal.estimate_focal(Xc, px_noisy, pp)
    check("focal under noise", abs(f_n - f) / f < 0.02, f"{f_n:.2f} vs {f}")


def test_pnp_recovery():
    """numpy DLT+RANSAC PnP recovers known per-view extrinsics (cheirality ok)."""
    print("test_pnp_recovery")
    K, pp, f, H, W = make_intrinsics()
    Pw = RNG.uniform([-2, -2, 4], [2, 2, 8], size=(500, 3))   # world == view-0 cam frame
    worst_R, worst_t = 0.0, 0.0
    for _ in range(8):
        # small motion so all points stay in front of the moved camera
        ax = RNG.normal(size=3)
        ax /= np.linalg.norm(ax)
        ang = np.radians(RNG.uniform(3, 20))
        K_ = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(ang) * K_ + (1 - np.cos(ang)) * K_ @ K_
        t = RNG.uniform(-0.6, 0.6, size=3)
        px, z = project(Pw, R, t, K)
        if np.any(z <= 0):
            continue
        c2w, inl = pnp.solve_pnp_numpy(Pw, px, K)
        w2c = np.linalg.inv(c2w)
        worst_R = max(worst_R, rot_angle(w2c[:3, :3] @ R.T))
        worst_t = max(worst_t, np.linalg.norm(w2c[:3, 3] - t))
    check("rotation recovered", worst_R < 1e-4, f"worst {worst_R:.2e} deg")
    check("translation recovered", worst_t < 1e-5, f"worst {worst_t:.2e}")


def test_pnp_outliers():
    """PnP RANSAC tolerates corrupted correspondences (floaters)."""
    print("test_pnp_outliers")
    K, pp, f, H, W = make_intrinsics()
    Pw = RNG.uniform([-2, -2, 4], [2, 2, 8], size=(600, 3))
    R = rand_rotation(RNG)
    # keep rotation modest so points stay in front
    R = np.eye(3) + 0.15 * (R - R.T)
    R, _ = np.linalg.qr(R)
    t = np.array([0.3, -0.2, 0.4])
    px, z = project(Pw, R, t, K)
    keep = z > 0
    Pw, px = Pw[keep], px[keep]
    n_out = int(0.25 * len(px))
    oi = RNG.choice(len(px), n_out, replace=False)
    px[oi] += RNG.uniform(-150, 150, size=(n_out, 2))          # corrupt pixels
    c2w, inl = pnp.solve_pnp_numpy(Pw, px, K, thresh_px=2.0, iters=300)
    w2c = np.linalg.inv(c2w)
    check("rotation under outliers", rot_angle(w2c[:3, :3] @ R.T) < 0.2,
          f"{rot_angle(w2c[:3, :3] @ R.T):.3f} deg")
    check("translation under outliers", np.linalg.norm(w2c[:3, 3] - t) < 0.02)
    check("outliers rejected", (~inl[oi]).mean() > 0.85,
          f"{(~inl[oi]).mean()*100:.0f}% rejected")


def main():
    tests = [
        test_similarity_roundtrip, test_scale_detection, test_nonlinear_detection,
        test_outlier_robustness, test_loop_closure,
        test_focal_recovery, test_pnp_recovery, test_pnp_outliers,
    ]
    for t in tests:
        t()
    print()
    if _fails:
        print(f"FAILED ({len(_fails)}): " + ", ".join(_fails))
        raise SystemExit(1)
    print("ALL POSE TESTS OK")


if __name__ == "__main__":
    main()
