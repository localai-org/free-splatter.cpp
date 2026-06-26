"""Empirical cross-run test on REAL engine output.

A photo that appears in two separate inferences (an overlapping pair sharing it)
is reconstructed twice, in two different coordinate systems. Its per-pixel
gaussian centers from the two runs are the SAME physical points described two
ways -- exact correspondences. Fitting a similarity between them and reading the
residual ladder answers the open question on real data: is the cross-run mismatch
a clean uniform-scale similarity, or does it carry nonlinear warp?

Usage:
    nix develop -c python3 pose/empirical.py A.f32 viewA B.f32 viewB [label]

A.f32/B.f32 are raw engine outputs [N,H,W,23]; viewA/viewB pick the shared photo
within each (e.g. it's view 1 of pair (000,004) and view 0 of pair (004,013)).
"""
import sys
import numpy as np
import align

C = 23
OPACITY = 15


def load_view(path, view, H=512, W=512):
    a = np.fromfile(path, np.float32)
    N = a.size // (H * W * C)
    a = a.reshape(N, H, W, C)
    g = a[view]
    return g[..., 0:3].reshape(-1, 3).astype(np.float64), g[..., OPACITY].reshape(-1)


def main():
    pA, vA, pB, vB = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    label = sys.argv[5] if len(sys.argv) > 5 else ""
    XA, oA = load_view(pA, vA)
    XB, oB = load_view(pB, vB)

    print(f"=== {label} ===")
    print(f"opacity A: [{oA.min():.3f},{oA.max():.3f}] mean {oA.mean():.3f} | "
          f"B: [{oB.min():.3f},{oB.max():.3f}] mean {oB.mean():.3f}")

    thr = 0.05
    mask = (oA > thr) & (oB > thr)
    A, B = XA[mask], XB[mask]
    print(f"valid (opacity>{thr} in both): {mask.sum()}/{mask.size} "
          f"({100*mask.mean():.1f}%)")
    if mask.sum() < 100:
        print("too few valid correspondences"); return

    scene = align._rms(A - A.mean(0))
    # robust fit B -> A; threshold = 2% of scene extent
    s, R, t, inl = align.fit_similarity_ransac(B, A, thresh=0.02 * scene, iters=400)
    print(f"scene extent (rms): {scene:.4f}")
    print(f"robust similarity B->A:  scale={s:.4f}  inliers={inl.sum()}/{len(A)} "
          f"({100*inl.mean():.1f}%)")

    # how cross-run-consistent is the WHOLE valid set under the robust fit?
    res_all = np.linalg.norm(align.apply_sim((s, R, t), B) - A, axis=1)
    print("cross-run consistency (all valid pixels, under robust similarity):")
    for q in (0.01, 0.02, 0.05, 0.10):
        print(f"   within {100*q:4.0f}% of scene extent: "
              f"{100*(res_all < q*scene).mean():5.1f}% of pixels")

    # residual ladder on the inlier (well-corresponded) surface
    L = align.diagnose(B[inl], A[inl])
    sc = L["scene"]
    print("residual ladder (inliers, RMS  |  % of scene extent):")
    for k in ("rigid", "similarity", "affine"):
        print(f"   {k:11s} {L[k]:.5f}   {100*L[k]/sc:6.2f}%")
    print(f"   recovered scale  {L['scale']:.4f}")
    print(f"   depth_corr       {L['depth_corr']:+.3f}  (structured={L['structured']})")
    print(f"   VERDICT          {L['verdict']}")

    # interpretation
    drop_scale = (L["rigid"] - L["similarity"]) / sc
    drop_aff = (L["similarity"] - L["affine"]) / sc
    print(f"interpretation: rigid->similarity drop {100*drop_scale:.2f}% "
          f"(=how much was uniform scale); similarity->affine drop "
          f"{100*drop_aff:.2f}% (=non-uniform warp).")


if __name__ == "__main__":
    main()
