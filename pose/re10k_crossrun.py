"""Cross-run consistency on IN-DISTRIBUTION re10k -- the accumulation question.

A shared frame m is reconstructed by two overlapping pairs:
    run A = (m-s, m)  -> frame m is view 1 (gaussians in frame (m-s)'s world)
    run B = (m, m+s)  -> frame m is view 0 (gaussians in frame m's world)
Frame m's per-pixel gaussian centers from A and B are the SAME physical points in
two coordinate systems -- exact correspondences. Fit a robust similarity B->A and
read the residual ladder + how many pixels actually agree. High agreement => clean
accumulation; low => floaters dominate and we need consensus fusion.

This mirrors empirical.py but on data the model handles well (re10k), and sweeps
the baseline stride s so we can see whether a sweet spot maximizes consistency.

    FS_DEVICE=cpu nix develop -c python3 pose/re10k_crossrun.py CLIP.txt FRAME_DIR 120 20,40,80
"""
import os
import sys
import numpy as np

import re10k_control as rc
import re10k_experiment as ex          # run_engine
import align

SCRATCH = os.environ.get("SCRATCH", "/tmp")
THR = 0.05


def crossrun(clip, frame_dir, m, strides):
    _, frames = rc.parse_clip(clip)
    print(f"=== cross-run consistency: {os.path.basename(clip)}  shared frame #{m} ===")
    print(f"  {'stride':>6} {'GTrot(i-j)':>10} {'valid%':>6} {'scale':>6} "
          f"{'<1%':>5} {'<2%':>5} {'<5%':>5} {'<10%':>6} {'rigid%':>7} {'sim%':>6} "
          f"{'aff%':>6} {'verdict':>20}")
    fp = lambda i: os.path.join(frame_dir, f"f{i:04d}.png")
    for s in strides:
        i, j = m - s, m + s
        if not all(os.path.exists(fp(k)) for k in (i, m, j)):
            print(f"  {s:>6}  (missing a frame among {i},{m},{j})"); continue
        ptsA, opA = ex.run_engine(fp(i), fp(m), os.path.join(SCRATCH, f"xrA_{m}_{s}.f32"))
        ptsB, opB = ex.run_engine(fp(m), fp(j), os.path.join(SCRATCH, f"xrB_{m}_{s}.f32"))
        XA, oA = ptsA[1].reshape(-1, 3), opA[1].reshape(-1)    # m as view1 of A
        XB, oB = ptsB[0].reshape(-1, 3), opB[0].reshape(-1)    # m as view0 of B
        mask = (oA > THR) & (oB > THR)
        A, B = XA[mask], XB[mask]
        if mask.sum() < 200:
            print(f"  {s:>6}  (too few valid: {mask.sum()})"); continue
        scene = align._rms(A - A.mean(0))
        sc, R, t, inl = align.fit_similarity_ransac(B, A, thresh=0.02 * scene, iters=400)
        res = np.linalg.norm(align.apply_sim((sc, R, t), B) - A, axis=1)
        within = {q: 100 * (res < q * scene).mean() for q in (0.01, 0.02, 0.05, 0.10)}
        L = align.diagnose(B[inl], A[inl])
        gtrot = rc.rot_deg(rc.rel_pose(frames[i]["c2w"], frames[j]["c2w"]))
        print(f"  {s:>6} {gtrot:>10.2f} {100*mask.mean():>6.1f} {sc:>6.3f} "
              f"{within[0.01]:>5.0f} {within[0.02]:>5.0f} {within[0.05]:>5.0f} "
              f"{within[0.10]:>6.0f} {100*L['rigid']/L['scene']:>7.1f} "
              f"{100*L['similarity']/L['scene']:>6.1f} {100*L['affine']/L['scene']:>6.1f} "
              f"{L['verdict']:>20}")
    print("\n<q% = fraction of shared-frame pixels whose two reconstructions agree "
          "within q% of scene extent under the best similarity. High => accumulation\n"
          "is clean; low => most per-pixel gaussians are partner-dependent floaters "
          "(need consensus fusion).")


if __name__ == "__main__":
    clip, frame_dir = sys.argv[1], sys.argv[2]
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    strides = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 \
        else (20, 40, 80)
    crossrun(clip, frame_dir, m, strides)
