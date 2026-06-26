"""Loop-closure drift correction on a LOOPING re10k clip (see find_loop.py).

accumulate.py chains Sim(3) open-loop and the drift grows to the endpoint (~11%
ATE). When the camera RETURNS near its start, frame 0 and frame n overlap -- a
loop-closure constraint. Here we:
  1. chain open-loop  -> per-frame global poses P_k, recovered trajectory, ATE vs GT.
  2. measure the loop -> run the closing pair (f_0, f_n), scale-aligned to run 0,
     giving f_n's pose via the loop (P_n_loop), INDEPENDENT of the chain.
  3. the discrepancy D (P_n_loop = D . P_n) IS the accumulated drift (scale/rot/trans).
  4. distribute D over the chain (even Sim(3) relaxation, D^(k/n)) and show the
     trajectory ATE and endpoint error collapse.

Closes the drift story accumulate.py left open; exercises align's chaining on a
real loop. Poses are 4x4 similarity matrices [[sR,t],[0,1]] (compose = matmul).

    FS_DEVICE=cpu nix develop -c python3 pose/loop_closure.py CLIP FRAME_DIR \\
        --stride 20 --count 13 --cache DIR
"""
import os
import argparse
import numpy as np

import re10k_control as rc
import re10k_fetch as rf
import re10k_experiment as ex
import pnp
import align

THR = 0.05
simmat = align.sim_matrix          # Sim(3) 4x4
matfrac = align.sim_frac_power     # M^f for even loop-closure relaxation


def fit_sim_M(X, Y):
    scene = align._rms(Y - Y.mean(0))
    s, R, t, inl = align.fit_similarity_ransac(X, Y, thresh=0.02 * scene, iters=300)
    return simmat(s, R, t), inl


def decompose(M):
    A = M[:3, :3]
    s = float(np.linalg.det(A) ** (1.0 / 3.0))
    R = A / s
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return s, ang, M[:3, 3]


def load(dump):
    a = np.fromfile(dump, np.float32).reshape(2, 512, 512, 23)
    pts = [a[v, ..., 0:3].astype(np.float64) for v in (0, 1)]
    op = [a[v, ..., 15].astype(np.float64) for v in (0, 1)]
    return pts, op


def ate(C, gt):
    sA, RA, tA = align.fit_similarity(C, gt)
    Cal = align.apply_sim((sA, RA, tA), C)
    return align._rms(Cal - gt), np.linalg.norm(Cal - gt, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("frame_dir")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--count", type=int, default=13)
    ap.add_argument("--cache", default=os.environ.get("SCRATCH", "/tmp"))
    a = ap.parse_args()

    url, frames = rc.parse_clip(a.clip)
    idx = [a.start + k * a.stride for k in range(a.count + 1)]
    os.makedirs(a.cache, exist_ok=True)
    fp = lambda i: os.path.join(a.frame_dir, f"f{i:04d}.png")
    if any(not os.path.exists(fp(i)) for i in idx):
        rf.fetch(a.clip, a.frame_dir, indices=idx)
    print(f"=== loop closure: {os.path.basename(a.clip)}  frames {idx[0]}..{idx[-1]} "
          f"stride {a.stride}  device={ex.DEVICE} ===\n{url}")

    # --- run / load chain pairs ---
    runs = []
    for k in range(a.count):
        dump = os.path.join(a.cache, f"pair_{idx[k]}_{idx[k+1]}.f32")
        if not os.path.exists(dump):
            ex.run_engine(fp(idx[k]), fp(idx[k + 1]), dump)
        pts, op = load(dump)
        c2w = pnp.estimate_poses(pts, op, normalize=False)["cam2world"]
        runs.append({"pts": pts, "op": op, "c2w": c2w})

    # --- open-loop chain: T[k] maps run k -> global ---
    T = [np.eye(4)]; chain_diag = []
    for k in range(1, a.count):
        XA = runs[k]["pts"][0].reshape(-1, 3); oA = runs[k]["op"][0].reshape(-1)
        XB = runs[k-1]["pts"][1].reshape(-1, 3); oB = runs[k-1]["op"][1].reshape(-1)
        m = (oA > THR) & (oB > THR)
        S, inlk = fit_sim_M(XA[m], XB[m]); T.append(T[k-1] @ S)
        sk, _, _ = decompose(S)
        chain_diag.append((sk, 100 * inlk.mean(), 100 * m.mean()))
    P = [T[k] @ runs[k]["c2w"][0] for k in range(a.count)]
    P.append(T[a.count-1] @ runs[a.count-1]["c2w"][1])      # f_n from last run's view1
    centers = np.array([p[:3, 3] for p in P])
    gt = np.array([frames[i]["c2w"][:3, 3] for i in idx])
    ext = align._rms(gt - gt.mean(0))

    # --- loop measurement: closing pair (f_0, f_n) ---
    cdump = os.path.join(a.cache, f"close_{idx[0]}_{idx[-1]}.f32")
    if not os.path.exists(cdump):
        ex.run_engine(fp(idx[0]), fp(idx[-1]), cdump)
    cpts, cop = load(cdump)
    cc2w = pnp.estimate_poses(cpts, cop, normalize=False)["cam2world"]
    # scale/pose-align the closing run's f_0 (view0) to run 0's f_0 (view0, == global)
    XA = cpts[0].reshape(-1, 3); oA = cop[0].reshape(-1)
    XB = runs[0]["pts"][0].reshape(-1, 3); oB = runs[0]["op"][0].reshape(-1)
    m = (oA > THR) & (oB > THR)
    close_valid = 100 * m.mean()
    G, inl = fit_sim_M(XA[m], XB[m])
    P_n_loop = G @ cc2w[1]                                  # f_n via the loop, in global

    # --- drift D (P_n_loop = D . P_n) and even distribution ---
    D = P_n_loop @ np.linalg.inv(P[-1])
    s, ang, t = decompose(D)
    n = a.count
    centers_c = np.array([(matfrac(D, k / n) @ np.append(centers[k], 1.0))[:3]
                          for k in range(len(centers))])

    # --- report ---
    ate0, perr0 = ate(centers, gt)
    ate1, perr1 = ate(centers_c, gt)
    # diagnostics: is the open-loop chain a clean drift, and does it even return?
    sA, RA, tA = align.fit_similarity(centers, gt)
    pnl_al = align.apply_sim((sA, RA, tA), P_n_loop[:3, 3])
    print("\nchain per-link scale/inlier%/valid%: " +
          "  ".join(f"{c[0]:.2f}/{c[1]:.0f}/{c[2]:.0f}" for c in chain_diag))
    print(f"return-to-start dist: recovered {np.linalg.norm(centers[-1]-centers[0]):.3f}  "
          f"GT {np.linalg.norm(gt[-1]-gt[0]):.3f}  (GT extent {ext:.3f})")
    print(f"loop measurement P_n_loop endpoint err vs GT: "
          f"{100*np.linalg.norm(pnl_al-gt[-1])/ext:.0f}% of extent")
    print(f"\nclosing pair (f_{idx[0]}, f_{idx[-1]}): valid%={close_valid:.1f}  "
          f"inliers={100*inl.mean():.0f}%")
    print(f"LOOP-CLOSURE ERROR (open-loop drift at the loop point):")
    print(f"   scale {s:.3f} ({100*abs(np.log(s)):.1f}% log)  rotation {ang:.1f} deg  "
          f"translation {np.linalg.norm(t):.3f} ({100*np.linalg.norm(t)/ext:.0f}% of extent)")
    print(f"\ntrajectory ATE vs GT (after global Sim(3) align):")
    print(f"   open-loop : {100*ate0/ext:5.1f}% of extent   endpoint {100*perr0[-1]/ext:.0f}%")
    print(f"   closed    : {100*ate1/ext:5.1f}% of extent   endpoint {100*perr1[-1]/ext:.0f}%")
    print(f"   per-frame % (open ): " + " ".join(f"{100*e/ext:.0f}" for e in perr0))
    print(f"   per-frame % (closed): " + " ".join(f"{100*e/ext:.0f}" for e in perr1))


if __name__ == "__main__":
    main()
