"""Sliding-window accumulation prototype -- assemble the validated pieces.

Slide a window of overlapping pairs across a re10k clip:
    pair k = (f_k, f_{k+1})   ->  gaussians in f_k's camera frame (run k's world)
Consecutive runs share a frame (f_{k+1} is view1 of pair k and view0 of pair k+1),
so their per-pixel gaussians for that frame are exact correspondences. Fit a Sim(3)
S_k mapping run k's frame into run k-1's frame, and COMPOSE into a global transform
    T_0 = I ,  T_k = T_{k-1} . S_k
that drops every run's gaussians into ONE world (frame f_0's). Accumulate them,
and measure the recovered camera trajectory against ground truth (drift).

This is the live pipeline minus realtime: PnP (pnp.py) + Sim(3) align (align.py) +
chaining (align.compose) over a real sequence, justified by re10k_crossrun.py.

    FS_DEVICE=cpu nix develop -c python3 pose/accumulate.py CLIP.txt FRAME_DIR \\
        --start 0 --stride 20 --count 13 --cache DIR --ply out.ply

Engine dumps are cached (pair_*.f32); re-runs skip inference.
"""
import os
import argparse
import numpy as np

import re10k_control as rc
import re10k_fetch as rf
import re10k_experiment as ex          # run_engine
import pnp
import align

C0 = 0.28209479177387814
THR = 0.05


def load_dump(path):
    a = np.fromfile(path, np.float32).reshape(2, 512, 512, 23)
    pts = [a[v, ..., 0:3].astype(np.float64) for v in (0, 1)]
    op = [a[v, ..., 15].astype(np.float64) for v in (0, 1)]
    rgb = [np.clip(a[v, ..., 3:6] * C0 + 0.5, 0, 1) for v in (0, 1)]
    return pts, op, rgb


def write_ply(path, xyz, rgb):
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rec["r"], rec["g"], rec["b"] = c[:, 0], c[:, 1], c[:, 2]
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("frame_dir")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--count", type=int, default=13)        # number of pairs
    ap.add_argument("--cache", default=os.environ.get("SCRATCH", "/tmp"))
    ap.add_argument("--ply", default=None)
    ap.add_argument("--max-points", type=int, default=2_000_000)
    a = ap.parse_args()

    url, frames = rc.parse_clip(a.clip)
    idx = [a.start + k * a.stride for k in range(a.count + 1)]      # n+1 frame indices
    assert idx[-1] < len(frames), f"frame {idx[-1]} beyond clip ({len(frames)})"
    # make sure the frames exist (fetch on demand)
    missing = [i for i in idx if not os.path.exists(os.path.join(a.frame_dir, f"f{i:04d}.png"))]
    if missing:
        rf.fetch(a.clip, a.frame_dir, indices=idx)
    os.makedirs(a.cache, exist_ok=True)
    fp = lambda i: os.path.join(a.frame_dir, f"f{i:04d}.png")

    print(f"=== accumulate: {os.path.basename(a.clip)}  frames {idx[0]}..{idx[-1]} "
          f"stride {a.stride}  ({a.count} pairs)  device={ex.DEVICE} ===\n{url}")

    # --- run / load each pair, recover its relative pose ---
    runs = []           # per pair: dict(pts, op, rgb, c2w)
    for k in range(a.count):
        dump = os.path.join(a.cache, f"pair_{idx[k]}_{idx[k+1]}.f32")
        if not os.path.exists(dump):
            ex.run_engine(fp(idx[k]), fp(idx[k + 1]), dump)
            print(f"  ran pair {k}: ({idx[k]},{idx[k+1]})")
        pts, op, rgb = load_dump(dump)
        res = pnp.estimate_poses(pts, op, normalize=False)
        runs.append({"pts": pts, "op": op, "rgb": rgb, "c2w": res["cam2world"]})

    # --- chain Sim(3): T_k maps run k's frame -> global (frame f_0) ---
    T = [align.identity()]
    chain_info = []
    for k in range(1, a.count):
        XA, oA = runs[k]["pts"][0].reshape(-1, 3), runs[k]["op"][0].reshape(-1)     # f_k as view0 of run k
        XB, oB = runs[k-1]["pts"][1].reshape(-1, 3), runs[k-1]["op"][1].reshape(-1) # f_k as view1 of run k-1
        m = (oA > THR) & (oB > THR)
        scene = align._rms(XB[m] - XB[m].mean(0))
        s, R, t, inl = align.fit_similarity_ransac(XA[m], XB[m], thresh=0.02 * scene, iters=300)
        S_k = (s, R, t)
        T.append(align.compose(T[k-1], S_k))
        resid = align._rms(align.apply_sim(S_k, XA[m][inl]) - XB[m][inl]) / scene
        chain_info.append((k, s, 100*inl.mean(), 100*m.mean(), 100*resid))

    # --- accumulate gaussians into the global frame (each frame once) ---
    XYZ, RGB = [], []
    def add(run_idx, view, Tk):
        pts = runs[run_idx]["pts"][view].reshape(-1, 3)
        op = runs[run_idx]["op"][view].reshape(-1)
        rgb = runs[run_idx]["rgb"][view].reshape(-1, 3)
        keep = op > THR
        XYZ.append(align.apply_sim(Tk, pts[keep])); RGB.append(rgb[keep])
    add(0, 0, T[0])                                  # f_0 (view0 of pair 0)
    for k in range(a.count):                         # f_{k+1} = view1 of pair k
        add(k, 1, T[k])
    XYZ = np.concatenate(XYZ); RGB = np.concatenate(RGB)

    # --- recovered camera trajectory (global) vs GT ---
    rec = [align.apply_sim(T[k], runs[k]["c2w"][0][:3, 3]) for k in range(a.count)]
    rec.append(align.apply_sim(T[a.count-1], runs[a.count-1]["c2w"][1][:3, 3]))
    rec = np.array(rec)
    gt = np.array([frames[i]["c2w"][:3, 3] for i in idx])
    sA, RA, tA = align.fit_similarity(rec, gt)        # align recovered traj -> GT
    rec_al = align.apply_sim((sA, RA, tA), rec)
    perr = np.linalg.norm(rec_al - gt, axis=1)
    gt_ext = align._rms(gt - gt.mean(0))

    # --- report ---
    print("\nchain (per shared frame): step  scale  inlier%  valid%  resid%scene")
    for k, s, inlp, vp, rp in chain_info:
        print(f"   {k:>3}   {s:>6.3f}  {inlp:>6.1f}  {vp:>6.1f}  {rp:>7.2f}")
    total_scale = np.prod([ci[1] for ci in chain_info]) if chain_info else 1.0
    print(f"cumulative scale drift over {a.count-1} links: {total_scale:.3f} "
          f"(per-link mean {total_scale**(1/max(a.count-1,1)):.4f})")

    print(f"\ntrajectory vs GT ({len(idx)} frames, after global Sim(3) align):")
    print(f"   ATE rms = {align._rms(rec_al-gt):.4f}  ({100*align._rms(rec_al-gt)/gt_ext:.1f}% "
          f"of GT extent {gt_ext:.4f})")
    print(f"   per-frame error % of extent: " +
          " ".join(f"{100*e/gt_ext:.0f}" for e in perr))
    print(f"   drift: first-half {100*perr[:len(perr)//2].mean()/gt_ext:.1f}%  "
          f"second-half {100*perr[len(perr)//2:].mean()/gt_ext:.1f}%")

    print(f"\naccumulated cloud: {len(XYZ):,} points (opacity>{THR})")
    if a.ply:
        if len(XYZ) > a.max_points:
            sel = np.random.default_rng(0).choice(len(XYZ), a.max_points, replace=False)
            XYZ, RGB = XYZ[sel], RGB[sel]
        write_ply(a.ply, XYZ, RGB)
        print(f"   wrote {a.ply} ({len(XYZ):,} points)")


if __name__ == "__main__":
    main()
