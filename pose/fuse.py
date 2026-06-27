"""Consensus fusion -- remove the partner-dependent floaters (the edge noise).

Each per-pixel gaussian is conditioned on its partner view, so occlusion-edge /
depth-ambiguous points land differently per run = floaters. A REAL surface point,
though, is reconstructed by several overlapping frames and they agree in the global
frame. So: voxelize the accumulated cloud at the consistency scale and keep only
voxels corroborated by >= K DISTINCT frames, averaging the agreeing predictions
(which also denoises the surface). Singletons / scattered floaters are dropped.

This is the direct answer to "does accumulation remove the edge noise?" -- yes, once
you gate on cross-frame consensus. Reuses cached engine dumps from accumulate.py
(no new inference).

    nix develop -c python3 pose/fuse.py CLIP FRAME_DIR --cache DIR \\
        --stride 20 --count 13 --voxel 0.02 --k 2 --ply-raw raw.ply --ply-fused fused.ply
"""
import os
import argparse
import numpy as np

import align

THR = 0.05
C0 = 0.28209479177387814


def load(dump):
    a = np.fromfile(dump, np.float32).reshape(2, 512, 512, 23)
    pts = [a[v, ..., 0:3].astype(np.float64) for v in (0, 1)]
    op = [a[v, ..., 15].astype(np.float64) for v in (0, 1)]
    rgb = [np.clip(a[v, ..., 3:6] * C0 + 0.5, 0, 1) for v in (0, 1)]
    return pts, op, rgb


def fit_sim(X, Y):
    scene = align._rms(Y - Y.mean(0))
    s, R, t, _ = align.fit_similarity_ransac(X, Y, thresh=0.02 * scene, iters=300)
    return (s, R, t)


def write_ply(path, xyz, rgb):
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rec["r"], rec["g"], rec["b"] = c[:, 0], c[:, 1], c[:, 2]
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("frame_dir")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--count", type=int, default=13)
    ap.add_argument("--cache", default=os.environ.get("SCRATCH", "/tmp"))
    ap.add_argument("--voxel", type=float, default=0.02)     # fraction of scene extent
    ap.add_argument("--k", type=int, default=2)              # min distinct frames
    ap.add_argument("--ply-raw", default=None)
    ap.add_argument("--ply-fused", default=None)       # averaged consensus (denoised)
    ap.add_argument("--ply-kept", default=None)        # raw points in consensus voxels
    ap.add_argument("--ply-floaters", default=None)    # raw points dropped (support < K)
    a = ap.parse_args()

    idx = [a.start + k * a.stride for k in range(a.count + 1)]
    runs = []
    for k in range(a.count):
        dump = os.path.join(a.cache, f"pair_{idx[k]}_{idx[k+1]}.f32")
        pts, op, rgb = load(dump)
        runs.append({"pts": pts, "op": op, "rgb": rgb})

    # chain Sim(3) -> global (same as accumulate.py)
    T = [align.identity()]
    for k in range(1, a.count):
        XA = runs[k]["pts"][0].reshape(-1, 3); oA = runs[k]["op"][0].reshape(-1)
        XB = runs[k-1]["pts"][1].reshape(-1, 3); oB = runs[k-1]["op"][1].reshape(-1)
        m = (oA > THR) & (oB > THR)
        T.append(align.compose(T[k-1], fit_sim(XA[m], XB[m])))

    # accumulate every frame's gaussians, tagged with a source-frame id
    XYZ, RGB, FR = [], [], []
    def add(run, view, Tk, fid):
        op = runs[run]["op"][view].reshape(-1)
        keep = op > THR
        XYZ.append(align.apply_sim(Tk, runs[run]["pts"][view].reshape(-1, 3)[keep]))
        RGB.append(runs[run]["rgb"][view].reshape(-1, 3)[keep])
        FR.append(np.full(int(keep.sum()), fid, np.int64))
    add(0, 0, T[0], 0)
    for k in range(a.count):
        add(k, 1, T[k], k + 1)
    XYZ = np.concatenate(XYZ); RGB = np.concatenate(RGB); FR = np.concatenate(FR)

    # voxel multi-frame consensus
    ext = align._rms(XYZ - XYZ.mean(0))
    v = a.voxel * ext
    ijk = np.floor((XYZ - XYZ.min(0)) / v).astype(np.int64)
    uniq, inv = np.unique(ijk, axis=0, return_inverse=True)
    G = len(uniq)
    # distinct source-frames per voxel
    pairs = np.unique(np.stack([inv, FR], 1), axis=0)
    support = np.bincount(pairs[:, 0], minlength=G)
    cnt = np.bincount(inv, minlength=G).astype(float)
    mean = np.stack([np.bincount(inv, XYZ[:, c], minlength=G) / cnt for c in range(3)], 1)
    mrgb = np.stack([np.bincount(inv, RGB[:, c], minlength=G) / cnt for c in range(3)], 1)
    keep = support >= a.k

    print(f"=== consensus fusion  voxel={a.voxel:.3f}*extent  K>={a.k} ===")
    print(f"raw points: {len(XYZ):,}   voxels: {G:,}")
    hist = np.bincount(support)[:6]
    print("voxels by distinct-frame support: " +
          "  ".join(f"{i}f:{hist[i] if i < len(hist) else 0:,}" for i in range(1, 6)) +
          f"  6+f:{(support >= 6).sum():,}")
    print(f"FUSED points (>= {a.k} frames): {keep.sum():,}  "
          f"({100*keep.sum()/G:.0f}% of voxels, {100*(1-keep.sum()/len(XYZ)):.0f}% point reduction)")
    pkeep = keep[inv]                                  # per-point consensus mask
    print(f"per-point: kept {pkeep.sum():,}  floaters dropped {(~pkeep).sum():,} "
          f"({100*(~pkeep).mean():.0f}%)")
    if a.ply_raw:
        write_ply(a.ply_raw, XYZ, RGB); print(f"   raw      -> {a.ply_raw}")
    if a.ply_fused:
        write_ply(a.ply_fused, mean[keep], mrgb[keep]); print(f"   fused    -> {a.ply_fused}")
    if a.ply_kept:
        write_ply(a.ply_kept, XYZ[pkeep], RGB[pkeep]); print(f"   kept     -> {a.ply_kept}")
    if a.ply_floaters:
        write_ply(a.ply_floaters, XYZ[~pkeep], RGB[~pkeep]); print(f"   floaters -> {a.ply_floaters}")


if __name__ == "__main__":
    main()
