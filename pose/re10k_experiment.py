"""Run FreeSplatter-scene on RealEstate10K pairs (IN-distribution) and check the
recovered relative pose + focal against ground truth, vs frame stride (baseline).

Unlike Tanks-and-Temples (out of distribution -> garbage), the model is known-good
here, so this is a real control: low poseErr confirms our PnP against an INDEPENDENT
GT (re10k poses), and agreement validates the re10k camera convention.

Frames must already be fetched into frame_dir as fNNNN.png (re10k_fetch.py); both
anchor and anchor+stride must be among the fetched indices.

    FS_DEVICE=cpu nix develop -c python3 pose/re10k_experiment.py CLIP.txt FRAME_DIR 40 20,40,80,160
"""
import os
import sys
import subprocess
import numpy as np

import re10k_control as rc
import pnp

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF = os.environ.get("FS_GGUF", os.path.join(HERE, "..", ".cache", "freesplatter-scene-f16.gguf"))
CLI = os.environ.get("FS_CLI", os.path.join(HERE, "..", "build", "release", "bin", "free_splatter-cli"))
DEVICE = os.environ.get("FS_DEVICE", "cpu")
SCRATCH = os.environ.get("SCRATCH", "/tmp")


def run_engine(img0, img1, out):
    r = subprocess.run([CLI, "--device", DEVICE, "--out", out, GGUF, img0, img1],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CLI failed: {r.stderr[-500:]}")
    a = np.fromfile(out, np.float32).reshape(2, 512, 512, 23)
    pts = [a[v, ..., 0:3].astype(np.float64) for v in (0, 1)]
    op = [a[v, ..., 15].astype(np.float64) for v in (0, 1)]
    return pts, op


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def experiment(clip, frame_dir, anchor, strides):
    url, frames = rc.parse_clip(clip)
    f_gt = rc.gt_focal_512(frames[0])
    print(f"=== {os.path.basename(clip)}  anchor #{anchor}  GT focal(512)={f_gt:.1f}  "
          f"device={DEVICE} ===\n{url}")
    print(f"  {'stride':>6} {'GTrot':>6} {'valid%':>6} {'poseErr':>8} "
          f"{'transDir':>8} {'scaleRat':>8} {'focal':>7} {'fErr%':>6}")
    for s in strides:
        j = anchor + s
        img0 = os.path.join(frame_dir, f"f{anchor:04d}.png")
        img1 = os.path.join(frame_dir, f"f{j:04d}.png")
        if not (os.path.exists(img0) and os.path.exists(img1)):
            print(f"  {s:>6}  (missing frame {anchor} or {j})"); continue
        out = os.path.join(SCRATCH, f"re_{anchor}_{s}.f32")
        pts, op = run_engine(img0, img1, out)
        res = pnp.estimate_poses(pts, op, normalize=False)
        c2w, f_est = res["cam2world"], res["focal"]
        rel = np.linalg.inv(c2w[0]) @ c2w[1]
        G = rc.rel_pose(frames[anchor]["c2w"], frames[j]["c2w"])
        poseErr = rc.rot_deg(rel[:3, :3] @ G[:3, :3].T)
        tdir = cos(rel[:3, 3], G[:3, 3])
        srat = np.linalg.norm(rel[:3, 3]) / (np.linalg.norm(G[:3, 3]) + 1e-12)
        validp = min((op[0] > 0.05).mean(), (op[1] > 0.05).mean()) * 100
        ferr = abs(f_est - f_gt) / f_gt * 100
        print(f"  {s:>6} {rc.rot_deg(G):>6.2f} {validp:>6.1f} {poseErr:>8.2f} "
              f"{tdir:>8.4f} {srat:>8.3f} {f_est:>7.1f} {ferr:>6.1f}")


if __name__ == "__main__":
    clip, frame_dir = sys.argv[1], sys.argv[2]
    anchor = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    strides = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 \
        else (20, 40, 80, 160)
    experiment(clip, frame_dir, anchor, strides)
