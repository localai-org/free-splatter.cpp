"""Run FreeSplatter-scene on Tanks-and-Temples pairs; check recovered pose + focal
against GROUND TRUTH as a function of frame stride (baseline). The control: GT
poses are known-good, so error here is OUR pipeline or the model being out of
distribution -- not unknown data.

Metrics per pair (view0 -> view1):
  GTrot    : GT relative rotation (deg) = the viewpoint change we asked for.
  valid%   : min over views of opacity>0.05 fraction (PnP support).
  poseErr  : angle between our recovered relative rotation and GT (deg) -- the
             scale-free "is the pose right" check.
  transDir : cos angle between our relative-translation direction and GT (1=perfect;
             FreeSplatter's metric scale is arbitrary, so only direction is checked).
  scaleRat : ||our t|| / ||GT t||  -- the monocular scale FreeSplatter chose vs metric.
  focalErr%: |recovered focal - GT focal(512)| / GT.

    FS_DEVICE=cpu nix develop -c python3 pose/tt_experiment.py Truck 120 2,5,10,20
"""
import os
import sys
import subprocess
import numpy as np

import tt_control as tt
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


def experiment(scene="Truck", anchor=120, strides=(2, 5, 10, 20)):
    frames, intr = tt.load_scene(scene)
    f_gt = intr["fx"] * 512.0 / 1080.0
    print(f"=== {scene}  anchor #{anchor} ({frames[anchor]['name']})  "
          f"GT focal(512)={f_gt:.1f}  device={DEVICE} ===")
    print(f"  {'stride':>6} {'GTrot':>6} {'valid%':>6} {'poseErr':>8} "
          f"{'transDir':>8} {'scaleRat':>8} {'focal':>7} {'fErr%':>6}")
    for s in strides:
        j = anchor + s
        if j >= len(frames):
            continue
        out = os.path.join(SCRATCH, f"tt_{scene}_{anchor}_{s}.f32")
        pts, op = run_engine(frames[anchor]["img"], frames[j]["img"], out)
        res = pnp.estimate_poses(pts, op, normalize=False)     # scene recipe
        c2w, f_est = res["cam2world"], res["focal"]
        rel = np.linalg.inv(c2w[0]) @ c2w[1]
        G = tt.rel_pose(frames[anchor]["c2w"], frames[j]["c2w"])
        poseErr = tt.rot_deg(rel[:3, :3] @ G[:3, :3].T)
        tdir = cos(rel[:3, 3], G[:3, 3])
        srat = np.linalg.norm(rel[:3, 3]) / (np.linalg.norm(G[:3, 3]) + 1e-12)
        validp = min((op[0] > 0.05).mean(), (op[1] > 0.05).mean()) * 100
        ferr = abs(f_est - f_gt) / f_gt * 100
        print(f"  {s:>6} {tt.rot_deg(G):>6.1f} {validp:>6.1f} {poseErr:>8.2f} "
              f"{tdir:>8.4f} {srat:>8.3f} {f_est:>7.1f} {ferr:>6.1f}")


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "Truck"
    anchor = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    strides = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 \
        else (2, 5, 10, 20)
    experiment(scene, anchor, strides)
