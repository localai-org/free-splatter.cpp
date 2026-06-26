"""Tanks-and-Temples (NSVF release) loader -- the dense, GT-posed control.

A known-good, ground-truth-posed sequence so that when the live FreeSplatter path
misbehaves we can tell whether it's the data/model or our code. NSVF layout per
scene (.cache/tt/TanksAndTemple/<Scene>/):
    rgb/{split}_{idx}_{origframe}.png   1920x1080
    pose/{split}_{idx}_{origframe}.txt  4x4 camera-to-world, OpenGL convention
    intrinsics.txt                       4x4 (fx 0 cx 0 / 0 fy cy 0 / ...)
split 0 = train, 1 = test (held-out orbit views). Frames sorted by origframe give
true capture order.

What this gives the control:
  * GT relative pose between any two frames  -> check our PnP relative pose.
  * GT focal after 512x512 preprocessing     -> check FreeSplatter's recovered focal.
  * GT baseline / parallax per frame-stride   -> drive the baseline sweep.

OpenGL->OpenCV: negate the camera y,z axes (columns 1,2 of the c2w rotation),
matching the runner's `c2ws[:, :, 1:3] *= -1`. cam2world, OpenCV (x right, y down,
z forward) -- the same convention our pnp.estimate_poses returns.

Usage:
    nix develop -c python3 pose/tt_control.py [Scene]      # default Truck
"""
import os
import re
import sys
import numpy as np

TT_ROOT = os.environ.get(
    "TT_ROOT",
    os.path.join(os.path.dirname(__file__), "..", ".cache", "tt", "TanksAndTemple"))

_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])      # negate camera y,z axes (right-mul)


def gl_to_cv(c2w):
    """OpenGL camera-to-world -> OpenCV camera-to-world."""
    return c2w @ _GL2CV


def load_intrinsics(scene_dir):
    M = np.loadtxt(os.path.join(scene_dir, "intrinsics.txt"))
    return {"fx": float(M[0, 0]), "fy": float(M[1, 1]),
            "cx": float(M[0, 2]), "cy": float(M[1, 2])}


def _origframe(name):
    # {split}_{idx}_{origframe}.ext -> int(origframe)
    return int(re.split(r"[_.]", name)[2])


def load_scene(scene, split=None):
    """Return (frames, intr). frames: list of dicts {name, idx, orig, img, c2w(OpenCV)}
    sorted by capture order. split=0/1 to filter train/test, None for all."""
    scene_dir = scene if os.path.isdir(scene) else os.path.join(TT_ROOT, scene)
    intr = load_intrinsics(scene_dir)
    pose_dir, rgb_dir = os.path.join(scene_dir, "pose"), os.path.join(scene_dir, "rgb")
    frames = []
    for pf in os.listdir(pose_dir):
        if not pf.endswith(".txt"):
            continue
        sp = int(pf.split("_")[0])
        if split is not None and sp != split:
            continue
        c2w_gl = np.loadtxt(os.path.join(pose_dir, pf))
        img = os.path.join(rgb_dir, pf[:-4] + ".png")
        frames.append({"name": pf[:-4], "split": sp, "orig": _origframe(pf),
                       "img": img, "c2w": gl_to_cv(c2w_gl)})
    frames.sort(key=lambda f: f["orig"])
    return frames, intr


def rel_pose(c2w_a, c2w_b):
    """Camera b expressed in camera a's frame: inv(c2w_a) @ c2w_b (OpenCV)."""
    return np.linalg.inv(c2w_a) @ c2w_b


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R[:3, :3]) - 1) / 2, -1, 1))))


def scene_radius(frames):
    """Mean camera-center distance to the centroid of all camera centers."""
    cen = np.array([f["c2w"][:3, 3] for f in frames])
    return float(np.linalg.norm(cen - cen.mean(0), axis=1).mean()), cen.mean(0)


def baseline_report(scene="Truck"):
    frames, intr = load_scene(scene)
    n = len(frames)
    radius, _ = scene_radius(frames)
    f512 = intr["fx"] * 512.0 / 1080.0            # after center-crop 1080 + resize 512
    print(f"=== {scene}: {n} frames "
          f"(train {sum(f['split']==0 for f in frames)}, "
          f"test {sum(f['split']==1 for f in frames)}) ===")
    print(f"intrinsics 1920x1080: fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
          f"cx={intr['cx']:.1f} cy={intr['cy']:.1f}")
    print(f"GT focal after 512x512 preprocess (center-crop 1080 -> resize): "
          f"{f512:.1f} px  (pp=256)")
    print(f"orbit radius (mean cam dist to center): {radius:.4f}")
    print("\nframe-stride sweep -- viewpoint change between frame i and i+stride:")
    print(f"  {'stride':>6} {'pairs':>6} {'rot(deg) med':>13} {'p10..p90':>16} "
          f"{'baseline/r med':>15} {'parallax(deg)':>14}")
    for s in (1, 2, 3, 5, 10, 20, 40):
        rots, bl = [], []
        for i in range(0, n - s):
            T = rel_pose(frames[i]["c2w"], frames[i + s]["c2w"])
            rots.append(rot_deg(T))
            bl.append(np.linalg.norm(T[:3, 3]))
        if not rots:
            continue
        rots, bl = np.array(rots), np.array(bl)
        blr = bl / radius
        # small-angle parallax ~ baseline / radius (rad) -> deg
        par = np.degrees(np.median(blr))
        print(f"  {s:>6} {len(rots):>6} {np.median(rots):>13.2f} "
              f"{np.percentile(rots,10):>7.2f}..{np.percentile(rots,90):<7.2f} "
              f"{np.median(blr):>15.3f} {par:>14.2f}")
    print("\nUse this to pick pair strides for the sweet-spot sweep: too small a "
          "stride -> ~0 parallax (degenerate depth); too large -> low overlap.")


if __name__ == "__main__":
    baseline_report(sys.argv[1] if len(sys.argv) > 1 else "Truck")
