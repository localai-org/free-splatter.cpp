"""RealEstate10K loader -- the dense, GT-posed, IN-DISTRIBUTION control.

FreeSplatter-scene is trained on RealEstate10K-style wide-FOV walkthroughs, so
(unlike Tanks-and-Temples, which is out of distribution) the model is known-good
here. Each clip .txt is:

    line 1 : YouTube URL
    line k : timestamp(us)  fx fy cx cy  0 0  m00 m01 m02 m03 m10 ... m23
             ^int            ^normalized intrinsics  ^3x4 world->camera, row-major

Intrinsics are resolution-independent (fx normalized by width, fy by height).
Frames come from the YouTube video at the given timestamps (yt-dlp + ffmpeg).

Camera convention: the pose is world->camera extrinsic P=[R|t] (KP maps a world
point to the image), OpenCV-style. c2w = inv([[R,t],[0,0,0,1]]). We VALIDATE this
convention empirically by checking that GT relative rotation agrees with the pose
FreeSplatter recovers on the same pair (the model is accurate in-distribution).

GT focal after our 512x512 preprocessing (center-crop to height, resize): because
fy is normalized by height and we crop to the full height then scale to 512,
    focal_512 = fy * 512    (independent of source resolution).

Pure numpy for parsing; frame extraction shells out to yt-dlp/ffmpeg.
"""
import os
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RE10K = os.environ.get("RE10K_ROOT", os.path.join(HERE, "..", ".cache", "re10k", "RealEstate10K"))


def parse_clip(path):
    """Return (url, frames). frames: list of dicts {ts, fx,fy,cx,cy, w2c(3x4), c2w(4x4)}."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    url = lines[0]
    frames = []
    for ln in lines[1:]:
        v = ln.split()
        ts = int(v[0])
        fx, fy, cx, cy = (float(x) for x in v[1:5])
        # v[5], v[6] are zeros (distortion); v[7:19] are the 3x4 world->cam
        P = np.array([float(x) for x in v[7:19]], float).reshape(3, 4)
        w2c = np.eye(4)
        w2c[:3, :4] = P
        c2w = np.linalg.inv(w2c)
        frames.append({"ts": ts, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                       "w2c": w2c, "c2w": c2w})
    return url, frames


def gt_focal_512(frame):
    """GT focal in the engine's 512x512 input (center-crop to height + resize)."""
    return frame["fy"] * 512.0


def rel_pose(c2w_a, c2w_b):
    return np.linalg.inv(c2w_a) @ c2w_b


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R[:3, :3]) - 1) / 2, -1, 1))))


def list_clips(split="test"):
    d = os.path.join(RE10K, split)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".txt"))


def clip_video_id(url):
    """YouTube id from the clip header URL."""
    for key in ("watch?v=", "youtu.be/", "/embed/"):
        if key in url:
            return url.split(key)[1][:11]
    return url.rsplit("/", 1)[-1][:11]


def baseline_report(clip_path):
    url, fr = parse_clip(clip_path)
    n = len(fr)
    cen = np.array([f["c2w"][:3, 3] for f in fr])
    span = float(np.linalg.norm(cen - cen.mean(0), axis=1).mean())
    print(f"=== {os.path.basename(clip_path)}  ({n} frames)  {url} ===")
    print(f"video id: {clip_video_id(url)}   duration: "
          f"{(fr[-1]['ts']-fr[0]['ts'])/1e6:.1f}s   GT focal(512)~{gt_focal_512(fr[0]):.1f}")
    print(f"camera-center spread (mean dist to centroid): {span:.4f}")
    print(f"  {'stride':>6} {'pairs':>6} {'rot(deg) med':>13} {'p10..p90':>16} "
          f"{'baseline med':>13}")
    for s in (1, 2, 5, 10, 20, 40):
        rots, bl = [], []
        for i in range(0, n - s):
            T = rel_pose(fr[i]["c2w"], fr[i + s]["c2w"])
            rots.append(rot_deg(T)); bl.append(np.linalg.norm(T[:3, 3]))
        if not rots:
            continue
        rots, bl = np.array(rots), np.array(bl)
        print(f"  {s:>6} {len(rots):>6} {np.median(rots):>13.2f} "
              f"{np.percentile(rots,10):>7.2f}..{np.percentile(rots,90):<7.2f} "
              f"{np.median(bl):>13.4f}")


if __name__ == "__main__":
    import sys
    baseline_report(sys.argv[1])
