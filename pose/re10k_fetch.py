"""Fetch a RealEstate10K clip's frames via yt-dlp + ffmpeg -- the standard re10k
workflow, since the dataset ships only poses + YouTube URLs (frames can't be
redistributed). Gives us a dense, in-distribution, GT-posed sequence to drive the
control. Frames are written as fNNNN.png (NNNN = pose-line index) so they pair 1:1
with re10k_control.parse_clip()'s frames.

    nix develop -c python3 pose/re10k_fetch.py CLIP.txt OUT_DIR [step]

yt-dlp comes from `nix run nixpkgs#yt-dlp`; ffmpeg is on PATH. Dead/blocked videos
raise -- use find_live_clip to skip them.
"""
import os
import subprocess
import re10k_control as rc


def _ytdlp(*a):
    return ["nix", "run", "nixpkgs#yt-dlp", "--", *a]


def is_live(url, timeout=60):
    r = subprocess.run(_ytdlp("--no-warnings", "--simulate", "--get-duration", url),
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0


def download_video(url, out_dir, height=720, timeout=900):
    os.makedirs(out_dir, exist_ok=True)
    vid = os.path.join(out_dir, "video.mp4")
    if os.path.exists(vid) and os.path.getsize(vid) > 0:
        return vid
    fmt = (f"bv*[height<={height}][ext=mp4]/bv*[height<={height}]/"
           f"b[height<={height}]/best")
    r = subprocess.run(_ytdlp("-f", fmt, "--no-warnings", "--no-playlist",
                              "-o", vid, url),
                       capture_output=True, text=True, timeout=timeout)
    if not (os.path.exists(vid) and os.path.getsize(vid) > 0):
        raise RuntimeError("yt-dlp failed:\n" + r.stderr[-600:])
    return vid


def extract_frames(vid, frames, indices, out_dir):
    imgs = {}
    for i in indices:
        ts = frames[i]["ts"] / 1e6                     # absolute video time (s)
        p = os.path.join(out_dir, f"f{i:04d}.png")
        if not os.path.exists(p):
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error",
                            "-ss", f"{ts:.6f}", "-i", vid, "-frames:v", "1",
                            "-q:v", "2", "-y", p], check=True, timeout=120)
        imgs[i] = p
    return imgs


def fetch(clip_path, out_dir, indices=None, step=10, height=720):
    url, frames = rc.parse_clip(clip_path)
    if indices is None:
        indices = list(range(0, len(frames), step))
    vid = download_video(url, out_dir, height)
    imgs = extract_frames(vid, frames, indices, out_dir)
    return url, frames, imgs


def find_live_clip(split="test", start=0, limit=300, min_frames=120):
    """First clip in [start, start+limit) that has >= min_frames AND a live video."""
    for cp in rc.list_clips(split)[start:start + limit]:
        _, fr = rc.parse_clip(cp)
        if len(fr) < min_frames:
            continue
        try:
            if is_live(rc.parse_clip(cp)[0]):
                return cp
        except subprocess.TimeoutExpired:
            continue
    return None


if __name__ == "__main__":
    import sys
    clip, out_dir = sys.argv[1], sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    url, frames, imgs = fetch(clip, out_dir, step=step)
    print(f"got {len(imgs)} frames from {url}")
    print("indices:", sorted(imgs)[:6], "...", sorted(imgs)[-3:])
