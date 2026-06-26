"""Quick pinhole projection of a colored PLY to a PNG (z-buffered point splat) --
a visual coherence check for the accumulated cloud, no 3D viewer needed.

    nix develop -c python3 pose/render_ply.py world.ply out.png [focal] [radius]
"""
import sys
import numpy as np
import cv2


def read_ply(path):
    with open(path, "rb") as f:
        n = 0
        while True:
            l = f.readline()
            if l.startswith(b"element vertex"):
                n = int(l.split()[-1])
            if l.strip() == b"end_header":
                break
        rec = np.frombuffer(f.read(n * 15),
                            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
    rgb = np.stack([rec["r"], rec["g"], rec["b"]], 1)
    return xyz, rgb


def render(xyz, rgb, focal, H=512, W=512, c2w=None, radius=1):
    w2c = np.eye(4) if c2w is None else np.linalg.inv(c2w)
    Xc = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
    good = Xc[:, 2] > 1e-3
    Xc, col = Xc[good], rgb[good]
    u = focal * Xc[:, 0] / Xc[:, 2] + W / 2
    v = focal * Xc[:, 1] / Xc[:, 2] + H / 2
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    img = np.zeros((H, W, 3), np.uint8)
    order = np.argsort(-Xc[:, 2])                       # far first; near overwrites
    ui, vi, col = ui[order], vi[order], col[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu, vv = ui + dx, vi + dy
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[inb], uu[inb]] = col[inb]
    return img


if __name__ == "__main__":
    ply, out = sys.argv[1], sys.argv[2]
    focal = float(sys.argv[3]) if len(sys.argv) > 3 else 274.0
    radius = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    xyz, rgb = read_ply(ply)
    img = render(xyz, rgb, focal, radius=radius)
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"rendered {len(xyz):,} pts -> {out}  ({(img.any(2)).mean()*100:.0f}% filled)")
