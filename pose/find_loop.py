"""Search RealEstate10K poses for a LOOPING clip -- camera wanders away from its
start then returns near it (with a similar viewing direction, so the first and last
frames overlap). That overlap is the loop-closure constraint we need to test drift
correction. Pose-only: no video downloads until a candidate is chosen.

A good loop: large max distance from the start (real motion) but small end-to-start
distance (returns), and small first<->last rotation (same view => the closing pair
(f_0, f_n) is a valid stereo pair the engine can reconstruct).

    nix develop -c python3 pose/find_loop.py [split] [limit] [--live N]
"""
import sys
import numpy as np

import re10k_control as rc
import re10k_fetch as rf


def score(fr):
    C = np.array([f["c2w"][:3, 3] for f in fr])
    d = np.linalg.norm(C - C[0], axis=1)
    maxd = d.max()
    if maxd < 1e-2:
        return None
    rotend = rc.rot_deg(rc.rel_pose(fr[0]["c2w"], fr[-1]["c2w"]))
    return {"n": len(fr), "maxd": float(maxd), "endd": float(d[-1]),
            "return_ratio": float(d[-1] / maxd), "rotend": rotend,
            "loopiness": float(maxd / (d[-1] + 0.05 * maxd))}


def search(split="test", limit=None, min_frames=80,
           max_return_ratio=0.25, max_rotend=45.0):
    rows = []
    clips = rc.list_clips(split)
    if limit:
        clips = clips[:limit]
    for cp in clips:
        try:
            url, fr = rc.parse_clip(cp)
        except Exception:
            continue
        if len(fr) < min_frames:
            continue
        s = score(fr)
        if not s or s["return_ratio"] > max_return_ratio or s["rotend"] > max_rotend:
            continue
        s["clip"], s["id"] = cp, rc.clip_video_id(url)
        rows.append(s)
    rows.sort(key=lambda r: r["loopiness"], reverse=True)
    return rows


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "test"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else None
    n_live = int(sys.argv[sys.argv.index("--live") + 1]) if "--live" in sys.argv else 0
    rows = search(split, limit)
    print(f"{len(rows)} loop candidates (split={split}, limit={limit})")
    print(f"  {'loopiness':>9} {'n':>4} {'maxd':>6} {'ret%':>5} {'rotEnd':>6}  id")
    shown = 0
    for r in rows[:30]:
        live = ""
        if shown < n_live:
            try:
                live = "  LIVE" if rf.is_live(rc.parse_clip(r["clip"])[0]) else "  dead"
            except Exception:
                live = "  ?"
            shown += 1
        print(f"  {r['loopiness']:>9.1f} {r['n']:>4} {r['maxd']:>6.2f} "
              f"{100*r['return_ratio']:>5.0f} {r['rotend']:>6.1f}  {r['id']}{live}")
