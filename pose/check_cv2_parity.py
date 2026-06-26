"""Verify the numpy DLT/RANSAC PnP against the EXACT upstream solver.

FreeSplatter uses cv2.solvePnPRansac(reprojErr=5, SOLVEPNP_SQPNP). This compares
our dependency-free reference backend to that exact call on synthetic scenes with
known ground truth -- both must recover the true pose AND agree with each other.
Needs cv2 (nix develop, after opencv was added to the flake).

    nix develop -c python3 pose/check_cv2_parity.py
"""
import numpy as np
import pnp

RNG = np.random.default_rng(7)
fails = []


def rot_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def rand_small_pose(rng):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = np.radians(rng.uniform(3, 20))
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
    t = rng.uniform(-0.6, 0.6, size=3)
    return R, t


def run(label, with_outliers):
    f = 400.0
    K = np.array([[f, 0, 256.0], [0, f, 256.0], [0, 0, 1.0]])
    worst = {"np_R": 0, "np_t": 0, "cv_R": 0, "cv_t": 0, "agree_R": 0, "agree_t": 0}
    for _ in range(10):
        Pw = RNG.uniform([-2, -2, 4], [2, 2, 8], size=(500, 3))
        R, t = rand_small_pose(RNG)
        Xc = (R @ Pw.T).T + t
        if np.any(Xc[:, 2] <= 0):
            continue
        px = (K @ Xc.T).T
        px = px[:, :2] / px[:, 2:3]
        if with_outliers:
            oi = RNG.choice(len(px), int(0.2 * len(px)), replace=False)
            px[oi] += RNG.uniform(-120, 120, size=(len(oi), 2))

        c2w_np, _ = pnp.solve_pnp_numpy(Pw, px, K, thresh_px=4.0, iters=300)
        c2w_cv, _ = pnp.solve_pnp_cv2(Pw, px, K)
        w_np, w_cv = np.linalg.inv(c2w_np), np.linalg.inv(c2w_cv)
        worst["np_R"] = max(worst["np_R"], rot_angle(w_np[:3, :3] @ R.T))
        worst["np_t"] = max(worst["np_t"], np.linalg.norm(w_np[:3, 3] - t))
        worst["cv_R"] = max(worst["cv_R"], rot_angle(w_cv[:3, :3] @ R.T))
        worst["cv_t"] = max(worst["cv_t"], np.linalg.norm(w_cv[:3, 3] - t))
        worst["agree_R"] = max(worst["agree_R"], rot_angle(w_np[:3, :3] @ w_cv[:3, :3].T))
        worst["agree_t"] = max(worst["agree_t"], np.linalg.norm(w_np[:3, 3] - w_cv[:3, 3]))

    tolR, tolt = (0.5, 0.05) if with_outliers else (1e-3, 1e-4)
    print(f"{label}: worst-of-10  "
          f"numpy(R={worst['np_R']:.2e}deg t={worst['np_t']:.2e})  "
          f"cv2(R={worst['cv_R']:.2e}deg t={worst['cv_t']:.2e})  "
          f"agree(R={worst['agree_R']:.2e}deg t={worst['agree_t']:.2e})")
    ok = (worst["np_R"] < tolR and worst["cv_R"] < tolR and
          worst["agree_R"] < tolR and worst["agree_t"] < tolt)
    if not ok:
        fails.append(label)
    return ok


print("PnP backend parity: numpy DLT/RANSAC  vs  upstream cv2.solvePnPRansac")
run("clean   ", with_outliers=False)
run("outliers", with_outliers=True)
print("PARITY OK" if not fails else f"PARITY FAILED: {fails}")
raise SystemExit(1 if fails else 0)
