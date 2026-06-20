#!/usr/bin/env python3
"""
Root-drift benchmark across coordinate rotation variants.

Metrics per rotation case
─────────────────────────
D_drift   max_t ||r_t^{xy} - r_0^{xy}||_2   (worst-frame horizontal drift)
          Sensitive to rx. Blind to rz=±180° (L2 norm cancels sign).

f_fwd     mean signed projection of root displacement onto facing direction
          = mean_t  (r_t^{xy} - r_0^{xy}) · f̂^{xy}
          Positive  → root moves in the direction the avatar faces  (correct)
          Negative  → root moves OPPOSITE to where avatar faces     (mirrored)
          ≈0        → in-place motion (squat, jump) — check f_y instead

f_y       Y-component of the canonical forward vector after rotation.
          Purely rotation-matrix geometry — independent of motion data.
          > 0 → avatar faces +Y in HUGS world  (correct for rx=90, rz=180)
          < 0 → avatar faces -Y in HUGS world  (mirrored, rz=0 case)

Usage:
    python scripts/benchmark_root_drift.py \\
        --input path/to/hugs_smpl_original.npz [--tz 1.0] [--center]
"""

import argparse
from pathlib import Path
import numpy as np


# ─── rotation helpers ────────────────────────────────────────────────────────

def axis_angle_to_rotmat(aa, eps=1e-8):
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / np.clip(angle, eps, None)
    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(ax)
    K = np.stack([zero, -az, ay, az, zero, -ax, -ay, ax, zero], axis=-1
                 ).reshape(*ax.shape, 3, 3)
    I = np.broadcast_to(np.eye(3, dtype=np.float32), K.shape)
    sin = np.sin(angle)[..., 0][..., None, None]
    cos = np.cos(angle)[..., 0][..., None, None]
    R = I + sin * K + (1.0 - cos) * (K @ K)
    small = angle[..., 0] < eps
    if np.any(small):
        R[small] = I[small]
    return R


def euler_xyz_to_rotmat(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


# ─── metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(transl: np.ndarray, R_fix: np.ndarray,
                    global_orient_f0: np.ndarray):
    """
    Returns
    -------
    d_drift  : float  worst-frame ground-plane drift (L2, sign-blind)
    up_z     : float  Z-component of rotated body-up vector.
                      1.0=standing, 0.0=lying on side, -1.0=upside-down.
                      Motion-independent. Detects the rx=0 "lying down" problem.
    f_y      : float  Y-component of rotated canonical forward vector.
                      Positive=faces +Y (correct rz=180). Motion-independent.
                      Detects the rz=0 "facing wrong way" problem.
    f_fwd    : float  mean signed forward travel. Motion-dependent tiebreaker.
    fwd_vec  : ndarray (3,)
    face_lbl : str
    """
    # ── D_drift ──────────────────────────────────────────────────────────────
    r0_xy = transl[0, :2]
    diffs = transl[:, :2] - r0_xy
    d_drift = float(np.max(np.linalg.norm(diffs, axis=-1)))

    # ── body up-vector Z alignment (detects lying-down vs standing) ───────────
    # MDM canonical up = [0,1,0] (Y is up in HumanML3D).
    # After R_fix, its Z component tells us whether the body stands in HUGS space.
    up_z = float((R_fix @ np.array([0.0, 1.0, 0.0], dtype=np.float32))[2])

    # ── facing vector (geometry only, frame 0 global_orient) ─────────────────
    R0 = axis_angle_to_rotmat(global_orient_f0[None])[0]
    R_total = R_fix @ R0
    fwd_vec = R_total @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

    labels = ["+X", "+Y", "+Z", "-X", "-Y", "-Z"]
    axes   = np.eye(3, dtype=np.float32)
    signed = np.concatenate([axes, -axes], axis=0)
    face_lbl = labels[int(np.argmax(signed @ fwd_vec))]

    f_y = float(fwd_vec[1])

    # ── signed forward travel ─────────────────────────────────────────────────
    fwd_xy = fwd_vec[:2]
    norm_fwd = np.linalg.norm(fwd_xy)
    if norm_fwd > 1e-6:
        fwd_hat = fwd_xy / norm_fwd
        f_fwd = float(np.mean(diffs @ fwd_hat))
    else:
        f_fwd = 0.0

    return d_drift, up_z, f_y, f_fwd, fwd_vec, face_lbl


# ─── cases ───────────────────────────────────────────────────────────────────

CASES = [(0, 0), (90, 0), (0, 180), (90, 180)]


def run_benchmark(npz_path, center, tx, ty, tz, ground):
    data          = np.load(npz_path)
    global_orient = data["global_orient"].astype(np.float32)
    transl_raw    = data["transl"].astype(np.float32)

    print(f"\nInput : {npz_path}")
    print(f"Frames: {transl_raw.shape[0]}  |  "
          f"center={center}, tz={tz}, ground={ground}\n")

    W = 100
    print("─" * W)
    print(f"  {'rx':>5}  {'rz':>5}  {'D_drift':>9}  {'up_z':>6}  {'f_y':>6}  "
          f"{'f_fwd':>7}  {'Posture check'}")
    print("─" * W)

    for rx_deg, rz_deg in CASES:
        R_fix = euler_xyz_to_rotmat(np.deg2rad(rx_deg), 0.0, np.deg2rad(rz_deg))

        transl_new = (R_fix @ transl_raw.T).T
        if center:
            transl_new -= transl_new.mean(axis=0, keepdims=True)
        transl_new += np.array([tx, ty, tz], dtype=np.float32)
        if ground is not None:
            transl_new[:, 2] = transl_new[:, 2] - transl_new[:, 2].min() + ground

        d_drift, up_z, f_y, f_fwd, fwd_vec, face_lbl = compute_metrics(
            transl_new, R_fix, global_orient[0])

        # posture summary
        stand  = "STANDING" if abs(up_z)  > 0.9 and up_z  > 0 else \
                 "UPSIDE-DOWN" if up_z < -0.9 else "LYING DOWN"
        facing = "faces cam (+Y)" if f_y > 0.9 else \
                 "faces away (-Y)" if f_y < -0.9 else "faces sideways"
        posture = f"{stand} | {facing}"

        marker = "  ← current" if (rx_deg == 90 and rz_deg == 180) else ""
        print(f"  rx={rx_deg:+3d}°  rz={rz_deg:+4d}°  "
              f"{d_drift:>9.4f}  {up_z:>+6.3f}  {f_y:>+6.3f}  "
              f"{f_fwd:>+7.4f}  {posture}{marker}")

    print("─" * W)
    print()
    print("Metric guide")
    print("  D_drift : worst-frame ground-plane drift. "
          "Sensitive to rx. CANNOT detect lying-down for all motions.")
    print("  up_z    : body up-vector Z alignment. "
          "+1=standing, 0=lying on side, -1=upside-down. Motion-independent.")
    print("  f_y     : forward vector Y component. "
          "+1=faces cam, -1=faces away. Motion-independent.")
    print("  f_fwd   : mean signed forward travel. Motion-dependent tiebreaker.")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, type=Path)
    p.add_argument("--center", action="store_true")
    p.add_argument("--tx", type=float, default=0.0)
    p.add_argument("--ty", type=float, default=0.0)
    p.add_argument("--tz", type=float, default=0.0)
    p.add_argument("--ground", type=float, default=None)
    args = p.parse_args()
    run_benchmark(args.input, args.center, args.tx, args.ty, args.tz, args.ground)


if __name__ == "__main__":
    main()
