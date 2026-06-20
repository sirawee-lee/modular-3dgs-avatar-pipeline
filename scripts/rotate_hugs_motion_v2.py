#!/usr/bin/env python3
"""
Rotate a HUGS motion .npz by rotating ONLY the global_orient (root) and translation.
This keeps the body pose (relative to root) unchanged, fixing the case where arms are
correct but body is rotated 90 degrees relative to them.

Usage:
  python scripts/rotate_hugs_motion_v2.py \
    --input /path/to/hugs_motion.npz \
    --output /path/to/hugs_motion_rot.npz \
    --rx 90  # Rotate root by 90 degrees around X
"""

import argparse
from pathlib import Path
import numpy as np


def axis_angle_to_rotmat(aa, eps=1e-8):
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / np.clip(angle, eps, None)
    ax = axis[..., 0]
    ay = axis[..., 1]
    az = axis[..., 2]

    zero = np.zeros_like(ax)
    K = np.stack(
        [
            zero, -az, ay,
            az, zero, -ax,
            -ay, ax, zero,
        ],
        axis=-1,
    ).reshape(*ax.shape, 3, 3)

    I = np.eye(3, dtype=np.float32)
    I = np.broadcast_to(I, K.shape)

    sin = np.sin(angle)[..., 0][..., None, None]
    cos = np.cos(angle)[..., 0][..., None, None]
    R = I + sin * K + (1.0 - cos) * (K @ K)

    small = (angle[..., 0] < eps)
    if np.any(small):
        R[small] = I[small]
    return R


def rotmat_to_axis_angle(rotmat, eps=1e-8):
    trace = np.trace(rotmat, axis1=-2, axis2=-1)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    rx = rotmat[..., 2, 1] - rotmat[..., 1, 2]
    ry = rotmat[..., 0, 2] - rotmat[..., 2, 0]
    rz = rotmat[..., 1, 0] - rotmat[..., 0, 1]
    axis = np.stack([rx, ry, rz], axis=-1)

    sin_angle = np.sin(angle)
    small = np.abs(sin_angle) < eps
    axis = axis / np.expand_dims(2.0 * np.where(small, 1.0, sin_angle), axis=-1)
    axis = np.where(np.expand_dims(small, axis=-1), 0.0, axis)

    return axis * np.expand_dims(angle, axis=-1)


def euler_xyz_to_rotmat(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def main():
    parser = argparse.ArgumentParser(
        description="Rotate HUGS motion root-only (keeps body pose relative to root unchanged)"
    )
    parser.add_argument("--input", "-i", required=True, help="Input HUGS motion .npz")
    parser.add_argument("--output", "-o", required=True, help="Output rotated HUGS motion .npz")
    parser.add_argument("--rx", type=float, default=0.0, help="Rotate root X degrees")
    parser.add_argument("--ry", type=float, default=0.0, help="Rotate root Y degrees")
    parser.add_argument("--rz", type=float, default=0.0, help="Rotate root Z degrees")
    parser.add_argument("--center", action="store_true", help="Center translation to mean 0")
    parser.add_argument("--tx", type=float, default=0.0, help="Translate X offset")
    parser.add_argument("--ty", type=float, default=0.0, help="Translate Y offset")
    parser.add_argument("--tz", type=float, default=0.0, help="Translate Z offset")
    parser.add_argument("--ground", type=float, default=None,
                        help="Snap lowest Z frame to this value (e.g. 0.1) to fix floating avatar")
    args = parser.parse_args()

    data = np.load(args.input)
    global_orient = data["global_orient"].astype(np.float32)
    body_pose = data["body_pose"].astype(np.float32)
    transl = data["transl"].astype(np.float32)
    betas = data["betas"].astype(np.float32)

    # Build rotation matrix for root
    R_fix = euler_xyz_to_rotmat(
        np.deg2rad(args.rx), np.deg2rad(args.ry), np.deg2rad(args.rz)
    )

    # Rotate ONLY the global_orient (root orientation)
    global_orient_mat = axis_angle_to_rotmat(global_orient)  # Shape: (T, 3, 3)
    global_orient_mat_new = R_fix @ global_orient_mat  # Apply rotation on left
    global_orient_new = rotmat_to_axis_angle(global_orient_mat_new)

    # Keep body_pose unchanged (it's already relative to the root)
    body_pose_new = body_pose.copy()

    # Rotate translation
    transl_new = (R_fix @ transl.T).T
    if args.center:
        transl_new = transl_new - transl_new.mean(axis=0, keepdims=True)
    transl_new = transl_new + np.array([args.tx, args.ty, args.tz], dtype=np.float32)
    if args.ground is not None:
        # Shift so the minimum Z of the trajectory sits at --ground value
        transl_new[:, 2] = transl_new[:, 2] - transl_new[:, 2].min() + args.ground
        print(f"Ground snap: Z shifted to min={args.ground:.4f}")

    out = {
        "global_orient": global_orient_new.astype(np.float32),
        "body_pose": body_pose_new.astype(np.float32),
        "transl": transl_new.astype(np.float32),
        "betas": betas.astype(np.float32),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **out)
    print(f"Saved root-rotated motion to: {output_path}")


if __name__ == "__main__":
    main()
