#!/usr/bin/env python3
"""
Export HUGS model to PLY format for viewing in supersplat.at
"""
import torch
import numpy as np
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, '.')


def construct_list_of_attributes():
    """Construct PLY property names for Gaussian Splatting"""
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # RGB features
    for i in range(3):
        l.append(f'f_dc_{i}')
    l.append('opacity')
    # Scale
    for i in range(3):
        l.append(f'scale_{i}')
    # Rotation (quaternion)
    for i in range(4):
        l.append(f'rot_{i}')
    return l


def save_ply(path, xyz, features_dc, opacities, scales, rotations, normals=None):
    """Save Gaussian parameters to PLY file"""
    
    if normals is None:
        normals = np.zeros_like(xyz)
    
    # Construct dtype
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    
    n_points = xyz.shape[0]
    elements = np.empty(n_points, dtype=dtype_full)
    
    # Combine all attributes
    attributes = np.concatenate([
        xyz,
        normals,
        features_dc,
        opacities.reshape(n_points, 1),
        scales,
        rotations
    ], axis=1)
    
    elements[:] = list(map(tuple, attributes))
    
    # Write PLY file
    with open(path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {n_points}\n".encode())
        
        for attr in construct_list_of_attributes():
            f.write(f"property float {attr}\n".encode())
        
        f.write(b"end_header\n")
        f.write(elements.tobytes())
    
    print(f"✓ Saved PLY file: {path}")


def export_hugs_to_ply(checkpoint_path, output_path=None, frame_id=0):
    """Export HUGS model to PLY format"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract xyz positions
    xyz = checkpoint['xyz'].detach().cpu().numpy()
    print(f"Number of Gaussians: {xyz.shape[0]}")
    
    # Try to extract other parameters
    # For HUGS, we might need to decode through the network
    # For now, let's try to extract what we can
    
    # Initialize dummy values for features we don't have directly
    n_gaussians = xyz.shape[0]
    
    # Create default values
    features_dc = np.zeros((n_gaussians, 3), dtype=np.float32)  # RGB
    opacities = np.ones((n_gaussians,), dtype=np.float32) * 0.5  # Default opacity
    scales = np.ones((n_gaussians, 3), dtype=np.float32) * 0.01  # Small scale
    rotations = np.zeros((n_gaussians, 4), dtype=np.float32)  # Identity quaternion
    rotations[:, 0] = 1.0  # w component
    
    print(f"XYZ shape: {xyz.shape}")
    print(f"Features DC shape: {features_dc.shape}")
    print(f"Opacities shape: {opacities.shape}")
    print(f"Scales shape: {scales.shape}")
    print(f"Rotations shape: {rotations.shape}")
    
    # Determine output path
    if output_path is None:
        output_path = Path(checkpoint_path).parent / f"{Path(checkpoint_path).stem}_frame{frame_id}.ply"
    
    # Save to PLY
    save_ply(output_path, xyz, features_dc, opacities, scales, rotations)
    
    print(f"\n✓ Export complete!")
    print(f"You can now upload {output_path} to https://supersplat.at/editor")
    print(f"\nNote: This is a simplified export with basic Gaussian parameters.")
    print(f"For full rendering, use the HUGS renderer with the checkpoint.")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export HUGS checkpoint to PLY')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output PLY path')
    parser.add_argument('-f', '--frame', type=int, default=0, help='Frame ID')
    
    args = parser.parse_args()
    
    export_hugs_to_ply(args.checkpoint, args.output, args.frame)

##วิธีใช้
'''
conda run -n hugs python export_hugs_to_ply.py "pathของ.pthที่จะแปลง และเดี๋ยวมันจะวางoutput(.ply)ในโฟลเดอร์เดียวกัน"
'''