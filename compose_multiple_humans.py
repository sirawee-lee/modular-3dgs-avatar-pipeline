#!/usr/bin/env python3
"""
Compose multiple HUGS human models into a single scene
"""
import torch
import numpy as np
import argparse
import sys
import os
from pathlib import Path
import torchvision
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, '.')

from hugs.models import HugsTriMLP
from hugs.models.scene import SceneGS
from hugs.renderer.gs_renderer import render_human_scene
from hugs.datasets.neuman import NeumanDataset
from hugs.utils.vis import create_video


def load_human_model(human_ckpt_path, cfg):
    """Load a trained human model from checkpoint"""
    human_gs = HugsTriMLP(
        sh_degree=cfg.human.sh_degree,
        smpl_init=None,
        n_subdivision=cfg.human.n_subdivision,
        only_rgb=cfg.human.only_rgb,
        use_surface=cfg.human.use_surface,
        use_deformer=cfg.human.use_deformer,
        init_2d=cfg.human.init_2d,
        init_scale_multiplier=cfg.human.init_scale_multiplier,
        disable_posedirs=cfg.human.disable_posedirs,
        isotropic=cfg.human.isotropic,
        res_offset=cfg.human.res_offset,
        estimate_delta=cfg.human.estimate_delta,
        triplane_res=cfg.human.triplane_res,
        knn_n_hops=cfg.human.knn_n_hops,
        activation=cfg.human.activation,
    )
    
    # Load checkpoint
    state_dict = torch.load(human_ckpt_path, map_location='cuda')
    human_gs.load_state_dict(state_dict)
    human_gs.eval()
    
    logger.info(f"✓ Loaded human model from {human_ckpt_path}")
    return human_gs


def load_scene_model(scene_ckpt_path, cfg):
    """Load a trained scene model from checkpoint"""
    scene_gs = SceneGS(sh_degree=cfg.scene.sh_degree)
    
    # Load checkpoint
    state_dict = torch.load(scene_ckpt_path, map_location='cuda')
    scene_gs.load_state_dict(state_dict)
    scene_gs.eval()
    
    logger.info(f"✓ Loaded scene model from {scene_ckpt_path}")
    return scene_gs


def compose_humans(human_models, human_datasets, scene_model, output_dir, 
                   positions=None, scales=None, num_frames=100, bg_color='white'):
    """
    Compose multiple human models into a single scene
    
    Args:
        human_models: List of HugsTriMLP models
        human_datasets: List of corresponding datasets for SMPL params
        scene_model: SceneGS background model
        output_dir: Directory to save output frames
        positions: List of (x, y, z) positions for each human (relative offset)
        scales: List of scale factors for each human
        num_frames: Number of frames to render
        bg_color: Background color
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Default positions and scales
    if positions is None:
        # Arrange humans in a line
        spacing = 2.0
        positions = [(i * spacing - (len(human_models) - 1) * spacing / 2, 0, 0) 
                     for i in range(len(human_models))]
    
    if scales is None:
        scales = [1.0] * len(human_models)
    
    # Convert bg_color to tensor
    if bg_color == 'white':
        bg_color_tensor = torch.tensor([1.0, 1.0, 1.0], device='cuda')
    elif bg_color == 'black':
        bg_color_tensor = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    else:
        bg_color_tensor = torch.tensor([1.0, 1.0, 1.0], device='cuda')
    
    logger.info(f"Rendering {num_frames} frames with {len(human_models)} humans...")
    
    # Render each frame
    for frame_idx in range(num_frames):
        # Get camera from first dataset
        data = human_datasets[0][frame_idx % len(human_datasets[0])]
        
        # Collect outputs from all humans
        all_human_xyz = []
        all_human_shs = []
        all_human_opacity = []
        all_human_scales = []
        all_human_rotq = []
        
        for human_idx, (human_model, dataset) in enumerate(zip(human_models, human_datasets)):
            # Get SMPL params for this frame
            data_human = dataset[frame_idx % len(dataset)]
            
            # Forward pass
            with torch.no_grad():
                human_gs_out = human_model.forward(
                    global_orient=data_human['global_orient'].unsqueeze(0).cuda(),
                    body_pose=data_human['body_pose'].unsqueeze(0).cuda(),
                    betas=data_human['betas'].unsqueeze(0).cuda(),
                    transl=data_human['transl'].unsqueeze(0).cuda(),
                )
            
            # Apply position offset
            pos_offset = torch.tensor(positions[human_idx], device='cuda', dtype=torch.float32)
            xyz = human_gs_out['xyz'] + pos_offset[None, :]
            
            # Apply scale
            scale_factor = scales[human_idx]
            human_scales_scaled = human_gs_out['scales'] * scale_factor
            
            all_human_xyz.append(xyz)
            all_human_shs.append(human_gs_out['shs'])
            all_human_opacity.append(human_gs_out['opacity'])
            all_human_scales.append(human_scales_scaled)
            all_human_rotq.append(human_gs_out['rotq'])
        
        # Combine all humans
        combined_human_gs_out = {
            'xyz': torch.cat(all_human_xyz, dim=0),
            'shs': torch.cat(all_human_shs, dim=0),
            'opacity': torch.cat(all_human_opacity, dim=0),
            'scales': torch.cat(all_human_scales, dim=0),
            'rotq': torch.cat(all_human_rotq, dim=0),
            'active_sh_degree': human_models[0].active_sh_degree,
        }
        
        # Get scene output
        with torch.no_grad():
            scene_gs_out = scene_model.forward()
        
        # Render the combined scene
        render_pkg = render_human_scene(
            data=data,
            human_gs_out=combined_human_gs_out,
            scene_gs_out=scene_gs_out,
            bg_color=bg_color_tensor,
            render_mode='human_scene',
        )
        
        image = render_pkg['render']
        torchvision.utils.save_image(image, f'{output_dir}/{frame_idx:05d}.png')
        
        if frame_idx % 10 == 0:
            logger.info(f"Rendered frame {frame_idx}/{num_frames}")
    
    # Create video
    video_path = f'{output_dir}/composed_humans.mp4'
    create_video(output_dir, video_path, fps=30)
    logger.info(f"✓ Video saved to {video_path}")
    
    return video_path


def main():
    parser = argparse.ArgumentParser(description='Compose multiple HUGS human models')
    parser.add_argument('--scene_ckpt', required=True, help='Path to scene checkpoint')
    parser.add_argument('--human_ckpts', nargs='+', required=True, 
                       help='Paths to human checkpoints (e.g., bike citron jogging)')
    parser.add_argument('--human_seqs', nargs='+', required=True,
                       help='Sequence names corresponding to checkpoints')
    parser.add_argument('--cfg_file', default='cfg_files/release/neuman/hugs_human_scene.yaml',
                       help='Config file')
    parser.add_argument('--output_dir', default='output/composed_scene',
                       help='Output directory')
    parser.add_argument('--num_frames', type=int, default=100,
                       help='Number of frames to render')
    parser.add_argument('--positions', nargs='+', type=float, default=None,
                       help='Positions as x1 y1 z1 x2 y2 z2 ...')
    parser.add_argument('--scales', nargs='+', type=float, default=None,
                       help='Scale factors for each human')
    parser.add_argument('--bg_color', default='white', choices=['white', 'black'],
                       help='Background color')
    
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.cfg_file)
    
    # Parse positions
    positions = None
    if args.positions:
        assert len(args.positions) % 3 == 0, "Positions must be in groups of 3 (x, y, z)"
        positions = [(args.positions[i], args.positions[i+1], args.positions[i+2])
                    for i in range(0, len(args.positions), 3)]
        assert len(positions) == len(args.human_ckpts), \
            f"Number of positions ({len(positions)}) must match number of humans ({len(args.human_ckpts)})"
    
    # Load scene model
    logger.info("Loading scene model...")
    scene_model = load_scene_model(args.scene_ckpt, cfg)
    
    # Load human models and datasets
    logger.info(f"Loading {len(args.human_ckpts)} human models...")
    human_models = []
    human_datasets = []
    
    for ckpt_path, seq_name in zip(args.human_ckpts, args.human_seqs):
        # Load model
        human_model = load_human_model(ckpt_path, cfg)
        human_models.append(human_model)
        
        # Load corresponding dataset for SMPL params
        dataset = NeumanDataset(
            seq=seq_name,
            split='anim',
            render_mode='human',
        )
        human_datasets.append(dataset)
        logger.info(f"  {seq_name}: {len(dataset)} frames")
    
    # Compose and render
    logger.info("Composing scene...")
    compose_humans(
        human_models=human_models,
        human_datasets=human_datasets,
        scene_model=scene_model,
        output_dir=args.output_dir,
        positions=positions,
        scales=args.scales,
        num_frames=args.num_frames,
        bg_color=args.bg_color,
    )
    
    logger.info("✓ Done!")


if __name__ == '__main__':
    main()
