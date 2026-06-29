#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import time
import yaml
import numpy as np
import torch

from poseshield.common.network import ResidualMLP
from poseshield.common.utils import (
    quick_viz_6d, 
    sixd_to_mesh, 
    axis_angle_to_matrix, 
    matrix_to_rotation_6d, 
    rotation_6d_to_matrix, 
    matrix_to_axis_angle
)
from poseshield.pose.resolve_slsqp import optimize_slsqp
from poseshield.pose.utils import cost_function, constraint_function
from poseshield.common.config import get_cfg_defaults
from poseshield.common.utils import load_model

def get_args():
    parser = argparse.ArgumentParser(description="PoseShield Pose Optimization Demo")
    parser.add_argument(
        "--model-path", default="ckpts/poseshield/model.pth",
        help="Path to the trained model weights"
    )
    parser.add_argument(
        "--config-path", default="ckpts/poseshield/config.yaml",
        help="Path to the trained model specs"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Constraint threshold (lower means harder constraint, e.g. 0.1)"
    )
    parser.add_argument(
        "--max-itr", type=int, default=200,
        help="Maximum number of optimization iterations"
    )
    parser.add_argument(
        "--output-dir", default="demos/output",
        help="Directory to save optimization results"
    )
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Configurations
    print(f"Loading config from {args.config_path}...")
    cfg = get_cfg_defaults()
    if os.path.exists(args.config_path):
        cfg.merge_from_file(args.config_path)
    else:
        print(f"Warning: config file not found at {args.config_path}. Using defaults.")
    cfg.freeze()

    # 2. Check and Load Model
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at: {args.model_path}")
        print("Please download the pre-trained model and place it at ckpts/poseshield/model.pth")
        sys.exit(1)
        
    print(f"Loading pre-trained collision field model from {args.model_path}...")
    model = load_model(cfg, args.model_path, device)
    model.eval()

    # 3. Define Demo Samples
    demo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_asset")
    samples = ["x_ori_210.pkl", "x_ori_408.pkl", "x_ori_436.pkl"]
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Try loading SMPL-H and distance tools for verification
    # We will attempt to run collision verification if SMPL-H body model is available.
    global_config_path = os.path.join("config_files", "global_path.yaml")
    body_model_available = False
    smpl_model = None
    distances = None
    
    if os.path.exists(global_config_path):
        try:
            with open(global_config_path, "r") as f:
                global_config = yaml.safe_load(f)
            model_path = global_config.get("BODY_MODEL_PATH", "./body_models")
            distance_path = global_config.get("MESH_DISTANCE_PATH", "./dataset/distances.pkl")
            
            if os.path.exists(model_path):
                import smplx
                smpl_model = smplx.create(
                    model_path,
                    model_type='smplh',
                    gender='neutral',
                    ext='npz',
                    use_pca=False
                ).to(device)
                body_model_available = True
                print(f"Loaded SMPL-H model from {model_path} for verification.")
                
                if os.path.exists(distance_path):
                    distances = pickle.load(open(distance_path, "rb"))
                    print(f"Loaded mesh distance file from {distance_path}.")
                else:
                    print(f"Mesh distance file not found at {distance_path}. Skipping collision status checks.")
            else:
                print(f"SMPL-H body model folder not found at {model_path}. Skipping verification.")
        except Exception as e:
            print(f"Failed to initialize verification: {e}")

    # 5. Run Optimization
    for s_name in samples:
        s_path = os.path.join(demo_dir, s_name)
        if not os.path.exists(s_path):
            print(f"Demo sample {s_name} not found at {s_path}. Skipping.")
            continue
            
        print("\n" + "="*50)
        print(f"Optimizing Demo Sample: {s_name}")
        print("="*50)

        # Load parameters
        with open(s_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        # Extract pose parameters and convert to 6D rotation format
        pose_aa = data['pose'].reshape(21, 3)
        rotation_matrix = axis_angle_to_matrix(pose_aa)
        rotation_6d = matrix_to_rotation_6d(rotation_matrix)  # shape: (21, 6)
        
        sample_flat = rotation_6d.reshape(-1)

        # Print initial constraint value
        with torch.no_grad():
            init_val = constraint_function(
                model, torch.from_numpy(sample_flat).float().to(device)
            ).item()
        print(f"Initial constraint value (Collision field energy): {init_val:.6f}")

        # Run SLSQP Optimization
        start_time = time.time()
        optimized_x, loss_hist, cons_hist, success, message = optimize_slsqp(
            sample_flat, model, device, max_itr=args.max_itr, threshold=args.threshold
        )
        end_time = time.time()
        
        final_val = constraint_function(
            model, torch.from_numpy(optimized_x).float().to(device)
        ).item()
        final_error = cost_function(
            torch.from_numpy(optimized_x).to(device),
            torch.from_numpy(sample_flat).float().to(device)
        ).item()

        print(f"\nOptimization Finished in {end_time - start_time:.2f}s (Success: {success})")
        print(f"  - Message: {message}")
        print(f"  - Final constraint value: {final_val:.6f} (Threshold: {args.threshold})")
        print(f"  - Mean Vertex Deviation (MVD error): {final_error:.6f}")

        # 6. Save results & Visualizations
        stem = s_name.replace(".pkl", "")
        
        # Save optimized parameters to pickle
        opt_data = data.copy()
        # Convert optimized 6D back to axis-angles
        opt_rot_mats = rotation_6d_to_matrix(optimized_x.reshape(21, 6))
        opt_pose_aa = matrix_to_axis_angle(opt_rot_mats)
        opt_data['pose'] = opt_pose_aa.reshape(-1)
        
        # ==============================================================================
        # EXPORTED POSE PARAMETERS FORMAT DESCRIPTION (*_optimized.pkl):
        # A dictionary saved via pickle with the following key structure:
        # 
        # - 'pose': Flattened body joints rotation in axis-angles. Shape: [63] (21 joints * 3).
        #   Representing the 21 body joints of the SMPL-H model.
        # 
        # - 'betas': SMPL-H shape parameters. Shape: [10] (default: zeros).
        # 
        # - 'global_pose': Global root orientation. Shape: [3] (default: zeros).
        # ==============================================================================
        opt_pkl_path = os.path.join(args.output_dir, f"{stem}_optimized.pkl")
        with open(opt_pkl_path, 'wb') as f:
            pickle.dump(opt_data, f)
        print(f"Saved optimized parameters to {opt_pkl_path}")
        
        # Save visualizations
        try:
            # We visualize both the original and optimized poses
            ori_png = os.path.join(args.output_dir, f"{stem}_ori.png")
            ori_obj = os.path.join(args.output_dir, f"{stem}_ori.obj")
            opt_png = os.path.join(args.output_dir, f"{stem}_after.png")
            opt_obj = os.path.join(args.output_dir, f"{stem}_after.obj")
            
            print(f"Rendering original pose mesh to {ori_png} and {ori_obj}...")
            quick_viz_6d(rotation_6d, save_path=ori_png, mesh_path=ori_obj)
            
            print(f"Rendering optimized pose mesh to {opt_png} and {opt_obj}...")
            quick_viz_6d(optimized_x.reshape(21, 6), save_path=opt_png, mesh_path=opt_obj)
            
            # If SMPL-H and distance file are loaded, check actual collision status
            if body_model_available and distances is not None:
                from poseshield.pose.preprocess import pose_collision_status
                _, pen_depth_b = pose_collision_status(rotation_6d, smpl_model, distances, device)
                _, pen_depth_a = pose_collision_status(optimized_x, smpl_model, distances, device)


                print(f"Collision metrics:")
                print(f"  - Initial: Penetration Depth = {pen_depth_b:.6f}")
                print(f"  - Optimized: Penetration Depth = {pen_depth_a:.6f}")
                
        except Exception as e:
            print(f"Failed to generate visualization or run metrics verification: {e}")
            print("Note: SMPL-H models/packages must be correctly configured to run mesh visualizations.")

if __name__ == "__main__":
    main()
