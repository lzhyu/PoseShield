#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch

from poseshield.pose.resolve_slsqp import optimize_slsqp
from poseshield.pose.preprocess import pose_collision_status
from poseshield.pose.dataset import PosesFolderDataset
from poseshield.common.utils import quick_viz_6d, sixd_to_mesh
from poseshield.pose.utils import (
    cost_function,
    cost_function_weighted,
    constraint_function,
    save_sample_as_param_pickle
)
import smplx
from poseshield.common.config import get_cfg_defaults
from tqdm import tqdm
import time
import yaml
from poseshield.common.utils import load_model
from poseshield.common.collision import load_topology_distances

def main():
    global_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config_files", "global_path.yaml")
    with open(global_config_path, "r") as f:
        global_config = yaml.safe_load(f)
    dataset_test_default = global_config.get("DATASET_EVAL_PATH", global_config.get("DATASET_TEST_PATH", "dataset_test"))
    pose_model_default = global_config.get("POSE_CKPT_PATH", global_config.get("POSE_MODEL_PATH", "ckpts/poseshield/model.pth"))
    pose_config_default = global_config.get("POSE_CKPT_CONFIG", global_config.get("POSE_MODEL_CONFIG_PATH", "ckpts/poseshield/config_elu.yaml"))

    parser = argparse.ArgumentParser(
        description="PoseShield SLSQP collision resolution for test dataset"
    )
    parser.add_argument(
        "--model-path", default=pose_model_default,
        help="Path to the trained model weights",
    )

    parser.add_argument(
        "--data-dir", default=dataset_test_default,
        help="Directory containing augmented pose data",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of samples to search for a valid datum",
    )
    parser.add_argument(
        "--max-itr", type=int, default=300,
        help="Maximum number of optimization iterations",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-2,
        help="Learning rate for optimizer (not used in SLSQP but kept for compat)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Constraint threshold",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Whether to save the optimized poses and results",
    )
    parser.add_argument(
        "--config-path", default=pose_config_default,
        help="Path to the trained model specs",
    )
    parser.add_argument(
        "--cost-type", choices=["normal", "weighted"], default="normal",
        help="Type of cost function to use (normal or weighted)",
    )
    parser.add_argument(
        "--tol", type=float, default=0.1,
        help="Tolerance for 6D rotation validity constraints",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load cfg
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg.freeze()
    print(f'loading model from {args.model_path}')
    model = load_model(cfg, args.model_path, device, )

    # Load from the new folder structure like measure_correlation.py
    collision_dataset = PosesFolderDataset(directory_path=args.data_dir,)
    
    model_path = global_config["BODY_MODEL_PATH"]

    distance_path = global_config["MESH_DISTANCE_PATH"]
    distances = load_topology_distances(distance_path)
    smpl_model = smplx.create(
        model_path,
        model_type='smplh',
        gender='neutral',
        ext='npz',
        use_pca=False
    ).to(device)
    
    no_collision_samples = 0
    solver_success_samples = 0
    constraint_satisfied_samples = 0
    final_errors = []
    penetration_depth_reductions = []
    time_list = []
    mvd_list = []


    num_eval = min(args.n_samples, len(collision_dataset))
    
    for i in tqdm(range(num_eval)):
        sample = collision_dataset[i].reshape(-1)
        start_time = time.time()
        optimized_x, loss_hist, cons_hist, success, message = optimize_slsqp(
            sample, model, device, max_itr=args.max_itr, threshold=args.threshold, cost_type=args.cost_type, tol=args.tol
        )
        solver_success_samples += int(success)
        end_time = time.time()
        opt_time = end_time - start_time
        time_list.append(opt_time)
        print(f"Sample {i+1}/{num_eval} optimized in {opt_time:.2f} seconds")
        
        if args.cost_type == "weighted":
            final_error = cost_function_weighted(
                torch.from_numpy(optimized_x).to(device),
                torch.from_numpy(sample).float().to(device)
            ).item()
        else:
            final_error = cost_function(
                torch.from_numpy(optimized_x).to(device),
                torch.from_numpy(sample).float().to(device)
            ).item()
        try:
            final_constraint_val = constraint_function(
                model, torch.from_numpy(optimized_x).float().to(device)
            ).item()
        except Exception:
            final_constraint_val = float('nan')
        constraint_satisfied = bool(final_constraint_val >= args.threshold)
        constraint_satisfied_samples += int(constraint_satisfied)
            
        if args.save:
            os.makedirs("opt_samples", exist_ok=True)
            quick_viz_6d(sample.reshape(-1, 6), f"opt_samples/test_dataset_x_ori_{i}.png", f"opt_samples/test_dataset_x_ori_{i}.obj", smpl_model=smpl_model)
            quick_viz_6d(optimized_x.reshape(-1, 6), f"opt_samples/test_dataset_x_after_{i}.png", f"opt_samples/test_dataset_x_after_{i}.obj", smpl_model=smpl_model)
            save_sample_as_param_pickle(
                f"opt_samples/test_dataset_x_ori_{i}.pkl",
                sample,
                model_type='smplh',
            )
            save_sample_as_param_pickle(
                f"opt_samples/test_dataset_x_after_{i}.pkl",
                optimized_x,
                model_type='smplh',
            )
            
        # check for collision
        _, total_penetration_depth_b = pose_collision_status((sample), smpl_model, distances, device)
        has_problematic_collision_a, total_penetration_depth_a = pose_collision_status((optimized_x), smpl_model, distances, device)
        exact_collision_free = not has_problematic_collision_a
        
        if exact_collision_free:
            no_collision_samples+=1
        print(
            f"Sample {i+1}/{num_eval} status: "
            f"solver_success={success}, "
            f"constraint_satisfied={constraint_satisfied}, "
            f"exact_collision_free={exact_collision_free}, "
            f"final_cost={final_error:.6f}, "
            f"constraint_value={final_constraint_val:.6f}, "
            f"penetration_depth={total_penetration_depth_a:.6f}, "
            f"solver_message={message}"
        )
            
        final_errors.append(final_error)
        penetration_depth_reductions.append(max(0, 1 - total_penetration_depth_a/(total_penetration_depth_b+1e-6))) 
        
        # Calculate MVD
        vert_before, _ = sixd_to_mesh(smpl_model, sample.reshape(21, 6), device=device)
        vert_after, _ = sixd_to_mesh(smpl_model, optimized_x.reshape(21, 6), device=device)
        mvd = np.mean(np.linalg.norm(vert_before - vert_after, axis=-1))
        mvd_list.append(mvd)

    if num_eval > 0:
        print(f'solver success rate {solver_success_samples/num_eval}')
        print(f'constraint satisfied rate {constraint_satisfied_samples/num_eval}')
        print(f'exact collision-free rate {no_collision_samples/num_eval}')
        print(f'final error {np.mean(final_errors)}')
        print(f'penetration depth reduction {np.mean(penetration_depth_reductions)}')
        print(f'average optimization time {np.mean(time_list):.4f} seconds')
        print(f'average MVD {np.mean(mvd_list):.6f}')
    else:
        print('No samples evaluated.')

if __name__ == "__main__":
    main()
