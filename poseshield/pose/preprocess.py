import os
import numpy as np
import torch
import yaml
import pickle
import smplx
from tqdm import tqdm
from pathlib import Path

from poseshield.common.utils import sixd_to_mesh
from poseshield.common.collision import is_mesh_self_intersecting, self_collision_status

def label_pose(x, smpl_model, distances, device):
    """
    Placeholder for your “oracle”:
    given new pose x (flattened 21×6), return its ground-truth label.
    """
    pose_21x6 = x.reshape(21, 6)
    vertices, faces = sixd_to_mesh(smpl_model, pose_21x6, device=device)
    collision = is_mesh_self_intersecting(vertices, faces, distances)
    return -1 if collision else 1
    
def pose_collision_status(x, smpl_model, distances, device):
    pose_21x6 = x.reshape(21, 6)
    vertices, faces = sixd_to_mesh(smpl_model, pose_21x6, device=device)
    has_problematic_collision, total_penetration_depth = self_collision_status(vertices, faces, distances)
    return has_problematic_collision, total_penetration_depth

def process_gt_data(directory_path):
    """
    Process all .npy files in the given directory to extract poses and labels.
    """
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist. Skipping preprocessing.")
        return

    file_paths = [
        os.path.join(directory_path, f) 
        for f in os.listdir(directory_path) 
        if f.endswith('.npy')
    ]
    
    data_path = "dataset"

    global_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config_files", "global_path.yaml")
    with open(global_config_path, "r") as f:
        global_config = yaml.safe_load(f)
        model_path = global_config["BODY_MODEL_PATH"]
        distance_path = global_config["MESH_DISTANCE_PATH"]
    distances = pickle.load(open(distance_path, "rb"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = smplx.create(
        model_path,
        model_type='smplh',
        gender='neutral',
        ext='npz',
        use_pca=False
    ).to(device)

    # Loop through each file and load/cache all its frames
    for file_path in tqdm(file_paths):
        # Each file is a motion sequence: shape [num_frames, ...]
        try:
            dic_source = np.load(file_path, allow_pickle=True).item()['pose_source']
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

        file_poses = []
        file_labels = []
        stem = Path(file_path).stem
        
        target_file = os.path.join(data_path, f"{stem}.npz")
        if os.path.exists(target_file):
            print(f"{target_file} already exists, skipping...")
            continue

        # Extract and reshape each frame from this file
        for frame_data in dic_source:
            pose_21x6 = frame_data[9:].reshape(21, 6)
            file_poses.append(pose_21x6)

            vertices, faces = sixd_to_mesh(smpl_model, pose_21x6, device=device) 
            collision = is_mesh_self_intersecting(vertices, faces, distances)
            # if collision label=-1 else 1
            file_labels.append(-1 if collision else 1)
        np.savez(target_file, poses=file_poses, labels=file_labels)
        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess GT data")
    parser.add_argument("--data-dir", default=None, help="Path to raw motion directory (containing .npy sequences)")
    args = parser.parse_args()
    
    directory_path = args.data_dir
    if directory_path is None:
        global_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config_files", "global_path.yaml")
        if os.path.exists(global_config_path):
            with open(global_config_path, "r") as f:
                global_config = yaml.safe_load(f)
            directory_path = global_config.get("DATA_PATH", "dataset")
        else:
            directory_path = "dataset"
            
    print(f"Preprocessing data from {directory_path}...")
    process_gt_data(directory_path)