import os
import numpy as np
import torch
import smplx
import pickle
import yaml
from tqdm import tqdm
from pathlib import Path

from poseshield.common.utils import sixd_to_mesh, make_6d_rotation_valid
from poseshield.common.collision import is_mesh_self_intersecting

def augment_pose(pose, purturb_strength=0.2):
    # random purturbation
    return make_6d_rotation_valid(np.random.randn(21, 6)*purturb_strength + pose.reshape(21, 6))
    
if __name__=="__main__":
    global_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config_files", "global_path.yaml")
    with open(global_config_path, "r") as f:
        global_config = yaml.safe_load(f)

    model_path = global_config["BODY_MODEL_PATH"]
    distance_path = global_config["MESH_DISTANCE_PATH"]
    directory_path = str(Path(global_config["DATA_PATH"]) / "gt_data")
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model  = smplx.create(
        model_path,
        model_type='smplh',
        gender='neutral',
        ext='npz',
        use_pca=False
    ).to(device)
    distances = pickle.load(open(distance_path, "rb"))

    file_paths = [
        os.path.join(directory_path, f) 
        for f in os.listdir(directory_path) 
        if f.endswith('.npz')
    ]
    target_path = Path(directory_path).parent / "augmented_data"
    os.makedirs(target_path, exist_ok=True)

    for file_path in tqdm(file_paths):
        target_file = str(target_path / Path(file_path).name)
        if os.path.exists(target_file):
            print(f"File {target_file} already exists. Skipping...")
            continue
        poses = np.load(file_path, allow_pickle=True)['poses'] # list of (21, 6)
        labels = np.load(file_path, allow_pickle=True)['labels']
        # augment all poses
        for i in range(len(poses)):
            poses[i] = augment_pose(poses[i])
            vertices, faces = sixd_to_mesh(smpl_model, poses[i], device=device)
            collision = is_mesh_self_intersecting(vertices, faces, distances)
            labels[i] = (-1 if collision else 1)

        np.savez(target_file, poses=poses, labels=labels)
