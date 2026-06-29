import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from pathlib import Path
import csv
import pickle
from collections import defaultdict
from poseshield.common.utils import *

def load_from_path(data_path, directory_path, split):
    if split == 'all':
        file_paths = [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith('.npz')
        ]
    else:
        if split == 'train':
            split_csv = os.path.join(directory_path, 'train_list.csv')
        elif split == 'test':
            split_csv = os.path.join(directory_path, 'test_list.csv')
        else:
            raise ValueError(f"Unsupported split: '{split}'. Expected 'train', 'test', or 'all'.")

        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split file not found: {split_csv}")

        with open(split_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            selected_files = {row['filename'] for row in reader}

        # Create full paths for .npz files that match the split
        file_paths = [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith('.npz') and f in selected_files
        ]
        print(len(file_paths), f'files in {data_path}, split:', split)
    return file_paths

class PosesDataset(Dataset):
    def __init__(self, directory_path, split):
        """
        Args:
            directory_path (str): Path to the directory containing .npy files.
        """
        self.all_poses = []  # Will hold all frames from all files
        DOWNSAMPLE = 4
        
        self.all_labels = []
        self.all_names = []
        error_count = 0

        # NOTE: downsample GT
        data_path_gt = os.path.join(directory_path, 'gt_data')
        file_paths_gt = load_from_path(data_path_gt, directory_path, split)
        random.shuffle(file_paths_gt)
        for file_path in file_paths_gt:
            try:
                data = np.load(file_path, allow_pickle=True)
                poses = data['poses'] # list of (21, 6)
                labels = data['labels']
            except Exception as e:
                error_count += 1
                continue
            self.all_labels.extend(labels[::DOWNSAMPLE])
            self.all_poses.extend(poses[::DOWNSAMPLE])
            self.all_names.extend(Path(file_path).stem for _ in range(len(poses[::DOWNSAMPLE])))

        # NOTE: do not downsample augmented data 
        data_path_aug = os.path.join(directory_path, 'augmented_data')
        file_paths_aug = load_from_path(data_path_aug, directory_path, split)
        # add augmented data
        random.shuffle(file_paths_aug)
        # Loop through each file and load/cache all its frames
        for file_path in file_paths_aug:
            try:
                data = np.load(file_path, allow_pickle=True)
                poses = data['poses'] # list of (21, 6)
                labels = data['labels']
            except Exception as e:
                error_count += 1
                continue
            self.all_labels.extend(labels)
            self.all_poses.extend(poses)
            self.all_names.extend(Path(file_path).stem for _ in range(len(poses)))

        if error_count > 0:
            print(f"Skipped {error_count} corrupted or unreadable .npz files.")

    def __len__(self):
        # Total number of frames across all files
        return len(self.all_poses)

    def __getitem__(self, idx):
        # Return the frame at index idx as a (21,6) Tensor
        return self.all_poses[idx], self.all_labels[idx], self.all_names[idx]

    def add_data(self, poses, labels, names=None):
        self.all_poses.extend(poses)
        self.all_labels.extend(labels)
        self.all_names.extend(names or ['active']*len(poses))
    
    def sample_from_dataset(self, num_samples):
        """
        Randomly sample a subset of the dataset.
        """
        if num_samples > len(self.all_poses):
            num_samples = len(self.all_poses)
        indices = np.random.choice(len(self.all_poses), size=num_samples, replace=False)
        sampled_poses = [self.all_poses[i] for i in indices]
        sampled_labels = [self.all_labels[i] for i in indices]
        return sampled_poses, sampled_labels

class PosesFolderDataset(Dataset):
    def __init__(self, directory_path):
        """
        Args:
            directory_path (str): Path to the directory containing .npy files.
        """
        self.all_poses = []  # Will hold all frames from all files
    
        # NOTE: downsample GT
        #         
        def list_pkl_files(dir_path):
            import re
            files = []
            if not os.path.isdir(dir_path):
                return files
            for root, _, fnames in os.walk(dir_path):
                for fname in fnames:
                    if fname.lower().endswith(".pkl"):
                        files.append(os.path.join(root, fname))
            files.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])
            return files

        pkl_files = list_pkl_files(directory_path)
        
        for fn in pkl_files:
            params_dict = defaultdict(lambda: [])
            with open(fn, 'rb') as param_file:
                data = pickle.load(param_file, encoding='latin1')

            assert 'betas' in data, 'Missing key: betas'
            assert ('global_pose' in data) or ('global_orient' in data), 'Missing key: global_pose/orient'
            assert ('pose' in data) or ('body_pose' in data), 'Missing key: pose/body_pose'

            for k, v in data.items():
                params_dict[k].append(v)
            rotation_matrix = axis_angle_to_matrix(params_dict['pose'][0].reshape(21, 3))
            rotation_6d = matrix_to_rotation_6d(rotation_matrix)
            # TO6D
            self.all_poses.append(rotation_6d)  # (N, 21, 6)

    def __len__(self):
        # Total number of frames across all files
        return len(self.all_poses)

    def __getitem__(self, idx):
        # Return the frame at index idx as a (21,6) Tensor
        return self.all_poses[idx]
if __name__ == "__main__":
    dataset = PosesFolderDataset("dataset")
