import torch
import torch.nn.functional as F
import numpy as np
from poseshield.common.network import ResidualMLP
import yaml

# Use absolute paths from DNO project
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
global_config_path = os.path.join(project_root, "config_files", "global_path.yaml")
with open(global_config_path, 'r') as f:
    paths = yaml.safe_load(f)


def _resolve_project_path(path):
    """Resolve config paths relative to the PoseShield project root."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


BODY_MODEL_PATH_ = _resolve_project_path(paths['BODY_MODEL_PATH'])
MEAN_PATH_ = _resolve_project_path(paths['MEAN_PATH'])
STD_PATH_ = _resolve_project_path(paths['STD_PATH'])
POSESHIELD_CONFIG_PATH = _resolve_project_path(paths.get('MOTION_CKPT_CONFIG'))
POSESHIELD_WEIGHTS_PATH = _resolve_project_path(paths.get('MOTION_CKPT_PATH'))

# ── SMPLH kinematic-chain weights ─────────────────────────────────────────────
# Weight for each of the 21 body joints (indices 1-21, root excluded) =
# size of its downstream subtree in the SMPLH kinematic tree.
# The bigger the subtree, the more joints are affected by that rotation,
# so we penalise deviations there more in the cost function.
#
# SMPLH body kinematic tree (joints 1-21, parent → children):
#   spine1(3) → spine2(6) → spine3(9) → neck(12) → head(15)
#                                       → left_collar(13)  → left_shoulder(16)  → left_elbow(18)  → left_wrist(20)
#                                       → right_collar(14) → right_shoulder(17) → right_elbow(19) → right_wrist(21)
#   left_hip(1)  → left_knee(4)  → left_ankle(7)  → left_foot(10)
#   right_hip(2) → right_knee(5) → right_ankle(8) → right_foot(11)
#
# Subtree sizes  (joint 1 … 21):
_SUBTREE_SIZES = [
#  j1   j2   j3   j4  j5   j6  j7  j8   j9  j10 j11  j12  j13  j14 j15  j16  j17 j18 j19 j20 j21
    4,   4,  13,   3,  3,  12,  2,  2,  11,   1,  1,   2,   4,   4,  1,   3,   3,  2,  2,  1,  1
]
# Expand to 126 dims (×6 per joint), normalize to sum = 1
import numpy as _np
_w21 = _np.array(_SUBTREE_SIZES, dtype=_np.float32)
_w21 /= _w21.sum()

# Per-joint weight vector, shape (21,).  One entry per local rotation.
SMPLH_POSE_WEIGHTS: torch.Tensor = torch.from_numpy(_w21)   # (21,)


class MotionTemporalRegularizer:
    """Measure direct 6D-rotation position and velocity drift from a reference."""

    def __init__(self, motion_file, device, mean=None, std=None):
        if isinstance(motion_file, np.ndarray):
            motion_target = torch.from_numpy(motion_file)
        elif torch.is_tensor(motion_file):
            motion_target = motion_file
        else:
            motion_target = torch.from_numpy(np.load(motion_file))
        if motion_target.ndim != 2 or motion_target.shape[-1] != 135:
            raise ValueError(f"Expected target motion [frames, 135], got {motion_target.shape}")
        self.motion_target = motion_target.to(device).float()
        self.mean = mean.to(device) if mean is not None else None
        self.std = std.to(device) if std is not None else None

    def __call__(self, x, mean=None, std=None):
        """Return full-body and upper-body temporal drift components."""
        cur_mean = mean if mean is not None else self.mean
        cur_std = std if std is not None else self.std
        if cur_mean is None or cur_std is None:
            raise ValueError("Motion normalization mean and std are required")
        cur_std = torch.where(cur_std < 1e-3, torch.zeros_like(cur_std), cur_std)
        x_unnorm = x * cur_std + cur_mean

        generated_rotations = torch.cat(
            [x_unnorm[..., 3:9], x_unnorm[..., 9:135]], dim=-1
        ).reshape(x.shape[0], x.shape[1], 22, 6)
        target_rotations = self.motion_target[:, :132].reshape(1, -1, 22, 6)
        target_rotations = target_rotations.expand(x.shape[0], -1, -1, -1)
        min_len = min(generated_rotations.shape[1], target_rotations.shape[1])
        generated_rotations = generated_rotations[:, :min_len]
        target_rotations = target_rotations[:, :min_len]

        generated_velocity = torch.diff(generated_rotations, dim=1)
        target_velocity = torch.diff(target_rotations, dim=1)
        return {
            "rotation_velocity": F.l1_loss(generated_velocity, target_velocity),
            "upper_body_velocity": F.l1_loss(
                generated_velocity[:, :, 15:22],
                target_velocity[:, :, 15:22],
            ),
            "upper_body_rotation": F.l1_loss(
                generated_rotations[:, :, 15:22],
                target_rotations[:, :, 15:22],
            ),
        }

class MotionSimilarityLoss:
    """
    Computes similarity loss between generated motion and a target motion.
    
    Format contract:
        motion_file (target): public 135-dim canonical format.
            [0:6]     = root rotation (column-interleaved 6D: [c1x,c2x,c1y,c2y,c1z,c2z])
            [6:132]   = 21 body joint rotations (column-interleaved 6D each)
            [132:135] = absolute global translation [abs_x, abs_y, abs_z]
        x (generated):  201-dim normalized model latent.
            After denorm: [0:3]=abs trans, [3:9]=root rot, [9:135]=body rot (all col-interleaved)
        
    Compares: body rots [9:135] vs [6:132], root rot [3:9] vs [0:6].
    Translation losses, when enabled, compare absolute global XYZ directly.
    """
    def __init__(self, motion_file, device, mean=None, std=None,
                 joint_position_coef=0.0, hand_coef=0.0, joint_velocity_coef=0.0,
                 hand_joint_velocity_coef=0.0, wrist_position_coef=0.0,
                 lower_body_coef=0.0, translation_coef=0.0,
                 translation_smooth_coef=0.0, motion_length=None, use_weighted_loss=False,
                 translation_smooth_mode="first_diff", translation_smooth_loss_type="mse",
                 translation_mode="abs_3d", use_final_output_geometry=False,
                 rotation_coef=1.0, rot_velocity_coef=0.0,
                 upper_body_rot_weight=1.0, head_rotation_coef=0.0,
                 head_rot_velocity_coef=0.0):
        self.device = device
        # Load motion file (expecting [Frames, D] numpy array)
        self.use_weighted_loss = use_weighted_loss
        if use_weighted_loss:
            self.joint_weights = SMPLH_POSE_WEIGHTS.to(device)  # (21,)
            print("Using weighted rotation loss (subtree-size weights).")
        if isinstance(motion_file, str):
            motion_target = np.load(motion_file)
            self.motion_target = torch.from_numpy(motion_target).to(device).float()
        elif isinstance(motion_file, np.ndarray):
            self.motion_target = torch.from_numpy(motion_file).to(device).float()
        elif torch.is_tensor(motion_file):
            self.motion_target = motion_file.to(device).float()
        else:
            raise ValueError(f"Unsupported motion_file type: {type(motion_file)}")

        if mean is not None and std is not None:
             self.mean = mean.to(device)
             self.std = std.to(device)
        else:
             self.mean = None
             self.std = None
             
        self.joint_position_coef = joint_position_coef
        self.hand_coef = hand_coef
        self.joint_velocity_coef = joint_velocity_coef
        self.hand_joint_velocity_coef = hand_joint_velocity_coef
        self.wrist_position_coef = wrist_position_coef
        self.lower_body_coef = lower_body_coef
        self.translation_coef = translation_coef
        self.translation_smooth_coef = translation_smooth_coef
        self.translation_smooth_mode = translation_smooth_mode
        self.translation_smooth_loss_type = translation_smooth_loss_type
        self.translation_mode = translation_mode
        self.use_final_output_geometry = use_final_output_geometry
        self.rotation_coef = rotation_coef
        self.rot_velocity_coef = rot_velocity_coef
        self.upper_body_rot_weight = upper_body_rot_weight
        self.head_rotation_coef = head_rotation_coef
        self.head_rot_velocity_coef = head_rot_velocity_coef

        self.use_smpl_loss = (
            joint_position_coef > 0.0
            or hand_coef > 0.0
            or joint_velocity_coef > 0.0
            or hand_joint_velocity_coef > 0.0
            or wrist_position_coef > 0.0
            or lower_body_coef > 0.0
        )
        self.last_wrist_position_mse = None
        self.last_wrist_mean_distance = None
                              
        if self.use_smpl_loss:
            if motion_length is None:
                motion_length = self.motion_target.shape[0] if self.motion_target.dim() > 1 else 1
            self.smpl_model = smplx.create(
                BODY_MODEL_PATH_,
                model_type='smplh',
                gender='neutral',
                ext='npz',
                use_pca=False,
                batch_size=motion_length
            ).to(device)
             
    def __call__(self, x, mean=None, std=None):
        """Compute similarity loss between generated latent x and stored target.
        
        Args:
            x: [B, L, 201] normalized model latent.
            mean, std: model normalization stats [201].
        Returns:
            Scalar loss (body rot L2 + root rot L2, optionally + SMPL joint losses).
        """
        # Reference (135-format, column-interleaved): 
        #   0:6 root, 6:132 body(126), 132:135 trans
        # Generated (HY-Motion 201-format, column-interleaved):
        #   0:3 trans, 3:9 root, 9:135 body(126), ...
        
        # Unnormalize generated motion
        cur_mean = mean if mean is not None else self.mean
        cur_std = std if std is not None else self.std

        std_zero = cur_std < 1e-3
        cur_std = torch.where(std_zero, torch.zeros_like(cur_std), cur_std)
        x_unnorm = x * cur_std + cur_mean
        
        # Extract body rotations (126 dims) from generated
        # HY-Motion: 9:135
        gen_body_rot = x_unnorm[..., 9:135] 
        
        # Extract body rotations (126 dims) from target
        # Target is [L, 135] or [Frames, 135]
        target_body_rot = self.motion_target[..., 6:132]
        
        # Expand target to batch
        target_body_rot = target_body_rot.unsqueeze(0).expand(gen_body_rot.shape[0], -1, -1)
        
        # Handle length mismatch (crop to min length)
        min_len = min(gen_body_rot.shape[1], target_body_rot.shape[1])
        
        # Per-joint rotation distance (reference: loss_f.py)
        # Reshape body rotations to per-joint: [B, L, 21, 6]
        gen_joints_6d = gen_body_rot[:, :min_len].reshape(-1, min_len, 21, 6)
        tgt_joints_6d = target_body_rot[:, :min_len].reshape(-1, min_len, 21, 6)
        
        # Per-joint L2 norm of 6D rotation difference, then mean over joints and frames
        diff = gen_joints_6d - tgt_joints_6d                        # [B, L, 21, 6]
        per_joint_norm = torch.linalg.norm(diff, dim=-1)            # [B, L, 21]
        if self.upper_body_rot_weight != 1.0:
            body_weights = torch.ones(21, device=self.device)
            body_weights[11:] = self.upper_body_rot_weight
            per_joint_norm = per_joint_norm * body_weights.view(1, 1, 21)
        
        # Also compute root rotation loss (both are column-interleaved 6D format)
        gen_root_rot = x_unnorm[..., 3:9]
        target_root_rot = self.motion_target[..., 0:6]
        target_root_rot = target_root_rot.unsqueeze(0).expand(gen_root_rot.shape[0], -1, -1)
        
        diff_root = gen_root_rot[:, :min_len] - target_root_rot[:, :min_len]
        root_loss = torch.linalg.norm(diff_root, dim=-1).mean()
        
        if self.use_weighted_loss:
            # Weighted sum over joints (reference: cost_function_weighted in loss_f.py)
            base_loss = (per_joint_norm * self.joint_weights).sum(dim=-1).mean()  # weighted sum per frame, then mean over B,L
        else:
            base_loss = per_joint_norm.mean()                       # uniform mean
            
        base_loss = self.rotation_coef * base_loss + self.rotation_coef * root_loss

        if self.head_rotation_coef > 0.0:
            head_rot_loss = F.mse_loss(gen_joints_6d[:, :, 14], tgt_joints_6d[:, :, 14])
            base_loss = base_loss + self.head_rotation_coef * head_rot_loss

        if self.head_rot_velocity_coef > 0.0:
            gen_head_vel = gen_joints_6d[:, 1:, 14] - gen_joints_6d[:, :-1, 14]
            tgt_head_vel = tgt_joints_6d[:, 1:, 14] - tgt_joints_6d[:, :-1, 14]
            head_rot_vel_loss = F.mse_loss(gen_head_vel, tgt_head_vel)
            base_loss = base_loss + self.head_rot_velocity_coef * head_rot_vel_loss

        if self.rot_velocity_coef > 0.0:
            gen_rot_vel = gen_joints_6d[:, 1:] - gen_joints_6d[:, :-1]
            tgt_rot_vel = tgt_joints_6d[:, 1:] - tgt_joints_6d[:, :-1]
            rot_vel_loss = F.mse_loss(gen_rot_vel, tgt_rot_vel)
            base_loss = base_loss + self.rot_velocity_coef * rot_vel_loss
        
        if self.translation_coef > 0.0:
            gen_trans = x_unnorm[:, :min_len, 0:3]
            gen_pos_abs = gen_trans[:, :min_len]
            target_trans = self.motion_target[..., 132:135].unsqueeze(0).expand(gen_trans.shape[0], -1, -1)
            target_pos_abs = target_trans[:, :min_len]
            
            if self.translation_mode == "abs_3d":
                gen_pos_rel = gen_pos_abs - gen_pos_abs[:, :1]
                target_pos_rel = target_pos_abs - target_pos_abs[:, :1]
                trans_loss = F.mse_loss(gen_pos_rel, target_pos_rel)
            else:
                trans_loss = F.mse_loss(gen_pos_abs, target_pos_abs)
                
            base_loss = base_loss + self.translation_coef * trans_loss
            
            if self.translation_smooth_coef > 0.0:
                if self.translation_smooth_mode == "second_diff_abs":
                    # Second difference of absolute 3D position (acceleration / jitter)
                    # Shape: [B, L-2, 3]
                    gen_acc = gen_pos_abs[:, 2:] - 2 * gen_pos_abs[:, 1:-1] + gen_pos_abs[:, :-2]
                    target_acc = target_pos_abs[:, 2:] - 2 * target_pos_abs[:, 1:-1] + target_pos_abs[:, :-2]
                    
                    # Choose loss function
                    if self.translation_smooth_loss_type == "l1":
                        smooth_loss = F.l1_loss(gen_acc, target_acc)
                    elif self.translation_smooth_loss_type == "huber":
                        smooth_loss = F.smooth_l1_loss(gen_acc, target_acc)
                    else:  # mse
                        smooth_loss = F.mse_loss(gen_acc, target_acc)
                else:
                    gen_diff = gen_pos_abs[:, 1:] - gen_pos_abs[:, :-1]
                    target_diff = target_pos_abs[:, 1:] - target_pos_abs[:, :-1]
                    
                    if self.translation_smooth_loss_type == "l1":
                        smooth_loss = F.l1_loss(gen_diff, target_diff)
                    elif self.translation_smooth_loss_type == "huber":
                        smooth_loss = F.smooth_l1_loss(gen_diff, target_diff)
                    else:  # mse
                        smooth_loss = F.mse_loss(gen_diff, target_diff)
                
                base_loss = base_loss + self.translation_smooth_coef * smooth_loss
        
        if not self.use_smpl_loss:
            return base_loss
            
        loss_list = []
        batch_size = x.shape[0]
        
        # Format target and generated into 135-format for parameters_to_joints
        motion_target_unnorm = self.motion_target[:min_len] # [L, 135]
        
        # Try forwarding target; handle SMPL dynamic batch size update if min_len != motion_length
        try:
            joints_target = parameters_to_joints(motion_target_unnorm, self.smpl_model, self.device)
        except Exception:
            self.smpl_model = smplx.create(
                BODY_MODEL_PATH_, model_type='smplh', gender='neutral',
                ext='npz', use_pca=False, batch_size=min_len
            ).to(self.device)
            joints_target = parameters_to_joints(motion_target_unnorm, self.smpl_model, self.device)
            
        gen_body = x_unnorm[:, :min_len, 9:135]
        if self.use_final_output_geometry:
            # Match the geometry saved by Stage 2: generated rotations with the
            # original reference translation copied into the final 135-D motion.
            gen_root = x_unnorm[:, :min_len, 3:9]
            target_translation = motion_target_unnorm[:, 132:135].unsqueeze(0).expand(
                batch_size, -1, -1
            )
            gen_135 = torch.cat([gen_root, gen_body, target_translation], dim=-1)
        else:
            # Use generated rotations with generated absolute translation.
            gen_trans = x_unnorm[:, :min_len, 0:3]
            target_root = motion_target_unnorm[:, 0:6].unsqueeze(0).expand(
                batch_size, -1, -1
            )
            gen_135 = torch.cat([target_root, gen_body, gen_trans], dim=-1)

        wrist_position_losses = []
        wrist_mean_distances = []
        for i in range(batch_size):
            joints_pred = parameters_to_joints(gen_135[i], self.smpl_model, self.device)
            wrist_delta = joints_pred[:, [20, 21]] - joints_target[:, [20, 21]]
            wrist_position_loss = torch.mean(wrist_delta ** 2)
            wrist_mean_distance = torch.linalg.vector_norm(
                wrist_delta, dim=-1
            ).mean()
            wrist_position_losses.append(wrist_position_loss)
            wrist_mean_distances.append(wrist_mean_distance)
            
            sample_loss = 0.0
            if self.joint_position_coef > 0.0:
                joint_position_loss = F.mse_loss(joints_pred, joints_target)
                sample_loss += self.joint_position_coef * joint_position_loss
                
            if self.hand_coef > 0.0:
                hand_joint_position_loss = F.mse_loss(joints_pred[:, 23:], joints_target[:, 23:])
                sample_loss += self.hand_coef * hand_joint_position_loss

            if self.wrist_position_coef > 0.0:
                sample_loss += self.wrist_position_coef * wrist_position_loss
                
            if self.lower_body_coef > 0.0:
                lower_body_position_loss = F.mse_loss(joints_pred[:, [1,2,4,5,7,8,10,11]], joints_target[:, [1,2,4,5,7,8,10,11]])
                sample_loss += self.lower_body_coef * lower_body_position_loss
                
            if self.joint_velocity_coef > 0.0:
                gen_velocity = joints_pred[1:] - joints_pred[:-1]
                target_velocity = joints_target[1:] - joints_target[:-1]
                velocity_loss = F.mse_loss(gen_velocity, target_velocity)
                sample_loss += self.joint_velocity_coef * velocity_loss
                
            if self.hand_joint_velocity_coef > 0.0:
                hand_velocity = joints_pred[:, 23:][1:] - joints_pred[:, 23:][:-1]
                hand_target_velocity = joints_target[:, 23:][1:] - joints_target[:, 23:][:-1]
                hand_velocity_loss = F.mse_loss(hand_velocity, hand_target_velocity)
                sample_loss += self.hand_joint_velocity_coef * hand_velocity_loss
                
            loss_list.append(sample_loss)
            
        smpl_losses = torch.stack(loss_list).mean()
        self.last_wrist_position_mse = torch.stack(
            wrist_position_losses
        ).mean().detach().item()
        self.last_wrist_mean_distance = torch.stack(
            wrist_mean_distances
        ).mean().detach().item()
        return base_loss + smpl_losses

class MotionCollisionLoss:
    """
    Adapted loss function for HY-Motion that uses the DNO collision network.
    """
    
    def __init__(self, device, collision_threshold, mean=None, std=None, config_path=None, weights_path=None):
        # Load configuration
        if config_path is None:
            config_path = POSESHIELD_CONFIG_PATH
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.g_network = ResidualMLP(
            in_dim=self.cfg['MODEL']['IN_DIM'], 
            hidden_dim=self.cfg['MODEL']['HIDDEN_DIM'], 
            num_layers=self.cfg['MODEL']['NUM_LAYERS'],
            activation=self.cfg['MODEL'].get('ACTIVATION', 'relu')
        ).to(device)
        
        if weights_path is None:
            weights_path = POSESHIELD_WEIGHTS_PATH
        self.g_network.load_state_dict(torch.load(weights_path, map_location=device))
        self.g_network.eval()
        
        # Use provided mean/std (from HY-Motion) or load defaults (likely incompatible dims if mismatched)
        if mean is not None:
            self.mean = mean.to(device)
        else:
            self.mean = torch.from_numpy(np.load(MEAN_PATH_)).to(device)
            
        if std is not None:
            self.std = std.to(device)
        else:
            self.std = torch.from_numpy(np.load(STD_PATH_)).to(device)
            
        self.device = device
        self.collision_threshold = collision_threshold
    
    def unnormalize(self, x):
        return x * self.std + self.mean
    
    def __call__(self, xstart_in, mean=None, std=None, y=None):
        """
        Args:
            xstart_in: [bs, features, 1, frames] or [bs, frames, features](HY-Motion)
            mean: optional override for mean
            std: optional override for std
        """
        # Determine strict mean/std to use
        cur_mean = mean if mean is not None else self.mean
        cur_std = std if std is not None else self.std
        

        # Handle input shape flexibility
        if xstart_in.ndim == 3: # [bs, frames, features] (HY-Motion typical)
            motion_data = xstart_in
            batch_size = xstart_in.shape[0]
        elif xstart_in.ndim == 4: # [bs, features, 1, frames] (DNO typical)
             motion_data = xstart_in.permute(0, 2, 3, 1).squeeze(1)
             batch_size = xstart_in.shape[0]
        else:
            raise ValueError(f"Unexpected input shape: {xstart_in.shape}")
        
        loss_list = []
        
        for i in range(batch_size):
            sample_motion = motion_data[i] # [frames, features]
            
            # HY-Motion Slicing:
            # 0-3: Trans, 3-9: Root Rot, 9-135: Body Rot (21*6=126)
            # DNO network expects 126 dims.
            # We slice [:, 9:135]
            
            # unnorm_motion = self.unnormalize(sample_motion) # Old method used self.mean/std
            unnorm_motion = sample_motion * cur_std + cur_mean
            body_rot = unnorm_motion[:, 9:135]
            
            frames = body_rot.shape[0]
            body_rot_6d = body_rot.view(frames, 21, 6)
            R_3x3 = rotation_6d_to_matrix_torch(body_rot_6d)
            r1_r2 = R_3x3[..., :2, :]
            body_rot_dno = r1_r2.flatten(1)
            
            # Check shape
            if body_rot_dno.shape[1] != 126:
                # Fallback or error?
                # If using DNO original data, it was 6:-3 of 135 dims = 126.
                # If HY-Motion is 201 dims.
                raise ValueError(f"Sliced body rotation has shape {body_rot_dno.shape}, expected feature dim 126. Total dim: {unnorm_motion.shape[-1]}")

            gs = self.g_network(body_rot_dno) # [frames, 1]
            
            penalty = torch.relu(self.collision_threshold - gs)
            collision_loss = (penalty).mean() # mean
            
            loss_list.append(collision_loss)
        
        total_loss = torch.stack(loss_list)
        return total_loss.mean()

    def get_g(self, xstart_in, mean=None, std=None):
        """Returns the raw g values from the collision network for penetration depth estimation."""
        cur_mean = mean if mean is not None else self.mean
        cur_std = std if std is not None else self.std
        
        if xstart_in.ndim == 3:
            motion_data = xstart_in
            batch_size = xstart_in.shape[0]
        elif xstart_in.ndim == 4:
             motion_data = xstart_in.permute(0, 2, 3, 1).squeeze(1)
             batch_size = xstart_in.shape[0]
        else:
            raise ValueError(f"Unexpected input shape: {xstart_in.shape}")
        
        all_gs = []
        for i in range(batch_size):
            sample_motion = motion_data[i]
            unnorm_motion = sample_motion * cur_std + cur_mean
            body_rot = unnorm_motion[:, 9:135]
            
            frames = body_rot.shape[0]
            body_rot_6d = body_rot.view(frames, 21, 6)
            R_3x3 = rotation_6d_to_matrix_torch(body_rot_6d)
            r1_r2 = R_3x3[..., :2, :]
            body_rot_dno = r1_r2.flatten(1)
            
            gs = self.g_network(body_rot_dno) # [frames, 1]
            all_gs.append(gs)
        
        return torch.cat(all_gs, dim=0)

    def get_pd(self, xstart_in, mean=None, std=None):
        """Returns the estimated penetration depth torch.relu(-gs)"""
        gs = self.get_g(xstart_in, mean, std)
        return torch.relu(-gs)

import smplx

def rotation_6d_to_matrix_torch(d6: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Converts 6D rotation representation to a batch of 3x3 rotation matrices using
    Gram-Schmidt orthogonalisation (Zhou et al., CVPR 2019).
    
    Updated to parse HY-Motion's column-interleaved format:
    [c1_x, c2_x, c1_y, c2_y, c1_z, c2_z]

    Args:
        d6 (torch.Tensor): shape (..., 6)
        eps (float): numerical epsilon for normalization

    Returns:
        torch.Tensor: shape (..., 3, 3)
    """
    assert d6.size(-1) == 6, f"Expected last dim 6, got {d6.size(-1)}"

    # Unflatten from interleaved column format [c1_x, c2_x, c1_y, c2_y, c1_z, c2_z]
    # d6 is (..., 6) -> (..., 3, 2)
    cols = d6.view(*d6.shape[:-1], 3, 2)
    
    # Extract the two columns
    a1 = cols[..., 0] # First column, shape (..., 3)
    a2 = cols[..., 1] # Second column, shape (..., 3)

    # Normalize first column
    b1 = F.normalize(a1, dim=-1, eps=eps)

    # Make second column orthogonal to first, then normalize
    dot = (b1 * a2).sum(dim=-1, keepdim=True)          # (..., 1)
    b2 = a2 - dot * b1                                  # remove projection
    b2 = F.normalize(b2, dim=-1, eps=eps)

    # Right-handed system: b3 = b1 x b2
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack as columns to form rotation matrix (..., 3, 3)
    R = torch.stack((b1, b2, b3), dim=-1)
    return R

def matrix_to_axis_angle_torch(R: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Numerical stability analysis:
    1. This algorithm is numerically stable in most cases, especially for orthogonal, noise-free rotation matrices.
    2. However, when the angle is close to 0 or pi, sin(theta) approaches 0, the denominator may become small, introducing a risk of numerical instability.
    3. The code mitigates division by zero and tiny values by adding eps and using a small angle approximation (0.5 when s < 1e-6).
    4. If the input matrix deviates from orthogonality (e.g., due to cumulative error), acos and norm may bring instability. Orthogonalization before input is recommended.

    Args:
        R (torch.Tensor): shape (..., 3, 3), assumed near-orthogonal.
        eps (float): small epsilon to avoid division by zero.

    Returns:
        torch.Tensor: shape (..., 3)
    """
    # Compute trace
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    # Clamp trace range to prevent numerical overflow
    tr = torch.clamp(tr, -1.0 + eps, 3.0 - eps)
    # Compute rotation angle

    angles = torch.acos((tr - 1.0) / 2.0)  # (...,)

    # Compute vee(R - R^T)
    vee = torch.stack((
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ), dim=-1)  # (..., 3)

    # Compute sin(theta)
    s = torch.linalg.norm(vee, dim=-1, keepdim=True) / 2.0  # (..., 1)

    # Small angle approximation to avoid division by zero
    coeff = angles.unsqueeze(-1) / (2.0 * s + eps)          # (..., 1)
    small = s < 1e-6
    coeff = torch.where(small, 0.5 * torch.ones_like(coeff), coeff)

    axis_angle = coeff * vee  # (..., 3)
    return axis_angle

def parameters_to_joints(parameters, smpl_model, device):
    # parameters: [T, 135]
    T = parameters.shape[0]
    # SMPL's full pose has 72 dims: [3 (global) + 69 (23 joints × 3)]
    # We'll keep the global orientation at zero
    r_6d = parameters[:, 6:-3].reshape(T, 21, 6)
    rot_mats = rotation_6d_to_matrix_torch(r_6d)     # (T, 21, 3, 3)
    axis_angles = matrix_to_axis_angle_torch(rot_mats)  # (T, 21, 3)
    axis_angles = axis_angles.reshape(T, 63)
    
    global_orient_6d = parameters[:, :6].reshape(T, 6)
    global_rot_mats = rotation_6d_to_matrix_torch(global_orient_6d)
    global_axis_angles = matrix_to_axis_angle_torch(global_rot_mats)
    global_axis_angles = global_axis_angles.reshape(T, 3)
    trans = parameters[:, -3:]
    # Forward pass through SMPL
    output = smpl_model(
        global_orient=global_axis_angles,    # zero global orientation
        body_pose=axis_angles,        # 23 axis-angle joints, should be [1, 63]
        betas=None,
        transl=trans,
        return_verts=False
    )# T, N_J, 3
    # Return the joints from the SMPL output
    joints = output.joints # T, N_J=73, 3
    return joints

class MotionJointPositionSimilarityLoss:
    # TO be tested
    """
    Simplified loss function that works directly with motion data without SMPL processing
    This avoids gradient issues while still providing meaningful constraints
    """
    # NOTE: add joint position loss
    def __init__(self, motion_file,motion_length, hand_coef, joint_velocity_coef, hand_joint_velocity_coef, device, \
    lower_body_coef=0.0):
        motion_target = np.load(motion_file)
        assert motion_target.shape[0] == motion_length, f"motion_target.shape[0]: {motion_target.shape[0]}, motion_length: {motion_length}"
        self.motion_target = torch.from_numpy(motion_target).to(device) # [frames, 135] features, unnormalized
        mean = np.load(MEAN_PATH_)
        std = np.load(STD_PATH_)
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device
        self.smpl_model = smplx.create(
                BODY_MODEL_PATH_,
                model_type='smplh',
                gender='neutral',
                ext='npz',
                use_pca=False,
                batch_size=motion_length
            ).to(device)
        self.hand_coef = hand_coef
        self.joint_velocity_coef = joint_velocity_coef
        self.hand_joint_velocity_coef = hand_joint_velocity_coef
        self.lower_body_coef = lower_body_coef
        
    def __call__(self, xstart_in, y=None):
        """
        Args:
            xstart_in: [bs, features, 1, frames] - generated motion
            y: model kwargs
        Returns:
            loss: [bs] - loss for each sample in batch
        """
        batch_size = xstart_in.shape[0]
        
        # Convert motion data to [bs, frames, features] format
        # xstart_in shape: [bs, features, 1, frames]
        motion_data = xstart_in.permute(0, 2, 3, 1)  # [bs, 1, frames, features]
        motion_data = motion_data.squeeze(1)  # [bs, frames, features]
        
        # Collect losses in a list to avoid in-place operations
        loss_list = []
        
        for i in range(batch_size):
            # Get motion for this sample - keep gradients
            sample_motion = motion_data[i]  # [frames, features]
            
            # NOte: COMPUTE JOINT POSITION LOSS
            # INSERT_YOUR_CODE
            # Unnormalize the generated motion before passing to parameters_to_joints
            # INSERT_YOUR_CODE
            assert not torch.isnan(sample_motion).any(), (sample_motion,"sample_motion contains NaN values")
            sample_motion_unnorm = self.unnormalize(sample_motion)
            motion_target_unnorm = self.motion_target
            joints_pred = parameters_to_joints(sample_motion_unnorm, self.smpl_model, self.device) # t, 73, 3
            joints_target = parameters_to_joints(motion_target_unnorm, self.smpl_model, self.device) # t, 73, 3
            joint_position_loss = F.mse_loss(joints_pred, joints_target)
            hand_joint_position_loss = F.mse_loss(joints_pred[:, 23:], joints_target[:, 23:])
            # INSERT_YOUR_CODE
            lower_body_position_loss = F.mse_loss(joints_pred[:, [1,2,4,5,7,8,10,11]], joints_target[:, [1,2,4,5,7,8,10,11]])
            # Add velocity loss: encourage the velocity (frame-to-frame difference) to match the target
            # Compute velocity for generated and target motion: v[t] = x[t+1] - x[t]
            gen_velocity = joints_pred[1:] - joints_pred[:-1]  # [frames-1, 73, 3]
            target_velocity = joints_target[1:] - joints_target[:-1]  # [frames-1, 73, 3]
            velocity_loss = F.mse_loss(gen_velocity, target_velocity)
            
            hand_velocity = joints_pred[:, 23:][1:] - joints_pred[:, 23:][:-1]  # [frames-1, 50, 3]
            hand_target_velocity = joints_target[:, 23:][1:] - joints_target[:, 23:][:-1]  # [frames-1, 50, 3]
            hand_velocity_loss = F.mse_loss(hand_velocity, hand_target_velocity)
            # Add velocity loss to joint position loss (weighted sum, can tune weight if needed)
            sequence_loss = joint_position_loss + self.joint_velocity_coef * velocity_loss \
                + self.hand_coef * hand_joint_position_loss + self.hand_joint_velocity_coef * hand_velocity_loss + \
                lower_body_position_loss * self.lower_body_coef
            # Compute MSE loss directly on motion features
            assert not torch.isnan(joint_position_loss), "joint_position_loss contains NaN"
            assert not torch.isnan(hand_joint_position_loss), "hand_joint_position_loss contains NaN"

            # Add to loss list
            loss_list.append(sequence_loss)
        
        # Convert list to tensor
        total_loss = torch.stack(loss_list)
        return total_loss
    def unnormalize(self, x):
        return x * self.std + self.mean
