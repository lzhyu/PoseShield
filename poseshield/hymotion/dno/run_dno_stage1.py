"""
Stage 1: Standalone Similarity Fitting.
Optimizes the initial latent code z to fit the target reference motion (similarity only, no collision field).
Saves the optimized latent as `stage1_z.pt`.
"""

import sys
import os

# Insert workspace root and poseshield sub-directory to sys.path to allow absolute imports of poseshield and hymotion
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "poseshield"))

import argparse
import torch
import yaml
from poseshield.hymotion.utils.motion_format import load_motion

def get_args():
    parser = argparse.ArgumentParser(description="Stage 1 DNO: fit latent z to target motion (similarity only)")
    parser.add_argument("--model_path", type=str, default="ckpts/tencent/HY-Motion-1.0-Lite", help="Path to HY-Motion checkpoint")
    parser.add_argument("--text", type=str, default=None, help="Text prompt")
    parser.add_argument("--output_dir", type=str, default="dno_stage1_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to reference motion file (npy)")

    # Stage 1: pure similarity fitting
    parser.add_argument("--s1_steps", type=int, default=300, help="Stage 1 optimization steps")
    parser.add_argument("--s1_lr", type=float, default=0.05, help="Stage 1 learning rate")
    parser.add_argument("--s1_ode_steps", type=int, default=50, help="Number of ODE steps for Stage 1")
    parser.add_argument("--use_adjoint", action="store_true", help="Use adjoint method for ODE backpropagation")
    parser.add_argument("--no_use_adjoint", action="store_false", dest="use_adjoint", help="Use standard direct backpropagation (faster)")
    parser.set_defaults(use_adjoint=False)
    parser.add_argument("--s1_early_stop_err", type=float, default=0.005, help="Stop optimization if joint position error is below this threshold (set <= 0 to disable)")
    parser.add_argument("--s1_phase1_steps", type=int, default=0, help="Number of steps in Phase 1 (pure rot + trans loss, no SMPL FK). Default 0 (phased optimization disabled)")

    # Stage 1 similarity coefficients
    parser.add_argument("--s1_joint_position_coef", type=float, default=0.1)
    parser.add_argument("--s1_hand_coef", type=float, default=0.1)
    parser.add_argument("--s1_joint_velocity_coef", type=float, default=0.1)
    parser.add_argument("--s1_hand_joint_velocity_coef", type=float, default=0.0)
    parser.add_argument("--s1_lower_body_coef", type=float, default=0.0)
    parser.add_argument("--s1_translation_coef", type=float, default=0.0, help="Weight coefficient for translation loss in Stage 1")
    parser.add_argument("--s1_translation_smooth_coef", type=float, default=0.0, help="Weight coefficient for translation smoothness loss in Stage 1")
    parser.add_argument("--s1_translation_smooth_mode", type=str, default="first_diff", choices=["first_diff", "second_diff_abs"], help="Translation smoothing loss mode")
    parser.add_argument("--s1_translation_smooth_loss_type", type=str, default="mse", choices=["mse", "l1", "huber"], help="Loss type for translation smoothing")
    parser.add_argument("--s1_weighted_rot_loss", action="store_true", help="Use subtree-size weighted rotation loss for Stage 1")
    parser.add_argument("--no_s1_weighted_rot_loss", action="store_false", dest="s1_weighted_rot_loss", help="Do not use subtree-size weighted rotation loss for Stage 1")
    parser.set_defaults(s1_weighted_rot_loss=True)

    # LR Scheduler options
    parser.add_argument("--s1_lr_warmup_steps", type=int, default=50, help="Number of warmup steps for Stage 1 LR scheduler")
    parser.add_argument("--s1_lr_decay_steps", type=int, default=None, help="Number of decay steps for Stage 1 LR scheduler")
    return parser.parse_args()

def main():
    args = get_args()
    from poseshield.hymotion.pipeline.motion_diffusion import MotionFlowMatching, length_to_mask
    from poseshield.hymotion.dno.dno_solver import DNOSolver, DNOOptions, ode_loop_with_gradient
    from poseshield.hymotion.dno.dno_loss import MotionSimilarityLoss, MotionCollisionLoss

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ─── 1. Load Model ───────────────────────────────────────────────────────
    print("Loading model...")
    config_path = os.path.join(args.model_path, "config.yml")
    if not os.path.exists(config_path):
        config_path = os.path.join(args.model_path, "config.yaml")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    network_cfg = cfg.get('network_module_args',
                          cfg.get('model', {}).get('params', {}).get('network_config', {}).get('params', {}))
    input_dim = network_cfg.get('input_dim', 201)

    model = MotionFlowMatching(
        network_module="hymotion.network.hymotion_mmdit.HunyuanMotionMMDiT",
        network_module_args=network_cfg,
        text_encoder_module="hymotion.network.text_encoders.text_encoder.HYTextModel",
        text_encoder_cfg={"llm_type": "qwen3", "max_length_llm": 128},
        noise_scheduler_cfg={"method": "euler"},
        infer_noise_scheduler_cfg={"validation_steps": 50},
        mean_std_dir=os.path.join(args.model_path, "stats")
    ).to(device)

    ckpt_path = os.path.join(args.model_path, "latest.ckpt")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        # Resize blank mean/std buffers to match checkpoint (stats dir may be missing)
        for key in ("mean", "std"):
            if key in state_dict and hasattr(model, key):
                ckpt_shape = state_dict[key].shape
                buf_shape = getattr(model, key).shape
                if ckpt_shape != buf_shape:
                    print(f"Resizing buffer '{key}': {buf_shape} -> {ckpt_shape}")
                    buf = torch.zeros(ckpt_shape, device=device) if key == "mean" else torch.ones(ckpt_shape, device=device)
                    model.register_buffer(key, buf)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model.eval()

    # ─── 2. Load Reference Motion ────────────────────────────────────────────
    motion_np = load_motion(args.motion_file)
    target_motion = torch.from_numpy(motion_np).float().to(device)
    seq_len = target_motion.shape[0]
    print(f"Reference motion loaded: {seq_len} frames (canonical absolute-XYZ)")

    # ─── 3. Prepare Model Inputs ─────────────────────────────────────────────
    bsz = 1
    y0 = torch.randn(bsz, seq_len, input_dim, device=device)

    if args.text:
        hidden_state = model.encode_text({"text": [args.text]})
        ctxt_input = hidden_state["text_ctxt_raw"].to(device)
        vtxt_input = hidden_state["text_vec_raw"].to(device)
        ctxt_length = hidden_state["text_ctxt_raw_length"].to(device)
        if len(vtxt_input.shape) == 2:
            vtxt_input = vtxt_input.unsqueeze(0)
            ctxt_input = ctxt_input.unsqueeze(0)
            ctxt_length = ctxt_length.unsqueeze(0)
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
    else:
        print("Using unconditional mode.")
        vtxt_input = model.null_vtxt_feat.expand(bsz, 1, -1).to(device)
        ctxt_input = model.null_ctxt_input.expand(bsz, 1, -1).to(device)
        ctxt_length = torch.tensor([1], device=device).expand(bsz)
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1]).expand(bsz, -1)

    x_length = torch.LongTensor([seq_len] * bsz).to(device)
    x_mask_temporal = length_to_mask(x_length, seq_len)

    model_kwargs = {
        "ctxt_input": ctxt_input, "vtxt_input": vtxt_input,
        "x_mask_temporal": x_mask_temporal, "ctxt_mask_temporal": ctxt_mask_temporal
    }

    u_seq_len = ctxt_input.shape[1]
    u_ctxt_input = (model.null_ctxt_input.expand(bsz, u_seq_len, -1).to(device)
                    if model.enable_ctxt_null_feat else ctxt_input.clone())
    u_vtxt_input = model.null_vtxt_feat.expand(bsz, vtxt_input.shape[1], -1).to(device)
    uncond_model_kwargs = {
        "ctxt_input": u_ctxt_input, "vtxt_input": u_vtxt_input,
        "ctxt_mask_temporal": ctxt_mask_temporal
    }

    t_span = torch.linspace(0, 1, args.s1_ode_steps).to(device)

    def model_fn(z):
        return ode_loop_with_gradient(
            model=model.motion_transformer, y0=z, t_span=t_span,
            model_kwargs=model_kwargs, noise_scheduler_cfg={"method": "euler"},
            cfg_scale=args.cfg_scale, uncond_model_kwargs=uncond_model_kwargs,
            use_adjoint=args.use_adjoint
        )

    # FBX converter
    try:
        from poseshield.hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
        from poseshield.hymotion.pipeline.body_model import construct_smpl_data_dict
        fbx_converter = SMPLH2WoodFBX()
        fbx_available = True
    except Exception as e:
        print(f"FBX not available: {e}")
        fbx_available = False
        fbx_converter = None

    def save_fbx(x_tensor, name):
        if not fbx_available:
            return
        with torch.no_grad():
            decoded = model.decode_motion_from_latent(x_tensor, should_apply_smooothing=True)
        for i in range(x_tensor.shape[0]):
            smpl_data = construct_smpl_data_dict(decoded['rot6d'][i], decoded['transl'][i])
            out_path = os.path.join(args.output_dir, f"{name}_{i:03d}.fbx")
            if fbx_converter.convert_npz_to_fbx(smpl_data, out_path):
                print(f"Saved FBX: {out_path}")

    # ─── 4. Build Losses ─────────────────────────────────────────────────────
    loss_similarity_s1 = MotionSimilarityLoss(
        target_motion, device, model.mean, model.std,
        joint_position_coef=args.s1_joint_position_coef,
        hand_coef=args.s1_hand_coef,
        joint_velocity_coef=args.s1_joint_velocity_coef,
        hand_joint_velocity_coef=args.s1_hand_joint_velocity_coef,
        lower_body_coef=args.s1_lower_body_coef,
        translation_coef=args.s1_translation_coef,
        translation_smooth_coef=args.s1_translation_smooth_coef,
        use_weighted_loss=args.s1_weighted_rot_loss,
        translation_smooth_mode=args.s1_translation_smooth_mode,
        translation_smooth_loss_type=args.s1_translation_smooth_loss_type,
        translation_mode="abs_3d"
    )
    loss_collision = MotionCollisionLoss(device, collision_threshold=999.0) # monitor only

    loss_logs = {"similarity": [], "collision": []}
    joint_pos_err_history = []

    # Pre-build SMPL model for logging joint position MSE
    import smplx
    from poseshield.hymotion.dno.dno_loss import BODY_MODEL_PATH_, parameters_to_joints
    min_len = min(seq_len, target_motion.shape[0])
    smpl_model_eval = smplx.create(
        BODY_MODEL_PATH_, model_type='smplh', gender='neutral',
        ext='npz', use_pca=False, batch_size=min_len
    ).to(device)

    with torch.no_grad():
        motion_target_unnorm = target_motion[:min_len]
        joints_target = parameters_to_joints(motion_target_unnorm, smpl_model_eval, device)

    # ─── 5. Run Stage 1 Optimization ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1 STANDALONE: Fitting z to target (similarity only)")
    print("=" * 60)

    step_count = [0]
    def criterion_s1(x):
        if args.s1_phase1_steps > 0:
            if step_count[0] < args.s1_phase1_steps:
                # Phase 1: pure rotation + translation loss (no SMPL FK)
                loss_similarity_s1.use_smpl_loss = False
            else:
                # Phase 2: restore SMPL FK losses
                loss_similarity_s1.use_smpl_loss = (
                    args.s1_joint_position_coef > 0.0 or args.s1_hand_coef > 0.0 or
                    args.s1_joint_velocity_coef > 0.0 or args.s1_hand_joint_velocity_coef > 0.0 or
                    args.s1_lower_body_coef > 0.0
                )
        l_sim = loss_similarity_s1(x, model.mean, model.std)
        with torch.no_grad():
            l_col = loss_collision(x, model.mean, model.std)
            # Compute joint position error for tracking (only every 10 steps to save time)
            # Also track at the last step of Phase 1 (args.s1_phase1_steps - 1) to inspect its accuracy
            should_track = (
                step_count[0] % 10 == 0 or 
                step_count[0] == args.s1_steps - 1 or 
                (args.s1_phase1_steps > 0 and step_count[0] == args.s1_phase1_steps - 1)
            )
            if should_track:
                cur_mean = model.mean
                cur_std = model.std
                std_zero = cur_std < 1e-3
                std_safe = torch.where(std_zero, torch.ones_like(cur_std), cur_std)
                x_unnorm = x * std_safe + cur_mean
                
                target_root = motion_target_unnorm[:, 0:6].unsqueeze(0).expand(x.shape[0], -1, -1)
                gen_body = x_unnorm[:, :min_len, 9:135]
                target_translation = motion_target_unnorm[:, 132:135].unsqueeze(0).expand(
                    x.shape[0], -1, -1
                )
                
                gen_135 = torch.cat([target_root, gen_body, target_translation], dim=-1)
                # Assuming batch_size=1
                joints_pred = parameters_to_joints(gen_135[0], smpl_model_eval, device)
                step_joint_err = torch.nn.functional.mse_loss(joints_pred, joints_target).item()
            else:
                step_joint_err = joint_pos_err_history[-1] if joint_pos_err_history else 9.99

            joint_pos_err_history.append(step_joint_err)
            stop_early = (args.s1_early_stop_err > 0 and step_joint_err <= args.s1_early_stop_err)
            step_count[0] += 1

        loss_logs["similarity"].append(l_sim.item())
        loss_logs["collision"].append(l_col.item())
        return l_sim, {"sim": l_sim.item(), "col": l_col.item(), "joint_err": step_joint_err, "stop_early": stop_early}

    s1_options = DNOOptions(
        num_opt_steps=args.s1_steps,
        lr=args.s1_lr,
        perturb_scale=0,
        diff_penalty_scale=0,
        lr_warm_up_steps=args.s1_lr_warmup_steps,
        lr_decay_steps=args.s1_lr_decay_steps
    )
    import time
    start_time = time.time()
    solver_s1 = DNOSolver(model_fn, criterion_s1, y0, s1_options, col_threshold=999.0)
    results_s1 = solver_s1()
    run_time = time.time() - start_time

    z_fitted = results_s1["z"]
    print(f"Stage 1 done. Final sim loss: {loss_logs['similarity'][-1]:.6f}, "
          f"col loss (monitor only): {loss_logs['collision'][-1]:.6f}")

    # Determine joint_pos_err and steps_to_target
    final_joint_pos_err = joint_pos_err_history[-1] if joint_pos_err_history else 9.99
    
    # steps_to_target: step index (1-based) where joint_pos_err <= 0.01
    steps_to_target = 999
    for idx, val in enumerate(joint_pos_err_history):
        if val <= 0.01:
            steps_to_target = idx + 1
            break

    # Calculate translation velocity and jitter ratios in absolute XYZ space.
    with torch.no_grad():
        x_fitted_device = results_s1["x"].to(device)
        cur_mean = model.mean
        cur_std = model.std
        std_safe = torch.where(cur_std < 1e-3, torch.ones_like(cur_std), cur_std)
        x_unnorm = x_fitted_device * std_safe + cur_mean
        
        # Generated absolute root translation [abs_x, abs_y, abs_z]
        gen_trans_abs = x_unnorm[0, :min_len, 0:3] # [L, 3]
        
        ref_trans_abs = target_motion[:min_len, 132:135]
        
        # Frame-to-frame velocity
        gen_vel = gen_trans_abs[1:] - gen_trans_abs[:-1]
        ref_vel = ref_trans_abs[1:] - ref_trans_abs[:-1]
        
        gen_vel_norm = torch.linalg.norm(gen_vel, dim=-1)
        ref_vel_norm = torch.linalg.norm(ref_vel, dim=-1)
        
        mean_gen_vel = gen_vel_norm.mean().item()
        mean_ref_vel = ref_vel_norm.mean().item()
        trans_vel_ratio = mean_gen_vel / (mean_ref_vel + 1e-8)
        
        # Frame-to-frame acceleration (second difference) to measure jitter
        gen_acc = gen_trans_abs[2:] - 2 * gen_trans_abs[1:-1] + gen_trans_abs[:-2]
        ref_acc = ref_trans_abs[2:] - 2 * ref_trans_abs[1:-1] + ref_trans_abs[:-2]
        
        gen_acc_norm = torch.linalg.norm(gen_acc, dim=-1)
        ref_acc_norm = torch.linalg.norm(ref_acc, dim=-1)
        
        mean_gen_acc = gen_acc_norm.mean().item()
        mean_ref_acc = ref_acc_norm.mean().item()
        trans_jitter_ratio = mean_gen_acc / (mean_ref_acc + 1e-8)

    # Print final results with a clean prefix for parsing
    print(f"RESULT joint_pos_err={final_joint_pos_err:.6f}")
    print(f"RESULT trans_vel_ratio={trans_vel_ratio:.6f}")
    print(f"RESULT trans_jitter_ratio={trans_jitter_ratio:.6f}")
    print(f"RESULT steps_to_target={steps_to_target}")
    print(f"RESULT run_time={run_time:.2f}")

    # Save outputs
    torch.save(z_fitted.cpu(), os.path.join(args.output_dir, "stage1_z.pt"))
    torch.save(results_s1["x"].cpu(), os.path.join(args.output_dir, "stage1_x.pt"))
    save_fbx(results_s1["x"], "stage1")
    
    # Save target FBX
    with torch.no_grad():
        s1_x = model_fn(z_fitted)
        target_x = s1_x.clone()
        cur_mean = model.mean
        cur_std = model.std
        std_safe = torch.where(cur_std < 1e-3, torch.ones_like(cur_std), cur_std)
        tgt_body = target_motion[..., 6:132].unsqueeze(0).expand(target_x.shape[0], -1, -1)
        target_x[..., 9:135] = (tgt_body - cur_mean[..., 9:135]) / std_safe[..., 9:135]
        target_decoded = model.decode_motion_from_latent(target_x)
        if fbx_available:
            t_rot = target_decoded['rot6d']
            t_trans = target_decoded['transl']
            if t_rot.shape[0] == 1:
                t_rot = t_rot[0]
                t_trans = t_trans[0]
            tgt_smpl = construct_smpl_data_dict(t_rot, t_trans)
            tgt_fbx_path = os.path.join(args.output_dir, "target.fbx")
            fbx_converter.convert_npz_to_fbx(tgt_smpl, tgt_fbx_path)

    import json
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
        
    print(f"Saved stage1_z.pt and stage1_x.pt to {args.output_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
