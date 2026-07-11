"""Run standalone Stage 2 collision resolution from a frozen Stage 1 latent."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

# Insert workspace root and poseshield sub-directory to sys.path to allow absolute imports of poseshield and hymotion
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "poseshield"))

import numpy as np
import torch
import yaml

from poseshield.hymotion.dno.stage2_metrics import compute_motion_metrics
from poseshield.hymotion.utils.motion_format import latent_to_public_motion, load_motion

def get_args() -> argparse.Namespace:
    """Parse Stage 2 command-line arguments."""
    parser = argparse.ArgumentParser(description="Stage 2 DNO: resolve collisions starting from Stage 1 z")
    parser.add_argument("--model_path", type=str, default="ckpts/tencent/HY-Motion-1.0-Lite", help="Path to HY-Motion checkpoint")
    parser.add_argument("--text", type=str, default=None, help="Text prompt")
    parser.add_argument("--output_dir", type=str, default="dno_stage2_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--motion_file", type=str, required=True, help="Path to reference motion file (npy)")
    parser.add_argument("--stage1_z", type=str, required=True, help="Path to cached stage1_z.pt (required)")
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Optional collision-free ground-truth motion",
    )
    parser.add_argument("--ode_steps", type=int, default=20, help="Number of Stage 2 ODE integration points")
    parser.add_argument("--save_fbx", action="store_true", help="Save an FBX visualization artifact")
    parser.add_argument(
        "--evaluate_penetration",
        action="store_true",
        help="Run the slower exact-mesh penetration-depth diagnostic",
    )
    parser.add_argument(
        "--save_checkpoint_motions_every",
        type=int,
        default=10,
        help=(
            "If positive, save public-format motion snapshots every N Stage 2 "
            "steps for exact-FCL checkpoint selection. Default 10."
        ),
    )
    parser.add_argument(
        "--checkpoint_motion_dir",
        type=str,
        default=None,
        help="Optional output directory for checkpoint motion snapshots.",
    )
    parser.add_argument(
        "--exact_fcl_select_checkpoint",
        dest="exact_fcl_select_checkpoint",
        action="store_true",
        default=True,
        help=(
            "After Stage 2, run exact-FCL on saved checkpoint motions and use "
            "the exact-aware checkpoint as the final output. Requires "
            "--save_checkpoint_motions_every > 0. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no_exact_fcl_select_checkpoint",
        dest="exact_fcl_select_checkpoint",
        action="store_false",
        help="Disable exact-FCL checkpoint selection and keep the proxy-selected Stage 2 output.",
    )
    parser.add_argument(
        "--exact_fcl_selection_dir",
        type=str,
        default=None,
        help="Directory for exact-FCL checkpoint-selection records.",
    )
    parser.add_argument(
        "--exact_fcl_selection_distances",
        type=str,
        default="deps/topology_distances_30_60.npz",
        help="Mesh topology distances used by exact-FCL checkpoint selection.",
    )
    parser.add_argument(
        "--exact_fcl_selection_device",
        type=str,
        default="cpu",
        help=(
            "Device for exact-FCL checkpoint selection. Defaults to CPU to "
            "avoid competing with the loaded Stage 2 model for GPU memory."
        ),
    )
    parser.add_argument(
        "--exact_fcl_selection_topology_threshold",
        type=int,
        default=None,
        help="Topology threshold for exact-FCL checkpoint selection.",
    )
    parser.add_argument(
        "--exact_fcl_selection_resume",
        action="store_true",
        help=(
            "Reuse existing exact-FCL checkpoint result JSON files when present. "
            "Default false forces recomputation to avoid stale cached results."
        ),
    )
    parser.add_argument(
        "--exact_fcl_selection_proxy_col_threshold",
        type=float,
        default=1e-3,
        help=(
            "If set, run exact-FCL checkpoint selection only on checkpoints "
            "whose proxy collision loss is at or below this threshold. If no "
            "checkpoint qualifies, the minimum-proxy-collision checkpoint is "
            "evaluated as a fallback. Default 1e-3."
        ),
    )

    # Stage 2: similarity + collision
    parser.add_argument("--s2_steps", type=int, default=100, help="Stage 2 optimization steps")
    parser.add_argument("--s2_lr", type=float, default=0.02, help="Stage 2 learning rate")
    parser.add_argument("--s2_lr_warmup_steps", type=int, default=50)
    parser.add_argument("--s2_lr_decay_steps", type=int, default=None)
    parser.add_argument("--collision_threshold", type=float, default=0.08, help="Collision field threshold")
    parser.add_argument("--collision_scale", type=float, default=6.0, help="Collision loss weight (Stage 2)")
    parser.add_argument("--similarity_scale", type=float, default=1.0, help="Similarity loss weight")
    parser.add_argument(
        "--col_return_threshold",
        type=float,
        default=1e-4,
        help="Maximum collision loss for fidelity-based checkpoint selection",
    )

    # Stage 2 similarity coefficients
    parser.add_argument("--joint_position_coef", type=float, default=0.0)
    parser.add_argument("--hand_coef", type=float, default=1.0)
    parser.add_argument("--joint_velocity_coef", type=float, default=0.0)
    parser.add_argument("--hand_joint_velocity_coef", type=float, default=0.5)
    parser.add_argument("--wrist_position_coef", type=float, default=12.0)
    parser.add_argument("--lower_body_coef", type=float, default=0.0)
    parser.add_argument("--joint_jerk_coef", type=float, default=0.0)
    parser.add_argument("--lower_body_joint_jerk_coef", type=float, default=0.0)
    parser.add_argument("--rotation_velocity_scale", type=float, default=5.0)
    parser.add_argument("--upper_body_velocity_scale", type=float, default=3.0)
    parser.add_argument("--upper_body_rotation_scale", type=float, default=0.0)
    parser.add_argument("--weighted_rot_loss", action="store_true", help="Use subtree-size weighted rotation loss")
    parser.add_argument("--no_weighted_rot_loss", action="store_false", dest="weighted_rot_loss")
    parser.set_defaults(weighted_rot_loss=True)
    return parser.parse_args()


def _load_inputs(args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    """Load canonical motion and its sequence-matched frozen Stage 1 latent."""
    motion_path = Path(args.motion_file).resolve()
    stage1_path = Path(args.stage1_z).resolve()
    if not stage1_path.is_file():
        raise FileNotFoundError(f"Stage 1 latent not found: {stage1_path}")

    motion_np = load_motion(
        motion_path,
        translation_layout="xyz",
        rotation_joint_layout="root_first",
    )

    z_fitted = torch.load(stage1_path, map_location=device)
    if not torch.is_tensor(z_fitted):
        raise TypeError(f"Expected a tensor in {stage1_path}, got {type(z_fitted)}")
    if z_fitted.dim() == 2:
        z_fitted = z_fitted.unsqueeze(0)
    if z_fitted.dim() != 3 or z_fitted.shape[0] != 1 or z_fitted.shape[2] != 201:
        raise ValueError(f"Expected Stage 1 latent shape [1, frames, 201], got {tuple(z_fitted.shape)}")
    if z_fitted.shape[1] != motion_np.shape[0]:
        raise ValueError(
            "Stage 1 latent length does not match the motion: "
            f"{z_fitted.shape[1]} != {motion_np.shape[0]}"
        )
    return motion_np, z_fitted


def _translation_135_to_absolute(translation: torch.Tensor) -> torch.Tensor:
    """Return absolute global XYZ translation from the public motion format."""
    if translation.ndim != 2 or translation.shape[1] != 3:
        raise ValueError(f"Expected reference translation [frames, 3], got {translation.shape}")
    return translation


def _latent_to_motion_135(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    reference_translation: torch.Tensor,
) -> torch.Tensor:
    """Convert HY-Motion rotations to public 135-D format and copy reference translation."""
    return latent_to_public_motion(x, mean, std, reference_translation)


def _load_ground_truth(path: str) -> np.ndarray:
    """Load a ground-truth motion in the public canonical format."""
    return load_motion(
        path,
        translation_layout="xyz",
        rotation_joint_layout="root_first",
    )


def _load_checkpoint_metadata(metadata_path: Path) -> list[dict]:
    """Load checkpoint metadata JSONL records."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
    records: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("step", "motion", "x", "z", "checkpoint_score"):
                if key not in record:
                    raise ValueError(
                        f"Missing '{key}' in {metadata_path}:{line_number}"
                    )
            records.append(record)
    if not records:
        raise ValueError(f"No checkpoint records found in {metadata_path}")
    return records


def _run_exact_fcl_for_checkpoint(
    *,
    motion_path: Path,
    output_dir: Path,
    distances_path: Path,
    device: str,
    topology_threshold: int,
    log_path: Path,
    resume: bool = False,
) -> dict:
    """Run exact-FCL for one checkpoint motion and return parsed results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "exact_fcl_results.json"
    if resume and result_path.is_file():
        with result_path.open("r", encoding="utf-8") as handle:
            return {
                "returncode": 0,
                "log": str(log_path),
                "result_path": str(result_path),
                "results": json.load(handle),
                "reused": True,
            }

    command = [
        sys.executable,
        "tools/evaluate_exact_fcl.py",
        "--motion",
        str(motion_path),
        "--output-dir",
        str(output_dir),
        "--distances",
        str(distances_path),
        "--device",
        device,
        "--topology-threshold",
        str(topology_threshold),
    ]
    run = subprocess.run(
        command,
        cwd=workspace_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.write(run.stdout)

    if not result_path.is_file():
        raise RuntimeError(
            f"Exact-FCL did not write {result_path}; see {log_path}"
        )
    with result_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    return {
        "returncode": run.returncode,
        "log": str(log_path),
        "result_path": str(result_path),
        "results": results,
        "reused": False,
    }


def _select_checkpoint_by_exact_fcl(
    *,
    metadata_path: Path,
    output_dir: Path,
    distances_path: Path,
    device: str,
    topology_threshold: int,
    resume: bool = False,
    proxy_col_threshold: Optional[float] = None,
) -> dict:
    """Select the best saved checkpoint using exact-FCL validation."""
    records = _load_checkpoint_metadata(metadata_path)
    candidate_source = "all_checkpoints"
    candidate_records = records
    if proxy_col_threshold is not None:
        def proxy_col(record: dict) -> float:
            value = record.get("col")
            return float(value) if value is not None else float("inf")

        finite_proxy_records = [
            record for record in records if proxy_col(record) < float("inf")
        ]
        if finite_proxy_records:
            candidate_records = [
                record
                for record in records
                if proxy_col(record) <= float(proxy_col_threshold)
            ]
            candidate_source = "proxy_collision_threshold"
            if not candidate_records:
                candidate_records = [
                    min(
                        finite_proxy_records,
                        key=lambda item: (proxy_col(item), float(item.get("checkpoint_score", float("inf")))),
                    )
                ]
                candidate_source = "minimum_proxy_collision_fallback"
        else:
            candidate_source = "all_checkpoints_no_proxy_col"
            candidate_records = records

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated: list[dict] = []
    for record in candidate_records:
        step = int(record["step"])
        step_name = f"step_{step:04d}"
        exact = _run_exact_fcl_for_checkpoint(
            motion_path=Path(record["motion"]).resolve(),
            output_dir=output_dir / step_name,
            distances_path=distances_path.resolve(),
            device=device,
            topology_threshold=topology_threshold,
            log_path=output_dir / f"{step_name}.log",
            resume=resume,
        )
        evaluated_record = {
            **record,
            "exact_fcl": {
                "returncode": exact["returncode"],
                "log": exact["log"],
                "result_path": exact["result_path"],
                "reused": exact["reused"],
            },
            "exact_fcl_results": exact["results"],
        }
        with (output_dir / f"{step_name}.record.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(evaluated_record, handle, indent=2)
        evaluated.append(evaluated_record)

    def checkpoint_score(record: dict) -> float:
        value = record.get("checkpoint_score")
        return float(value) if value is not None else float("inf")

    def exact_mean(record: dict) -> float:
        results = record["exact_fcl_results"]
        return float(results.get("mean_penetration_depth", float("inf")))

    def exact_frames(record: dict) -> int:
        results = record["exact_fcl_results"]
        return int(results.get("num_collision_frames", 10**9))

    exact_free = [
        record
        for record in evaluated
        if bool(record["exact_fcl_results"].get("exact_collision_free", False))
    ]
    if exact_free:
        selected = min(exact_free, key=lambda item: (checkpoint_score(item), exact_mean(item)))
        reason = "exact_collision_free_lowest_checkpoint_score"
    else:
        selected = min(
            evaluated,
            key=lambda item: (exact_mean(item), exact_frames(item), checkpoint_score(item)),
        )
        reason = "minimum_exact_fcl_mean_penetration"

    summary = {
        "selection_reason": reason,
        "topology_threshold": int(topology_threshold),
        "proxy_col_threshold": proxy_col_threshold,
        "candidate_source": candidate_source,
        "num_total_checkpoints": len(records),
        "num_candidate_checkpoints": len(candidate_records),
        "num_evaluated_checkpoints": len(evaluated),
        "num_reused_exact_fcl_results": sum(
            1 for record in evaluated if record["exact_fcl"].get("reused", False)
        ),
        "selected_step": int(selected["step"]),
        "selected_motion": selected["motion"],
        "selected_x": selected["x"],
        "selected_z": selected["z"],
        "selected_checkpoint_score": selected.get("checkpoint_score"),
        "selected_exact_collision_free": bool(
            selected["exact_fcl_results"].get("exact_collision_free", False)
        ),
        "selected_num_collision_frames": int(
            selected["exact_fcl_results"].get("num_collision_frames", -1)
        ),
        "selected_mean_penetration_depth": exact_mean(selected),
        "records": [
            {
                "step": int(record["step"]),
                "motion": record["motion"],
                "proxy_col": record.get("col"),
                "checkpoint_score": record.get("checkpoint_score"),
                "exact_collision_free": bool(
                    record["exact_fcl_results"].get("exact_collision_free", False)
                ),
                "num_collision_frames": int(
                    record["exact_fcl_results"].get("num_collision_frames", -1)
                ),
                "mean_penetration_depth": exact_mean(record),
            }
            for record in evaluated
        ],
    }
    with (output_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    """Optimize Stage 2, save artifacts, and report frozen research metrics."""
    args = get_args()
    from poseshield.hymotion.pipeline.motion_diffusion import MotionFlowMatching, length_to_mask
    from poseshield.hymotion.dno.dno_solver import DNOSolver, DNOOptions, ode_loop_with_gradient
    from poseshield.hymotion.dno.dno_loss import (
        BODY_MODEL_PATH_,
        MotionCollisionLoss,
        MotionSimilarityLoss,
        MotionTemporalRegularizer,
        matrix_to_axis_angle_torch,
        rotation_6d_to_matrix_torch,
    )

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

    # ─── 2. Load Canonical Reference and Frozen Stage 1 Latent ──────────────
    motion_np, z_fitted = _load_inputs(args, device)
    target_motion = torch.from_numpy(motion_np).float().to(device)
    reference_absolute_translation = _translation_135_to_absolute(
        target_motion[:, 132:135]
    )
    seq_len = target_motion.shape[0]
    print(f"Reference motion loaded: {seq_len} frames (canonical absolute-XYZ)")
    print(f"Loaded frozen Stage 1 z: {args.stage1_z}")

    # ─── 4. Prepare Model Inputs ─────────────────────────────────────────────
    bsz = 1
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

    if args.ode_steps < 2:
        raise ValueError(f"--ode_steps must be at least 2, got {args.ode_steps}")
    t_span = torch.linspace(0, 1, args.ode_steps).to(device)

    def model_fn(z):
        return ode_loop_with_gradient(
            model=model.motion_transformer, y0=z, t_span=t_span,
            model_kwargs=model_kwargs, noise_scheduler_cfg={"method": "euler"},
            cfg_scale=args.cfg_scale, uncond_model_kwargs=uncond_model_kwargs
        )

    # FBX converter
    fbx_available = False
    fbx_converter = None
    if args.save_fbx:
        try:
            from poseshield.hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
            from poseshield.hymotion.pipeline.body_model import construct_smpl_data_dict

            fbx_converter = SMPLH2WoodFBX()
            fbx_available = True
        except Exception as error:
            print(f"FBX not available: {error}")

    def save_fbx(x_tensor: torch.Tensor, name: str) -> None:
        """Save an optional FBX artifact for visual inspection."""
        if not fbx_available:
            return
        with torch.no_grad():
            decoded = model.decode_motion_from_latent(x_tensor, should_apply_smooothing=True)
            decoded["transl"] = reference_absolute_translation.unsqueeze(0).cpu()
        for i in range(x_tensor.shape[0]):
            smpl_data = construct_smpl_data_dict(decoded['rot6d'][i], decoded['transl'][i])
            out_path = os.path.join(args.output_dir, f"{name}_{i:03d}.fbx")
            if fbx_converter.convert_npz_to_fbx(smpl_data, out_path):
                print(f"Saved FBX: {out_path}")

    # ─── 5. Build Losses ─────────────────────────────────────────────────────
    loss_similarity_s2 = MotionSimilarityLoss(
        target_motion, device, model.mean, model.std,
        joint_position_coef=args.joint_position_coef,
        hand_coef=args.hand_coef,
        joint_velocity_coef=args.joint_velocity_coef,
        hand_joint_velocity_coef=args.hand_joint_velocity_coef,
        wrist_position_coef=args.wrist_position_coef,
        lower_body_coef=args.lower_body_coef,
        joint_jerk_coef=args.joint_jerk_coef,
        lower_body_joint_jerk_coef=args.lower_body_joint_jerk_coef,
        use_weighted_loss=args.weighted_rot_loss,
        use_final_output_geometry=True,
    )
    loss_collision = MotionCollisionLoss(device, collision_threshold=args.collision_threshold)
    temporal_regularizer = MotionTemporalRegularizer(
        target_motion,
        device,
        model.mean,
        model.std,
    )

    loss_logs = {
        "total": [],
        "collision": [],
        "similarity": [],
        "temporal": [],
        "checkpoint_score": [],
    }

    # ─── 6. Run Stage 2 Optimization ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2 STANDALONE: Resolving collisions (similarity + collision)")
    print("=" * 60)

    def criterion_s2(x):
        l_col = loss_collision(x, model.mean, model.std)
        l_sim = loss_similarity_s2(x, model.mean, model.std)
        temporal = temporal_regularizer(x, model.mean, model.std)
        temporal_loss = (
            args.rotation_velocity_scale * temporal["rotation_velocity"]
            + args.upper_body_velocity_scale * temporal["upper_body_velocity"]
            + args.upper_body_rotation_scale * temporal["upper_body_rotation"]
        )
        loss = (
            args.collision_scale * l_col
            + args.similarity_scale * l_sim
            + temporal_loss
        )
        loss_logs["total"].append(loss.item())
        loss_logs["collision"].append(l_col.item())
        loss_logs["similarity"].append(l_sim.item())
        checkpoint_score = args.similarity_scale * l_sim + temporal_loss
        loss_logs["temporal"].append(temporal_loss.item())
        loss_logs["checkpoint_score"].append(checkpoint_score.item())
        return loss, {
            "col": l_col.item(),
            "sim": l_sim.item(),
            "temporal": temporal_loss.item(),
            "checkpoint_score": checkpoint_score.item(),
        }

    if args.exact_fcl_select_checkpoint and args.save_checkpoint_motions_every <= 0:
        raise ValueError(
            "--exact_fcl_select_checkpoint requires "
            "--save_checkpoint_motions_every > 0"
        )

    checkpoint_callback = None
    checkpoint_metadata_path = None
    if args.save_checkpoint_motions_every > 0:
        checkpoint_motion_dir = Path(
            args.checkpoint_motion_dir
            if args.checkpoint_motion_dir is not None
            else os.path.join(args.output_dir, "checkpoint_motions")
        )
        checkpoint_motion_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata_path = checkpoint_motion_dir / "metadata.jsonl"
        if checkpoint_metadata_path.exists():
            checkpoint_metadata_path.unlink()

        def checkpoint_callback(
            step: int,
            z: torch.Tensor,
            x: torch.Tensor,
            details: dict,
        ) -> None:
            """Save Stage 2 checkpoint tensors and public-format motion."""
            should_save = (
                step % args.save_checkpoint_motions_every == 0
                or step == args.s2_steps - 1
            )
            if not should_save:
                return
            with torch.no_grad():
                checkpoint_motion = _latent_to_motion_135(
                    x.detach(),
                    model.mean,
                    model.std,
                    target_motion[:, 132:135],
                )
            checkpoint_path = checkpoint_motion_dir / f"step_{step:04d}.npy"
            checkpoint_x_path = checkpoint_motion_dir / f"step_{step:04d}_x.pt"
            checkpoint_z_path = checkpoint_motion_dir / f"step_{step:04d}_z.pt"
            np.save(checkpoint_path, checkpoint_motion.cpu().numpy())
            torch.save(x.detach().cpu(), checkpoint_x_path)
            torch.save(z.detach().cpu(), checkpoint_z_path)
            metadata = {
                "step": int(step),
                "motion": str(checkpoint_path),
                "x": str(checkpoint_x_path),
                "z": str(checkpoint_z_path),
                "col": details.get("col"),
                "sim": details.get("sim"),
                "temporal": details.get("temporal"),
                "checkpoint_score": details.get("checkpoint_score"),
            }
            with checkpoint_metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metadata) + "\n")

    s2_options = DNOOptions(
        num_opt_steps=args.s2_steps,
        lr=args.s2_lr,
        perturb_scale=0,
        diff_penalty_scale=0,
        lr_warm_up_steps=args.s2_lr_warmup_steps,
        lr_decay_steps=args.s2_lr_decay_steps,
    )
    solver_s2 = DNOSolver(
        model_fn,
        criterion_s2,
        z_fitted,
        s2_options,
        col_threshold=args.col_return_threshold,
        checkpoint_callback=checkpoint_callback,
    )
    optimization_start = time.monotonic()
    results_s2 = solver_s2()
    optimization_seconds = time.monotonic() - optimization_start

    print(f"Stage 2 optimization runtime: {optimization_seconds:.1f} seconds")
    print(f"Last iteration col: {loss_logs['collision'][-1]:.6f}, "
          f"sim: {loss_logs['similarity'][-1]:.6f}")
    if results_s2["best_step"] >= 0:
        print(f"Selected checkpoint step: {results_s2['best_step']}")

    # ─── 7. Save Results ─────────────────────────────────────────────────────
    optimized_x = results_s2["x"]
    optimized_z = results_s2["z"]
    exact_selection_summary = None
    if args.exact_fcl_select_checkpoint:
        from poseshield.common.collision import DEFAULT_MOTION_TOPOLOGY_THRESHOLD

        exact_selection_dir = Path(
            args.exact_fcl_selection_dir
            if args.exact_fcl_selection_dir is not None
            else os.path.join(args.output_dir, "exact_fcl_checkpoint_selection")
        )
        distances_path = Path(args.exact_fcl_selection_distances)
        if not distances_path.is_absolute():
            distances_path = Path(workspace_root) / distances_path
        topology_threshold = (
            args.exact_fcl_selection_topology_threshold
            if args.exact_fcl_selection_topology_threshold is not None
            else DEFAULT_MOTION_TOPOLOGY_THRESHOLD
        )
        exact_selection_summary = _select_checkpoint_by_exact_fcl(
            metadata_path=checkpoint_metadata_path,
            output_dir=exact_selection_dir,
            distances_path=distances_path,
            device=args.exact_fcl_selection_device,
            topology_threshold=topology_threshold,
            resume=args.exact_fcl_selection_resume,
            proxy_col_threshold=args.exact_fcl_selection_proxy_col_threshold,
        )
        optimized_x = torch.load(
            exact_selection_summary["selected_x"],
            map_location=device,
        )
        optimized_z = torch.load(
            exact_selection_summary["selected_z"],
            map_location=device,
        )
        if optimized_x.dim() != 3 or optimized_z.dim() != 3:
            raise ValueError(
                "Exact-FCL checkpoint selection loaded invalid tensor shapes: "
                f"x={tuple(optimized_x.shape)}, z={tuple(optimized_z.shape)}"
            )
        print(
            "Exact-FCL checkpoint selection chose step "
            f"{exact_selection_summary['selected_step']} "
            f"({exact_selection_summary['selection_reason']}, "
            "collision_frames="
            f"{exact_selection_summary['selected_num_collision_frames']}, "
            "mean_penetration_depth="
            f"{exact_selection_summary['selected_mean_penetration_depth']:.9f})"
        )

    with torch.no_grad():
        selected_col = float(loss_collision(optimized_x, model.mean, model.std).item())
        selected_sim = float(loss_similarity_s2(optimized_x, model.mean, model.std).item())
        motion_135 = _latent_to_motion_135(
            optimized_x,
            model.mean,
            model.std,
            target_motion[:, 132:135],
        )
        optimized_unnorm = optimized_x * model.std + model.mean

    optimized_motion_np = motion_135.cpu().numpy()
    ground_truth_np = None
    if args.ground_truth is not None:
        ground_truth_np = _load_ground_truth(args.ground_truth)
    motion_metrics = compute_motion_metrics(
        optimized_motion_np,
        motion_np,
        ground_truth_np,
    )

    save_fbx(optimized_x, "final_out")
    torch.save(optimized_z.cpu(), os.path.join(args.output_dir, "optimized_z.pt"))
    np.save(os.path.join(args.output_dir, "optimized_motion.npy"), optimized_motion_np)

    summary = {
        "col": selected_col,
        "sim": selected_sim,
        **motion_metrics,
        "best_step": int(results_s2["best_step"]),
        "checkpoint_reason": results_s2["checkpoint_reason"],
        "checkpoint_col": results_s2["checkpoint_col"],
        "checkpoint_score": results_s2["checkpoint_score"],
        "wrist_position_mse": loss_similarity_s2.last_wrist_position_mse,
        "wrist_mean_distance": loss_similarity_s2.last_wrist_mean_distance,
        "optimization_seconds": optimization_seconds,
        "last_iteration_col": loss_logs["collision"][-1],
        "last_iteration_sim": loss_logs["similarity"][-1],
    }
    if exact_selection_summary is not None:
        summary["exact_fcl_checkpoint_selection"] = exact_selection_summary
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(loss_logs, f, indent=2)

    print(f"RESULT col={selected_col:.9f}")
    print(f"RESULT sim={selected_sim:.9f}")
    print(f"RESULT dyn_ratio={motion_metrics['dyn_ratio']:.9f}")
    print(f"RESULT hand_var_ratio={motion_metrics['hand_var_ratio']:.9f}")
    wrist_metrics = {
        "wrist_position_mse": loss_similarity_s2.last_wrist_position_mse,
        "wrist_mean_distance": loss_similarity_s2.last_wrist_mean_distance,
    }
    for metric_name, metric_value in wrist_metrics.items():
        rendered_value = "NA" if metric_value is None else f"{metric_value:.9f}"
        print(f"RESULT {metric_name}={rendered_value}")
    for metric_name in (
        "gt_pose_dist",
        "gt_rel_trans_dist",
        "gt_abs_trans_dist",
    ):
        metric_value = motion_metrics[metric_name]
        rendered_value = "NA" if metric_value is None else f"{metric_value:.9f}"
        print(f"RESULT {metric_name}={rendered_value}")

    # Free up memory before penetration depth check
    import gc
    del model
    del solver_s2
    del loss_similarity_s2
    del loss_collision
    del temporal_regularizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.evaluate_penetration:
        print("Evaluating exact-mesh penetration depth...")
        import smplx
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        from poseshield.common.collision import (
            DEFAULT_MOTION_TOPOLOGY_THRESHOLD,
            load_topology_distances,
            self_collision_status,
        )

        root_rotation = optimized_unnorm[0, :, 3:9]
        body_rotation = optimized_unnorm[0, :, 9:135].reshape(seq_len, 21, 6)
        global_axis_angles = matrix_to_axis_angle_torch(
            rotation_6d_to_matrix_torch(root_rotation)
        )
        body_axis_angles = matrix_to_axis_angle_torch(
            rotation_6d_to_matrix_torch(body_rotation)
        ).flatten(1)
        absolute_translation = reference_absolute_translation

        smpl_model = smplx.create(
            BODY_MODEL_PATH_, model_type='smplh', gender='neutral',
            ext='npz', use_pca=False, batch_size=seq_len
        ).to(device)
        
        with torch.no_grad():
            output = smpl_model(
                global_orient=global_axis_angles.to(device),
                body_pose=body_axis_angles.to(device),
                transl=absolute_translation.to(device),
                return_verts=True
            )
            vertices = output.vertices.cpu().numpy()
            faces = smpl_model.faces
            
        dist_path = "deps/topology_distances_30_60.npz"
        if not os.path.exists(dist_path):
            raise FileNotFoundError(f"Distances file not found: {dist_path}")
        distances = load_topology_distances(dist_path)

        collision_flags = []
        penetration_depths = []
        motion_topology_threshold = DEFAULT_MOTION_TOPOLOGY_THRESHOLD
        for i in tqdm(range(seq_len), desc="Detecting collision"):
            has_collision, penetration_depth = self_collision_status(
                vertices[i],
                faces,
                distances,
                topology_threshold=motion_topology_threshold,
            )
            collision_flags.append(bool(has_collision))
            penetration_depths.append(float(penetration_depth))

        plt.figure(figsize=(10, 5))
        plt.plot(range(seq_len), penetration_depths, marker="o", color="red", markersize=3)
        plt.xlabel("Frame")
        plt.ylabel("Penetration Depth")
        plt.title("Penetration Depth over Time")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "penetration_depth.png"))
        plt.close()

        mean_penetration = float(np.mean(penetration_depths))
        collision_frames = np.flatnonzero(np.asarray(collision_flags, dtype=bool)).tolist()
        exact_collision_free = len(collision_frames) == 0
        penetration_results = {
            "exact_collision_free": exact_collision_free,
            "num_collision_frames": len(collision_frames),
            "collision_frame_indices": collision_frames,
            "topology_threshold": motion_topology_threshold,
            "mean_penetration_depth": mean_penetration,
            "penetration_depth_seq": penetration_depths,
        }
        summary["exact_collision_free"] = exact_collision_free
        summary["num_collision_frames"] = len(collision_frames)
        summary["mean_penetration_depth"] = mean_penetration
        with open(
            os.path.join(args.output_dir, "penetration_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(penetration_results, f, indent=2)
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Exact collision free: {exact_collision_free}")
        print(f"Collision frames: {len(collision_frames)}/{seq_len}")
        print(f"Mean penetration depth: {mean_penetration:.9f}")

    print("Done.")

if __name__ == "__main__":
    main()
