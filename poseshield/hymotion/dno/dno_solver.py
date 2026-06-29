import math
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from torchdiffeq import odeint as odeint_direct
from torchdiffeq import odeint_adjoint as odeint_adjoint

@dataclass
class DNOOptions:
    num_opt_steps: int = field(default=500, metadata={"help": "Number of optimization steps"})
    lr: float = field(default=5e-2, metadata={"help": "Learning rate"})
    perturb_scale: float = field(default=0, metadata={"help": "scale of the noise perturbation"})
    diff_penalty_scale: float = field(default=0, metadata={"help": "penalty for difference from initial z"})
    lr_warm_up_steps: int = field(default=50, metadata={"help": "Number of warm-up steps"})
    lr_decay_steps: int = field(default=None, metadata={"help": "Number of decay steps"})
    
    def __post_init__(self):
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.num_opt_steps

class DNOSolver:
    """
    Optimization wrapper for Flow Matching models.
    """
    def __init__(
        self,
        model_fn, # Function that takes z and returns final motion x
        criterion, # Loss function
        start_z,
        conf: DNOOptions,
        col_threshold: float = 1e-4,
    ):
        self.model_fn = model_fn
        self.criterion = criterion
        self.start_z = start_z.detach()
        self.conf = conf
        self.col_threshold = col_threshold
        
        self.current_z = self.start_z.clone().requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.current_z], lr=conf.lr)
        
        self.lr_scheduler = []
        if conf.lr_warm_up_steps > 0:
            self.lr_scheduler.append(lambda step: self.warmup_scheduler(step, conf.lr_warm_up_steps))
        
        self.lr_scheduler.append(
            lambda step: self.cosine_decay_scheduler(
                step, conf.lr_decay_steps, conf.num_opt_steps, decay_first=False
            )
        )
        
        self.step_count = 0
        self.hist = []
        
        # Feasible checkpoint with the lowest non-collision fidelity score.
        self.best_z = None
        self.best_x = None
        self.best_step = -1
        self.best_col = math.inf
        self.best_score = math.inf
        self.early_stopped = False

        # Lowest-collision fallback when no checkpoint reaches the strict threshold.
        self.fallback_z = None
        self.fallback_x = None
        self.fallback_step = -1
        self.fallback_col = math.inf
        self.fallback_score = math.inf

    def _consider_checkpoint(
        self,
        x: torch.Tensor,
        col_value: Optional[float],
        checkpoint_score: float,
        allow_feasible: bool,
    ) -> None:
        """Track the best feasible checkpoint and the minimum-collision fallback."""
        if col_value is None:
            return
        col_value = float(col_value)
        checkpoint_score = float(checkpoint_score)
        if not math.isfinite(col_value) or not math.isfinite(checkpoint_score):
            return

        improves_collision = col_value < self.fallback_col
        ties_collision_with_lower_fidelity = (
            math.isclose(col_value, self.fallback_col)
            and checkpoint_score < self.fallback_score
        )
        if improves_collision or ties_collision_with_lower_fidelity:
            self.fallback_z = self.current_z.detach().clone()
            self.fallback_x = x.detach().clone()
            self.fallback_step = self.step_count
            self.fallback_col = col_value
            self.fallback_score = checkpoint_score

        is_feasible = allow_feasible and col_value <= self.col_threshold
        improves_fidelity = checkpoint_score < self.best_score
        ties_fidelity_with_lower_col = (
            math.isclose(checkpoint_score, self.best_score) and col_value < self.best_col
        )
        if is_feasible and (improves_fidelity or ties_fidelity_with_lower_col):
            self.best_z = self.current_z.detach().clone()
            self.best_x = x.detach().clone()
            self.best_step = self.step_count
            self.best_col = col_value
            self.best_score = checkpoint_score

    def warmup_scheduler(self, step, warmup_steps):
        if step < warmup_steps:
            return step / warmup_steps
        return 1

    def cosine_decay_scheduler(self, step, decay_steps, total_steps, decay_first=True):
        if step >= total_steps:
            return 0
        if decay_first:
            if step >= decay_steps:
                return 0
            return (math.cos((step) / decay_steps * math.pi) + 1) / 2
        else:
            if step < total_steps - decay_steps:
                return 1
            return (
                math.cos((step - (total_steps - decay_steps)) / decay_steps * math.pi) + 1
            ) / 2

    def __call__(self, num_steps: int = None):
        if num_steps is None:
            num_steps = self.conf.num_opt_steps
            
        batch_size = self.start_z.shape[0]
        
        with tqdm(range(num_steps)) as prog:
            for i in prog:
                info = {"step": [self.step_count] * batch_size}
                
                # LR scheduling
                lr_frac = 1
                if len(self.lr_scheduler) > 0:
                    for scheduler in self.lr_scheduler:
                        lr_frac *= scheduler(self.step_count)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.conf.lr * lr_frac
                info["lr"] = [self.conf.lr * lr_frac] * batch_size
                
                # Forward pass
                # model_fn runs the ODE integration from current_z
                x = self.model_fn(self.current_z)
                
                # Loss
                crit_out = self.criterion(x)
                if isinstance(crit_out, tuple):
                    loss, details = crit_out
                else:
                    loss = crit_out
                    details = {}
                    
                if details.get("stop_early", False):
                    print(f"Early stopping triggered at step {self.step_count} because stop_early condition was met.")
                    self.best_z = self.current_z.detach().clone()
                    self.best_x = x.detach().clone()
                    self.best_step = self.step_count
                    self.best_col = float(details.get("col", math.inf))
                    self.best_score = float(
                        details.get("checkpoint_score", loss.detach().mean().item())
                    )
                    self.early_stopped = True
                    break
                    
                if loss.shape == (batch_size,):
                     loss_cls = loss.clone().detach().cpu()
                     loss = loss.sum()
                else:
                     loss_cls = [loss.item()] * batch_size
                
                info["loss"] = loss_cls

                # Track a consistent (z, x) checkpoint before mutating current_z.
                # Values in `details` were computed from this exact x.
                col_val = details.get("col", None)
                warmup_done = self.step_count >= self.conf.num_opt_steps // 2
                checkpoint_score = details.get("checkpoint_score", loss.detach().item())
                self._consider_checkpoint(
                    x,
                    col_val,
                    checkpoint_score,
                    allow_feasible=warmup_done,
                )
                
                # Diff Penalty
                if self.conf.diff_penalty_scale > 0:
                    # Difference penalty (stay close to original trajectory)
                    # z shape: [B, L, D] -> 3 dims. Norm over [1, 2]
                    loss_diff = (self.current_z - self.start_z).norm(p=2, dim=[1, 2])
                    loss += self.conf.diff_penalty_scale * loss_diff.sum()
                    info["loss_diff"] = loss_diff.detach().cpu()
                else:
                    info["loss_diff"] = [0] * batch_size
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping/norm
                if self.current_z.grad is not None:
                    # Normalize gradients to unit sphere? DNO original code does this.
                    grad_norm = self.current_z.grad.norm(p=2, keepdim=True)
                    eps = 1e-8
                    self.current_z.grad.data /= (grad_norm + eps)
                
                self.optimizer.step()
                
                # Noise perturbation
                if self.conf.perturb_scale > 0:
                    noise_frac = lr_frac
                    noise = torch.randn_like(self.current_z)
                    self.current_z.data += noise * self.conf.perturb_scale * noise_frac
                
                # Log z_norm
                z_norm = self.current_z.detach().norm(p=2).item()
                info["z_norm"] = [z_norm] * batch_size
                
                self.hist.append(info)
                self.step_count += 1
                postfix_dict = {"loss": f"{loss.item() / batch_size:.4f}"}
                for k, v in details.items():
                    postfix_dict[k] = f"{v:.4f}" if isinstance(v, float) else v
                prog.set_postfix(postfix_dict)
                
        checkpoint_reason = "final_without_collision_metric"
        checkpoint_col = None
        checkpoint_score = None
        if self.best_z is not None:
            out_z = self.best_z
            out_x = self.best_x
            checkpoint_col = self.best_col if math.isfinite(self.best_col) else None
            checkpoint_score = self.best_score
            if self.early_stopped:
                print(f"Returning explicit early-stop checkpoint from step {self.best_step}.")
                checkpoint_reason = "explicit_early_stop"
            else:
                print(
                    "Returning lowest-fidelity feasible checkpoint from step "
                    f"{self.best_step} (col={self.best_col:.9f} <= {self.col_threshold}, "
                    f"score={self.best_score:.9f})"
                )
                checkpoint_reason = "lowest_fidelity_within_collision_threshold"
        elif self.fallback_z is not None:
            print(
                f"No checkpoint reached col <= {self.col_threshold}; returning "
                f"minimum-collision checkpoint from step {self.fallback_step} "
                f"(col={self.fallback_col:.9f}, score={self.fallback_score:.9f})."
            )
            out_z = self.fallback_z
            out_x = self.fallback_x
            self.best_step = self.fallback_step
            checkpoint_reason = "minimum_collision_fallback"
            checkpoint_col = self.fallback_col
            checkpoint_score = self.fallback_score
        else:
            print("No collision metric was reported; returning the final step.")
            out_z = self.current_z.detach()
            # current_z has already received the final optimizer update, so the
            # last loop-local x corresponds to the previous z and cannot be reused.
            out_x = self.model_fn(self.current_z).detach()
        
        return {
            "z": out_z,
            "x": out_x,
            "history": self.hist,
            "best_step": self.best_step,
            "checkpoint_reason": checkpoint_reason,
            "checkpoint_col": checkpoint_col,
            "checkpoint_score": checkpoint_score,
        }

def ode_loop_with_gradient(
    model,
    y0,
    t_span,
    model_kwargs,
    noise_scheduler_cfg,
    cfg_scale=1.0,
    uncond_model_kwargs=None,
    use_adjoint=True
):
    """
    Differentiable ODE loop with optional CFG.
    """
    
    # Extract conditioning from model_kwargs
    ctxt_input = model_kwargs.get("ctxt_input")
    vtxt_input = model_kwargs.get("vtxt_input") 
    x_mask_temporal = model_kwargs.get("x_mask_temporal")
    ctxt_mask_temporal = model_kwargs.get("ctxt_mask_temporal")
    
    # Prepare Unconditional inputs if CFG enabled
    do_cfg = cfg_scale > 1.0 and uncond_model_kwargs is not None
    if do_cfg:
        u_ctxt_input = uncond_model_kwargs.get("ctxt_input")
        u_vtxt_input = uncond_model_kwargs.get("vtxt_input")
        u_ctxt_mask_temporal = uncond_model_kwargs.get("ctxt_mask_temporal")
        
        # Concatenate conditioning [Uncond, Cond]
        # Assuming batch dimension is 0
        cat_ctxt_input = torch.cat([u_ctxt_input, ctxt_input], dim=0)
        cat_vtxt_input = torch.cat([u_vtxt_input, vtxt_input], dim=0)
        cat_ctxt_mask_temporal = torch.cat([u_ctxt_mask_temporal, ctxt_mask_temporal], dim=0)
        # x_mask_temporal needs to be doubled too
        cat_x_mask_temporal = torch.cat([x_mask_temporal, x_mask_temporal], dim=0)
    
    class ODEWrapper(nn.Module):
        def __init__(self, model, do_cfg, ctxt_input, vtxt_input, x_mask_temporal, ctxt_mask_temporal,
                     cat_ctxt_input=None, cat_vtxt_input=None, cat_x_mask_temporal=None, cat_ctxt_mask_temporal=None, cfg_scale=1.0):
            super().__init__()
            self.model = model
            self.do_cfg = do_cfg
            self.ctxt_input = ctxt_input
            self.vtxt_input = vtxt_input
            self.x_mask_temporal = x_mask_temporal
            self.ctxt_mask_temporal = ctxt_mask_temporal
            
            self.cat_ctxt_input = cat_ctxt_input
            self.cat_vtxt_input = cat_vtxt_input
            self.cat_x_mask_temporal = cat_x_mask_temporal
            self.cat_ctxt_mask_temporal = cat_ctxt_mask_temporal
            self.cfg_scale = cfg_scale
 
        def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            timesteps = t
            if t.ndim == 0:
                timesteps = t.expand(x.shape[0])
                
            if self.do_cfg:
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([timesteps, timesteps], dim=0)
                
                x_pred = self.model(
                    x=x_in,
                    ctxt_input=self.cat_ctxt_input,
                    vtxt_input=self.cat_vtxt_input,
                    timesteps=t_in,
                    x_mask_temporal=self.cat_x_mask_temporal,
                    ctxt_mask_temporal=self.cat_ctxt_mask_temporal
                )
                
                x_pred_uncond, x_pred_cond = x_pred.chunk(2, dim=0)
                return x_pred_uncond + self.cfg_scale * (x_pred_cond - x_pred_uncond)
                
            else:
                x_pred = self.model(
                    x=x,
                    ctxt_input=self.ctxt_input,
                    vtxt_input=self.vtxt_input,
                    timesteps=timesteps,
                    x_mask_temporal=self.x_mask_temporal,
                    ctxt_mask_temporal=self.ctxt_mask_temporal
                )
                return x_pred
 
    if do_cfg:
        fn = ODEWrapper(model, do_cfg, ctxt_input, vtxt_input, x_mask_temporal, ctxt_mask_temporal,
                        cat_ctxt_input, cat_vtxt_input, cat_x_mask_temporal, cat_ctxt_mask_temporal, cfg_scale)
    else:
        fn = ODEWrapper(model, do_cfg, ctxt_input, vtxt_input, x_mask_temporal, ctxt_mask_temporal)
    
    # Run ODE
    ode_solver_fn = odeint_adjoint if use_adjoint else odeint_direct
    trajectory = ode_solver_fn(fn, y0, t_span, **noise_scheduler_cfg)
    
    # Return the final state
    return trajectory[-1]
