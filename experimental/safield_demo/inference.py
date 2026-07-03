"""Shape-aware SLSQP collision resolution for SAField (Demo version).

Given a colliding pose θ and fixed shape β, optimizes θ to minimize deviation
from the original pose while satisfying g(θ; β) >= threshold.
"""
import numpy as np
import torch
from scipy.optimize import minimize
from network import SAFieldNetwork

# ── SMPLH kinematic-chain weights ─────────────────────────────────────────────
_SUBTREE_SIZES = [
    4,   4,  13,   3,  3,  12,  2,  2,  11,   1,  1,   2,   4,   4,  1,   3,   3,  2,  2,  1,  1
]
_w21 = np.array(_SUBTREE_SIZES, dtype=np.float32)
_w21 /= _w21.sum()
SMPLH_POSE_WEIGHTS = torch.from_numpy(_w21)


def _to_torch(x_np: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    """Convert numpy array to torch tensor on device."""
    return torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=requires_grad)


def optimize_pose(
    theta_init: np.ndarray,
    beta: np.ndarray,
    model: SAFieldNetwork,
    device: torch.device,
    max_itr: int = 100,
    threshold: float = 0.1,
) -> tuple:
    """Optimize pose via SLSQP to resolve collision predicted by SAField.

    Args:
        theta_init: (126,) initial 21×6 pose.
        beta: (10,) shape parameters (fixed).
        model: trained SAFieldNetwork.
        device: torch device.
        max_itr: maximum SLSQP iterations.
        threshold: target margin g(θ, β) >= threshold.

    Returns:
        theta_opt: (126,) optimized pose.
        success: bool indicating SLSQP convergence.
    """
    x0 = theta_init.astype(np.float64)
    beta_t = _to_torch(beta, device)
    x_ref_t = _to_torch(theta_init, device)
    weights_t = SMPLH_POSE_WEIGHTS.to(device)

    # --- Objective: Kinematic Weighted Distance to initial pose ---
    def cost_fn(x_np: np.ndarray) -> float:
        x_t = _to_torch(x_np, device)
        diff_6d = (x_t - x_ref_t).view(-1, 6)
        dist_per_joint = torch.sqrt(torch.sum(diff_6d**2, dim=-1) + 1e-8)
        val = torch.sum(weights_t * dist_per_joint)
        return float(val.item())

    def cost_jac(x_np: np.ndarray) -> np.ndarray:
        x_t = _to_torch(x_np, device, requires_grad=True)
        diff_6d = (x_t - x_ref_t).view(-1, 6)
        dist_per_joint = torch.sqrt(torch.sum(diff_6d**2, dim=-1) + 1e-8)
        val = torch.sum(weights_t * dist_per_joint)
        grad = torch.autograd.grad(val, x_t)[0]
        return grad.detach().cpu().numpy().astype(np.float64)

    # --- Neural field constraint: g(θ, β) >= threshold ---
    def cons_fn(x_np: np.ndarray) -> float:
        x_t = _to_torch(x_np, device).unsqueeze(0)
        beta_b = beta_t.unsqueeze(0)
        val = model(x_t, beta_b)[0, 0] - threshold
        return float(val.detach().cpu().item())

    def cons_jac(x_np: np.ndarray) -> np.ndarray:
        x_t = _to_torch(x_np, device, requires_grad=True).unsqueeze(0)
        beta_b = beta_t.unsqueeze(0)
        g_val = model(x_t, beta_b)[0, 0]
        grad = torch.autograd.grad(g_val, x_t)[0]
        return grad.squeeze(0).detach().cpu().numpy().astype(np.float64)

    # --- 6D rotation regularization constraints ---
    tol = 0.03

    def _view6(x_t: torch.Tensor) -> torch.Tensor:
        return x_t.view(-1, 6)

    def _make_norm_constraint(col_slice: slice, bound_type: str) -> dict:
        """Create upper/lower norm constraint for r1 or r2 columns."""
        def fn(x_np: np.ndarray) -> float:
            x_t = _to_torch(x_np, device)
            norms = _view6(x_t)[:, col_slice].norm(dim=1)
            if bound_type == "upper":
                return float((1.0 + tol - norms.mean()).detach().cpu().item())
            return float((norms.mean() - (1.0 - tol)).detach().cpu().item())

        def jac(x_np: np.ndarray) -> np.ndarray:
            x_t = _to_torch(x_np, device, requires_grad=True)
            norms = _view6(x_t)[:, col_slice].norm(dim=1)
            if bound_type == "upper":
                val = 1.0 + tol - norms.mean()
            else:
                val = norms.mean() - (1.0 - tol)
            g = torch.autograd.grad(val, x_t)[0]
            return g.detach().cpu().numpy().astype(np.float64)

        return {"type": "ineq", "fun": fn, "jac": jac}

    def _make_orth_constraint(bound_type: str) -> dict:
        """Create orthogonality constraint |dot(r1, r2)| <= tol."""
        def fn(x_np: np.ndarray) -> float:
            x_t = _to_torch(x_np, device)
            r1, r2 = _view6(x_t).split(3, dim=1)
            dot_mean = (r1 * r2).sum(dim=1).mean()
            if bound_type == "upper":
                return float((tol - dot_mean).detach().cpu().item())
            return float((tol + dot_mean).detach().cpu().item())

        def jac(x_np: np.ndarray) -> np.ndarray:
            x_t = _to_torch(x_np, device, requires_grad=True)
            r1, r2 = _view6(x_t).split(3, dim=1)
            dot_mean = (r1 * r2).sum(dim=1).mean()
            if bound_type == "upper":
                val = tol - dot_mean
            else:
                val = tol + dot_mean
            g = torch.autograd.grad(val, x_t)[0]
            return g.detach().cpu().numpy().astype(np.float64)

        return {"type": "ineq", "fun": fn, "jac": jac}

    constraints = [
        {"type": "ineq", "fun": cons_fn, "jac": cons_jac},
        _make_norm_constraint(slice(0, 3), "upper"),
        _make_norm_constraint(slice(0, 3), "lower"),
        _make_norm_constraint(slice(3, 6), "upper"),
        _make_norm_constraint(slice(3, 6), "lower"),
        _make_orth_constraint("upper"),
        _make_orth_constraint("lower"),
    ]

    res = minimize(
        cost_fn,
        x0,
        method="SLSQP",
        jac=cost_jac,
        constraints=constraints,
        options={"maxiter": max_itr, "ftol": 1e-6, "disp": False},
    )

    return res.x.astype(np.float32), bool(res.success)
