#!/usr/bin/env python3
import numpy as np
import torch
from scipy.optimize import minimize

from poseshield.common.network import ResidualMLP
from poseshield.pose.utils import cost_function, cost_function_weighted, constraint_function

def optimize_slsqp(
    sample: np.ndarray,
    model: ResidualMLP,
    device: torch.device,
    max_itr: int = 200,
    threshold = 0.1,
    cost_type: str = "normal",
    tol: float = 0.03,
):
    """
    Optimize using SciPy SLSQP to minimize cost_function(x, x_ref)
    subject to:
      - constraint_function(model, x) - threshold >= 0
        (i.e., the learned collision-field score must meet the threshold)
      - 6D rotation regularization as equalities:
          ||r1|| = 1, ||r2|| = 1, dot(r1, r2) = 0
    """
    # initial point and reference
    x0 = sample.reshape(-1).astype(np.float64)
    x_ref_np = x0.copy()

    # torch helpers
    def to_torch(x_np, requires_grad=False):
        return torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=requires_grad)

    x_ref_t = to_torch(x_ref_np, requires_grad=False)

    # objective: cost_function(x, x_ref) or cost_function_weighted(x, x_ref)
    def cost_fn_np(x_np):
        x_t = to_torch(x_np, requires_grad=False)
        if cost_type == "weighted":
            val = cost_function_weighted(x_t, x_ref_t)
        else:
            val = cost_function(x_t, x_ref_t)
        return float(val.detach().cpu().item())

    def cost_fn_jac(x_np):
        x_t = to_torch(x_np, requires_grad=True)
        if cost_type == "weighted":
            val = cost_function_weighted(x_t, x_ref_t)
        else:
            val = cost_function(x_t, x_ref_t)
        grad = torch.autograd.grad(val, x_t, retain_graph=False)[0]
        return grad.detach().cpu().numpy().astype(np.float64)

    # inequality constraint: constraint_function(model, x) - threshold >= 0
    def cons_ineq_fun(x_np):
        x_t = to_torch(x_np, requires_grad=False)
        val = constraint_function(model, x_t) - threshold
        return float(val.detach().cpu().item())

    def cons_ineq_jac(x_np):
        x_t = to_torch(x_np, requires_grad=True)
        cons_val = constraint_function(model, x_t)
        grad = torch.autograd.grad(cons_val, x_t, retain_graph=False)[0]
        return (grad).detach().cpu().numpy().astype(np.float64)

    # helpers to view x as [N x 6]
    def view6(x_t):
        return x_t.view(-1, 6)

    # equality constraints for 6D rotation parametrization
    # 1) ||r1|| = 1
    # replace equalities with two-sided inequality constraints
    # ||r1|| in [1 - tol, 1 + tol]
    def ineq_r1_upper_fun(x_np):
        x_t = to_torch(x_np, requires_grad=False)
        r1 = view6(x_t)[:, :3]
        return float((1.0 + tol - r1.norm(dim=1).mean()).detach().cpu().item())

    def ineq_r1_upper_jac(x_np):
        x_t = to_torch(x_np, requires_grad=True)
        r1 = view6(x_t)[:, :3]
        val = 1.0 + tol - r1.norm(dim=1).mean()
        grad = torch.autograd.grad(val, x_t, retain_graph=False)[0]
        return grad.detach().cpu().numpy().astype(np.float64)

    def ineq_r1_lower_fun(x_np):
        x_t = to_torch(x_np, requires_grad=False)
        r1 = view6(x_t)[:, :3]
        return float((r1.norm(dim=1).mean() - (1.0 - tol)).detach().cpu().item())

    def ineq_r1_lower_jac(x_np):
        x_t = to_torch(x_np, requires_grad=True)
        r1 = view6(x_t)[:, :3]
        val = r1.norm(dim=1).mean() - (1.0 - tol)
        grad = torch.autograd.grad(val, x_t, retain_graph=False)[0]
        return grad.detach().cpu().numpy().astype(np.float64)

    # ||r2|| in [1 - tol, 1 + tol]
    def ineq_r2_upper_fun(x_np):
        x_t = to_torch(x_np, requires_grad=False)
        r2 = view6(x_t)[:, 3:]
        return float((1.0 + tol - r2.norm(dim=1).mean()).detach().cpu().item())

    def ineq_r2_upper_jac(x_np):
        x_t = to_torch(x_np, requires_grad=True)
        r2 = view6(x_t)[:, 3:]
        val = 1.0 + tol - r2.norm(dim=1).mean()
        grad = torch.autograd.grad(val, x_t, retain_graph=False)[0]
        return grad.detach().cpu().numpy().astype(np.float64)

    def ineq_r2_lower_fun(x_np):
        x_t = to_torch(x_np, requires_grad=False)
        r2 = view6(x_t)[:, 3:]
        return float((r2.norm(dim=1).mean() - (1.0 - tol)).detach().cpu().item())

    def ineq_r2_lower_jac(x_np):
        x_t = to_torch(x_np, requires_grad=True)
        r2 = view6(x_t)[:, 3:]
        val = r2.norm(dim=1).mean() - (1.0 - tol)
        grad = torch.autograd.grad(val, x_t, retain_graph=False)[0]
        return grad.detach().cpu().numpy().astype(np.float64)

    # |dot(r1, r2)| <= tol (two inequalities)
    def ineq_orth_upper_fun(x_np):
        x_t = to_torch(x_np, requires_grad=False)
        r1, r2 = view6(x_t).split(3, dim=1)
        dot_mean = (r1 * r2).sum(dim=1).mean()
        return float((tol - dot_mean).detach().cpu().item())

    def ineq_orth_upper_jac(x_np):
        x_t = to_torch(x_np, requires_grad=True)
        r1, r2 = view6(x_t).split(3, dim=1)
        val = tol - (r1 * r2).sum(dim=1).mean()
        grad = torch.autograd.grad(val, x_t, retain_graph=False)[0]
        return grad.detach().cpu().numpy().astype(np.float64)

    def ineq_orth_lower_fun(x_np):
        x_t = to_torch(x_np, requires_grad=False)
        r1, r2 = view6(x_t).split(3, dim=1)
        dot_mean = (r1 * r2).sum(dim=1).mean()
        return float((tol + dot_mean).detach().cpu().item())

    def ineq_orth_lower_jac(x_np):
        x_t = to_torch(x_np, requires_grad=True)
        r1, r2 = view6(x_t).split(3, dim=1)
        val = tol + (r1 * r2).sum(dim=1).mean()
        grad = torch.autograd.grad(val, x_t, retain_graph=False)[0]
        return grad.detach().cpu().numpy().astype(np.float64)

    # collect constraints (all inequalities)
    constraints = [
        {'type': 'ineq', 'fun': cons_ineq_fun,       'jac': cons_ineq_jac},
        {'type': 'ineq', 'fun': ineq_r1_upper_fun,   'jac': ineq_r1_upper_jac},
        {'type': 'ineq', 'fun': ineq_r1_lower_fun,   'jac': ineq_r1_lower_jac},
        {'type': 'ineq', 'fun': ineq_r2_upper_fun,   'jac': ineq_r2_upper_jac},
        {'type': 'ineq', 'fun': ineq_r2_lower_fun,   'jac': ineq_r2_lower_jac},
        {'type': 'ineq', 'fun': ineq_orth_upper_fun, 'jac': ineq_orth_upper_jac},
        {'type': 'ineq', 'fun': ineq_orth_lower_fun, 'jac': ineq_orth_lower_jac},
    ]

    # history via callback
    loss_history, cons_history = [], []

    def callback(xk):
        try:
            loss_history.append(cost_fn_np(xk))
            cons_history.append(constraint_function(model, to_torch(xk)).detach().cpu().item())
        except Exception:
            pass

    res = minimize(
        cost_fn_np,
        x0,
        method='SLSQP',
        jac=cost_fn_jac,
        constraints=constraints,
        bounds=None,  # keep unbounded; add bounds if desired
        options={'maxiter': max_itr, 'ftol': 1e-6, 'disp': False},
        callback=callback
    )

    if not res.success:
        print(f"SLSQP solver_success=False")
        print(f"SLSQP message: {res.message}")
        print(f"SLSQP status: {res.status}")
        print(f"SLSQP iterations: {res.nit}")
        print(f"SLSQP final objective: {res.fun}")

    x_opt = res.x.astype(np.float32)
    print(f"Final parameter delta: {np.linalg.norm(x_opt - x0):.6f}")
    # ensure at least one entry in histories (final values)
    if len(loss_history) == 0:
        loss_history.append(cost_fn_np(x_opt))
    if len(cons_history) == 0:
        cons_history.append(constraint_function(model, to_torch(x_opt)).detach().cpu().item())

    return x_opt, loss_history, cons_history, bool(res.success), str(res.message)
