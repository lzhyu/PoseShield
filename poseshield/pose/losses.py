import torch
import torch.nn as nn
import logging

def consistency_score_strict(g_vals, y_vals):
    """
    Calculate consistency score (0~1):
      - When label y=1, requires g_vals>0 to be correct
      - When label y=-1, requires g_vals<0 to be correct
    Other cases (e.g. g_vals==0) are considered incorrect.

    Args:
      g_vals: torch.Tensor, shape [N] or [N,1]
              Model output values for N samples
      y_vals: torch.Tensor, shape [N] or [N,1]
              Binary label elements (1 or -1)

    Returns:
      Consistency score, float, range [0,1]
    """
    # Flatten shapes for logical check
    g_vals = g_vals.view(-1)
    y_vals = y_vals.view(-1)

    # "Correct" conditions:
    # 1) y=1 and g>0
    # 2) y=-1 and g<0
    correct_mask = ((y_vals == 1) & (g_vals > 0)) | ((y_vals == -1) & (g_vals < 0))
    
    # correct_mask is bool, compute accuracy
    score = correct_mask.float().mean().item()
    return score

def poseshield_loss(model, x, iota, dt=0.01, grad_loss_weight=1.0, td_loss_weight=1.0):
    """
    Compute the PoseShield training loss, which optimizes the surrogate collision constraint g(theta):
    
        L_PoseShield = mean over i of [ L_grad_i + L_sign_i ] (if Eikonal mode)
                  or [ L_TD_i + L_sign_i ] (if Temporal-Difference mode)
    
    where:
      L_grad_i = | ||∇g(x_i)|| - 1 |
      L_sign_i = - min( g(x_i)*iota_i, 0 )
      L_TD_i   = | g(x_i + v_i dt) - g(x_i - v_i dt) - 2 dt |
      v_i      = ( ∇g(x_i) ) / ||∇g(x_i)||,
      dt       = small timestep Delta t (default 0.01),
      iota_i   = binary collision label (+1 if collision-free, -1 if colliding).

    Arguments:
      - model:            a callable neural field g(·) : R^d -> R
      - x:                [batch_size, dim]  (the pose parameter samples theta)
      - iota:             [batch_size] binary/±1 feasibility labels (used in sign constraint)
      - dt:               scalar timestep Delta t for the symmetric TD-loss
      - grad_loss_weight: weight coefficient for Eikonal regularization (L_grad)
      - td_loss_weight:   weight coefficient for Temporal-Difference loss (L_TD)
    
    Returns:
      A scalar PyTorch loss tensor.
    """
    # Make sure x requires gradients for ∂g/∂x:
    x = x.requires_grad_(True)
    iota_signed = iota.unsqueeze(-1)  # shape: [batch_size, 1]

    # Forward pass g(x)
    g = model(x)  # shape: [batch_size, 1]

    # 2) Sign constraint term: L_sign = - min(g(x)*iota, 0)
    product = g * iota_signed
    l_sign_i = -torch.minimum(product, torch.zeros_like(product))

    if grad_loss_weight == 0.0 and td_loss_weight == 0.0:
        return l_sign_i.mean()

    # Compute ∇g(x):
    grad = torch.autograd.grad(
        outputs=g,
        inputs=x,
        grad_outputs=torch.ones_like(g),  # must match shape of g
        create_graph=True,
        retain_graph=True
    )[0]  # shape: [batch_size, dim]

    # 1) Eikonal / Gradient loss term: L_grad = | ||∇g|| - 1 |
    grad_norm = torch.norm(grad, dim=1, keepdim=True)   # ||∇g||
    l_grad_i = torch.abs(grad_norm - 1.0)               # | ||∇g|| - 1 |

    # 3) Symmetric Temporal-Difference (TD) loss term: L_TD = | g(x + v*dt) - g(x - v*dt) - 2*dt |
    # Normalized velocity: v = grad / (||grad|| + eps)
    eps = 1e-10
    v = grad / (grad_norm + eps)  # shape: [batch_size, dim]

    # Evaluate g at x+v*dt and x-v*dt:
    x_plus = x + v * dt
    x_minus = x - v * dt

    g_plus = model(x_plus)
    g_minus = model(x_minus)

    # The finite-timestep TD penalty: | g(x+v dt) - g(x-v dt) - 2 dt |
    l_td_i = torch.abs(g_plus - g_minus - 2.0 * dt)

    # Combine losses for each sample
    sample_loss = grad_loss_weight * l_grad_i + l_sign_i + td_loss_weight * l_td_i

    # Final mean across the batch
    return sample_loss.mean()
