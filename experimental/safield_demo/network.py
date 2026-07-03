"""Shape-Aware Collision Field network architecture for the demo.

Implements the basis-based model: g(θ; β) = g_0(θ) + A(β)^T Φ(θ)
"""
import torch
import torch.nn as nn


def make_6d_rotation_valid_tensor(rot_6d: torch.Tensor) -> torch.Tensor:
    """Orthonormalize 6D rotation representation (Gram-Schmidt)."""
    single_input = False
    if rot_6d.ndim == 1:
        rot_6d = rot_6d.unsqueeze(0)
        single_input = True

    x_raw = rot_6d[:, :3]
    y_raw = rot_6d[:, 3:]

    # Normalize x
    x_norm = x_raw / (x_raw.norm(dim=1, keepdim=True) + 1e-8)

    # Remove component of y in direction x
    dot = (x_norm * y_raw).sum(dim=1, keepdim=True)
    proj = dot * x_norm
    y_perp = y_raw - proj

    # Normalize y
    y_norm = y_perp / (y_perp.norm(dim=1, keepdim=True) + 1e-8)

    valid_6d = torch.cat([x_norm, y_norm], dim=1)

    if single_input:
        valid_6d = valid_6d.squeeze(0)
    return valid_6d


class ResidualSoftplusMLP(nn.Module):
    """Residual MLP block with Softplus activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, num_layers: int = 12):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.act = nn.Softplus()
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x)) + x
        return self.output_layer(x)


class SAFieldNetwork(nn.Module):
    """Basis-based architecture: g(θ; β) = g_0(θ) + A(β)^T Φ(θ)."""

    def __init__(
        self,
        theta_dim: int = 126,
        beta_dim: int = 10,
        hidden_dim: int = 512,
        K: int = 8,
        num_layers_g0: int = 12,
        num_layers_phi: int = 6,
        num_layers_shape: int = 4,
    ):
        super().__init__()
        self.K = K

        # g_0(θ): mean-shape field, maps θ → scalar
        self.g0_field = ResidualSoftplusMLP(
            in_dim=theta_dim, out_dim=1,
            hidden_dim=hidden_dim, num_layers=num_layers_g0,
        )

        # Φ(θ): K basis fields, each maps θ → scalar
        self.phi_fields = nn.ModuleList([
            ResidualSoftplusMLP(
                in_dim=theta_dim, out_dim=1,
                hidden_dim=hidden_dim, num_layers=num_layers_phi,
            )
            for _ in range(K)
        ])

        # A(β): shape encoder, maps β → K coefficients
        self.shape_encoder = ResidualSoftplusMLP(
            in_dim=beta_dim, out_dim=K,
            hidden_dim=hidden_dim, num_layers=num_layers_shape,
        )

    def forward(self, theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Compute field value g(θ; β)."""
        assert theta.shape[-1] == 126, f"Expected theta dim 126, got {theta.shape[-1]}"

        # Orthonormalize to ensure valid SO(3) representation
        theta_valid = make_6d_rotation_valid_tensor(
            theta.reshape(-1, 21, 6).reshape(-1, 6)
        ).reshape(-1, 21 * 6)

        # Mean shape field: (B, 1)
        g0 = self.g0_field(theta_valid)

        # Basis fields: (B, K)
        phi = torch.cat([field(theta_valid) for field in self.phi_fields], dim=-1)

        # Shape coefficients: (B, K)
        A = self.shape_encoder(beta)

        # Dot product A(β)^T Φ(θ): (B, 1)
        basis_term = (A * phi).sum(dim=-1, keepdim=True)

        return g0 + basis_term
