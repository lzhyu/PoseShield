"""PoseShield network: ResidualMLP and supporting rotation utilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_6d_rotation_valid_tensor(rot_6d: torch.Tensor) -> torch.Tensor:
    """Orthonormalize a 6D rotation vector via Gram-Schmidt.

    Given two raw 3D vectors (the first two columns of a rotation matrix),
    produce a valid orthonormal pair.

    Args:
        rot_6d: shape (6,) or (N, 6).

    Returns:
        Orthonormalized 6D rotation, same shape as input.
    """
    single_input = False
    if rot_6d.ndim == 1:
        rot_6d = rot_6d.unsqueeze(0)
        single_input = True

    x_raw = rot_6d[:, :3]
    y_raw = rot_6d[:, 3:]

    x_norm = x_raw / x_raw.norm(dim=1, keepdim=True)

    dot = (x_norm * y_raw).sum(dim=1, keepdim=True)
    y_perp = y_raw - dot * x_norm
    y_norm = y_perp / y_perp.norm(dim=1, keepdim=True)

    valid_6d = torch.cat([x_norm, y_norm], dim=1)

    if single_input:
        valid_6d = valid_6d.squeeze(0)
    return valid_6d

def get_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")

class ResidualMLP(nn.Module):
    """Residual MLP that maps 21×6 joint rotations to a scalar field value.

    Input is projected to valid 6D rotations before processing.
    """

    def __init__(self, in_dim: int = 126, hidden_dim: int = 64,
                 num_layers: int = 6, activation: str = 'relu'):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.act = get_activation(activation)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 126) flattened joint rotations.

        Returns:
            Scalar field value (B, 1).
        """
        assert len(x.shape) == 2, x.shape
        assert x.shape[-1] == 126, f"Expected input dim 126, got {x.shape[-1]}"
        assert not torch.isnan(x).any(), "x contains NaN values in ResidualMLP.forward"
        bs = x.shape[0]

        # Reshape to (B*21, 6) to normalize each rotation
        x_reshaped = x.reshape(-1, 6)
        x_valid = make_6d_rotation_valid_tensor(x_reshaped)
        x = x_valid.reshape(bs, 126)

        x = self.act(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x)) + x  # residual
        return self.output_layer(x)
