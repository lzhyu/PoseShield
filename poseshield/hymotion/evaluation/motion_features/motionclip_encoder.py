"""Pretrained motion encoder used by the MFD metric.

Third-party architecture: the network definitions here (`MotionEncoder`,
`_PositionalEncoding`) are transcribed from the `MotionCLIP` evaluator in
MARDM (github.com/neu-vi/MARDM, `utils/evaluators.py`), so that the pretrained
weights published for that evaluator load into them. The layer shapes and
hyperparameters below are fixed by those weights and must not be changed. Only
the motion branch is reproduced -- see the note below.

MFD compares a resolved motion against its input in a learned motion feature
space. The encoder is the Transformer-based motion branch of a motion/text
contrastive model: a stack of 8 Transformer encoder layers (d=512, 8 heads)
reading the 67-D HumanML3D feature (see `humanml3d_features.py`), with a
learnable CLS query token whose output is projected to a 512-D embedding and
L2-normalized.

This module builds **only the motion branch**. The upstream checkpoint also
contains a frozen text tower, which the released code used to instantiate just
so a strict `load_state_dict` would find every key. That tower is never
evaluated when encoding a motion, and the weights it was initialized with are
overwritten by the checkpoint anyway, so here the checkpoint's `motion_encoder.*`
entries are loaded on their own. Verified to produce bitwise-identical
embeddings while removing a large third-party download from the setup path.

Checkpoint
----------
The encoder weights are **not distributed with PoseShield**. Download them from
the original project and place the file at the path named in
`MOTIONCLIP_CHECKPOINT` below; see the evaluation section of the top-level
README. Without it, MFD is unavailable and the evaluator falls back to reporting
the other three metrics (or pass `--no_mfd` to skip it explicitly).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from poseshield.common.motion_fk import motion_to_joints

from .humanml3d_features import NUM_JOINTS, joints_to_67d

PROJECT_ROOT = Path(__file__).resolve().parents[4]
_EVALUATOR_ROOT = PROJECT_ROOT / "deps" / "motion_feature_encoder"

MOTIONCLIP_CHECKPOINT = _EVALUATOR_ROOT / "text_mot_match_clip" / "model" / "finest.tar"
FEATURE_MEAN = _EVALUATOR_ROOT / "eval_mean.npy"
FEATURE_STD = _EVALUATOR_ROOT / "eval_std.npy"

DIM_POSE = 67
LATENT_DIM = 512
FF_SIZE = 1024
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.2
ACTIVATION = "gelu"


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Build a `[B, max_len]` boolean mask that is True for valid frames."""
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)


class _PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class MotionEncoder(nn.Module):
    """Transformer motion encoder producing one embedding per sequence.

    A learnable CLS query token is prepended to the embedded motion frames; the
    transformer's output at that position is layer-normed and projected to the
    512-D joint embedding space.
    """

    def __init__(
        self,
        in_dim: int = DIM_POSE,
        latent_dim: int = LATENT_DIM,
        ff_size: int = FF_SIZE,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
        activation: str = ACTIVATION,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.query_token = nn.Parameter(torch.randn(1, latent_dim))
        self.embed_motion = nn.Linear(in_dim, latent_dim)
        self.sequence_pos_encoder = _PositionalEncoding(latent_dim, dropout, max_len=2000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_ln = nn.LayerNorm(latent_dim)
        self.out = nn.Linear(latent_dim, 512)

    def forward(self, motion: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        b = motion.shape[0]
        x_emb = self.embed_motion(motion)
        cls_tokens = self.query_token[torch.zeros(b, dtype=torch.long, device=motion.device)][:, None]
        emb = torch.cat([cls_tokens, x_emb], dim=1)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        h = self.sequence_pos_encoder(emb)
        h = h.permute(1, 0, 2)
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        h = h.permute(1, 0, 2)
        h = self.out_ln(h)
        return self.out(h[:, 0])


class MotionFeatureEncoder:
    """Loads the pretrained motion encoder and embeds canonical motions.

    Attributes:
        device: torch device the encoder runs on.
        mean: Per-channel mean used to Z-normalize the 67-D feature, shape `[67]`.
        std: Per-channel standard deviation for the same, shape `[67]`.
    """

    def __init__(
        self,
        device=None,
        checkpoint_path: Path = MOTIONCLIP_CHECKPOINT,
        mean_path: Path = FEATURE_MEAN,
        std_path: Path = FEATURE_STD,
    ):
        """Load encoder weights and feature normalization statistics.

        Raises:
            FileNotFoundError: If the checkpoint or the normalization statistics
                are missing, with the expected path in the message.
        """
        missing = [p for p in (checkpoint_path, mean_path, std_path) if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                "Motion feature encoder assets not found:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\n\nThese are third-party weights and are not distributed with "
                "PoseShield. See the Evaluation section of the README for where to "
                "download them and the exact layout they must be extracted to, or "
                "pass --no_mfd to report the other three metrics without MFD."
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mean = np.load(mean_path).astype(np.float32)
        self.std = np.load(std_path).astype(np.float32)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["contrast_model"]
        prefix = "motion_encoder."
        motion_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        self.model = MotionEncoder(in_dim=DIM_POSE).to(self.device)
        self.model.load_state_dict(motion_state, strict=True)
        self.model.eval()

    def encode_67d(self, feature_67: np.ndarray) -> np.ndarray:
        """Embed a 67-D feature sequence.

        Args:
            feature_67: Unnormalized feature, shape `[T - 1, 67]`, as returned
                by `joints_to_67d`.

        Returns:
            L2-normalized embedding, shape `[512]`.
        """
        t = feature_67.shape[0]
        normed = (feature_67 - self.mean) / self.std
        motion_t = torch.from_numpy(normed).float().unsqueeze(0).to(self.device)
        m_len = torch.tensor([t], dtype=torch.long, device=self.device)
        with torch.no_grad():
            padding_mask = ~_lengths_to_mask(m_len, motion_t.shape[1])
            emb = self.model(motion_t, padding_mask.to(motion_t.device))
            emb = emb / emb.norm(dim=1, keepdim=True)
        return emb.squeeze(0).cpu().numpy()


_ENCODER_SINGLETON: MotionFeatureEncoder | None = None


def encode(motion_135: np.ndarray) -> np.ndarray:
    """Embed a canonical motion, for use as `compute_mfd`'s `encode_fn`.

    Because both embeddings are unit-norm, the plain L2 distance `compute_mfd`
    takes between them is exactly the normalized feature distance. The encoder
    is built once per process and reused.

    Args:
        motion_135: Canonical motion, shape `[T, 135]`.

    Returns:
        L2-normalized embedding, shape `[512]`.
    """
    global _ENCODER_SINGLETON
    if _ENCODER_SINGLETON is None:
        _ENCODER_SINGLETON = MotionFeatureEncoder()

    joints = motion_to_joints(motion_135, _ENCODER_SINGLETON.device)[:, :NUM_JOINTS]
    return _ENCODER_SINGLETON.encode_67d(joints_to_67d(joints))
