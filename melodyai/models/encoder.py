"""Neural melody encoder (PyTorch).

Input: a key-normalised, time-normalised pitch contour of ``n_points`` samples (the same
representation used for window matching, see ``melodyai.features.windows``), turned into
two channels: centred pitch / 12 and its first difference.

Architecture (≈0.4 M parameters, sized for a few thousand training melodies on a laptop):

    Conv1d stem -> 3 residual dilated Conv1d blocks -> 2-layer Transformer encoder
    -> attention pooling -> linear projection -> L2-normalised embedding (dim 128)

One shared encoder embeds both hummed queries and reference windows (a siamese
setup). An optional small query-side adapter makes it a light asymmetric dual encoder.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation), nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation), nn.GroupNorm(8, ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class MelodyEncoder(nn.Module):
    def __init__(self, n_points: int = 96, channels: int = 128, dim: int = 128, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1, asymmetric: bool = False) -> None:
        super().__init__()
        self.config = dict(n_points=n_points, channels=channels, dim=dim, n_layers=n_layers, n_heads=n_heads,
                           dropout=dropout, asymmetric=asymmetric)
        self.stem = nn.Sequential(nn.Conv1d(2, channels, 5, padding=2), nn.GroupNorm(8, channels), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(channels, d) for d in (1, 2, 4)])
        self.pos = nn.Parameter(torch.zeros(1, n_points, channels))
        layer = nn.TransformerEncoderLayer(channels, n_heads, channels * 2, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.attn = nn.Linear(channels, 1)
        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, dim))
        self.query_adapter = nn.Linear(dim, dim) if asymmetric else None
        if self.query_adapter is not None:
            nn.init.eye_(self.query_adapter.weight)
            nn.init.zeros_(self.query_adapter.bias)

    @staticmethod
    def featurize(x: torch.Tensor) -> torch.Tensor:
        """(B, T) centred semitone contour -> (B, 2, T) normalised channels."""
        d = torch.diff(x, dim=1, prepend=x[:, :1])
        return torch.stack([x / 12.0, d.clamp(-12, 12) / 4.0], dim=1)

    def forward(self, x: torch.Tensor, is_query: bool = False) -> torch.Tensor:
        h = self.blocks(self.stem(self.featurize(x)))          # (B, C, T)
        h = self.transformer(h.transpose(1, 2) + self.pos)    # (B, T, C)
        w = torch.softmax(self.attn(h), dim=1)                 # (B, T, 1)
        z = self.proj((w * h).sum(dim=1))
        if is_query and self.query_adapter is not None:
            z = self.query_adapter(z)
        return nn.functional.normalize(z, dim=-1)


def save_checkpoint(path, model: MelodyEncoder, extra: dict | None = None) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_encoder(path, device: str = "cpu") -> tuple[MelodyEncoder, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MelodyEncoder(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt
