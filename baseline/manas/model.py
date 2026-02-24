"""
MANAS encoder wrapper for EEG-FM-Bench.
"""

from __future__ import annotations

import torch
from torch import nn

from baseline.manas.manas_config import ManasModelArgs
from baseline.manas.manasfiles.model import MAE


class ManasEncoder(nn.Module):
    """Encoder that exposes MAE features in (B, C, T, E) format."""

    def __init__(self, cfg: ManasModelArgs, fs: int):
        super().__init__()
        self.mae = MAE(
            fs=fs,
            patch_seconds=cfg.patch_seconds,
            overlap_seconds=cfg.overlap_seconds,
            embed_dim=cfg.embed_dim,
            encoder_depth=cfg.encoder_depth,
            encoder_heads=cfg.encoder_heads,
            decoder_depth=cfg.decoder_depth,
            decoder_heads=cfg.decoder_heads,
            mask_ratio=cfg.mask_ratio,
            aux_loss_weight=cfg.aux_loss_weight,
        )

        self.patch_size = self.mae.patch_size
        self.step = self.mae.step
        self.embed_dim = cfg.embed_dim

    def forward(self, eeg: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        eeg = eeg.float()
        patches = eeg.unfold(dimension=-1, size=self.patch_size, step=self.step)
        b, c, n_patches, _ = patches.shape

        tokens = self.mae.patch_embed.linear(patches).flatten(1, 2)
        coords = self.mae.prepare_coords(pos, n_patches)
        pos_emb = self.mae.pos_enc(coords)

        x = tokens + pos_emb
        x, _ = self.mae.encoder(x)

        x = x.reshape(b, c, n_patches, self.embed_dim)
        return x
