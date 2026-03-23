"""
MANAS encoder wrapper for EEG-FM-Bench.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from baseline.manas.manas_config import ManasModelArgs
from baseline.manas.manasfiles.model import MAE
from baseline.manas.manasfiles.mahirmodel import MAE as MahirMAE


class ManasEncoder(nn.Module):
    """Encoder that exposes MAE features in (B, C, T, E) format."""

    def __init__(self, cfg: ManasModelArgs, fs: int):
        super().__init__()
        if cfg.mae_type == "mahir":
            self.mae = MahirMAE(
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
                which_mask=cfg.which_mask,
                fuzzy_noise_std=cfg.fuzzy_noise_std,
                spatial_radius_black=cfg.spatial_radius_black,
                spatial_radius_fuzzy=cfg.spatial_radius_fuzzy,
                temporal_radius_black=cfg.temporal_radius_black,
                temporal_radius_fuzzy=cfg.temporal_radius_fuzzy,
                dropout_ratio=cfg.dropout_ratio,
                dropout_radius=cfg.dropout_radius,
                ema_mix_ratio=cfg.ema_mix_ratio,
                ema_temperature=cfg.ema_temperature,
                ema_floor_eps=cfg.ema_floor_eps,
                use_pairwise_channel_diffs=cfg.use_pairwise_channel_diffs,
                n_spatial_coords=cfg.n_spatial_coords,
                posenc_n_freqs=cfg.posenc_n_freqs,
            )
        else:
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
        self.use_pairwise_channel_diffs = bool(getattr(self.mae, "use_pairwise_channel_diffs", False))
        self.pairwise_exclude_datasets = {
            ds_name.strip().lower()
            for ds_name in getattr(cfg, "pairwise_exclude_datasets", [])
            if isinstance(ds_name, str) and ds_name.strip()
        }

    @staticmethod
    def _resolve_dataset_name(montage: Optional[str]) -> Optional[str]:
        if montage is None:
            return None
        if "/" in montage:
            return montage.split("/", 1)[0].strip().lower()
        return montage.strip().lower()

    def uses_pairwise_for_dataset(self, ds_name: Optional[str]) -> bool:
        if not self.use_pairwise_channel_diffs:
            return False
        if ds_name is None:
            return True
        return ds_name.strip().lower() not in self.pairwise_exclude_datasets

    def effective_num_channels(self, num_channels: int, ds_name: Optional[str] = None) -> int:
        if not self.uses_pairwise_for_dataset(ds_name):
            return num_channels
        return num_channels + (num_channels * (num_channels - 1)) // 2

    def forward(self, eeg: torch.Tensor, pos: torch.Tensor, montage: Optional[str] = None) -> torch.Tensor:
        eeg = eeg.float()
        pos = pos.float()

        ds_name = self._resolve_dataset_name(montage)
        use_pairwise = self.uses_pairwise_for_dataset(ds_name)

        if use_pairwise:
            eeg, pos = self.mae._to_pairwise_channels(eeg, pos)
        else:
            expected_spatial = int(getattr(self.mae, "n_spatial_coords", pos.shape[-1]))
            if pos.shape[-1] != expected_spatial:
                # Keep checkpoint-compatible positional encoding shape when pairwise channels are skipped.
                if self.use_pairwise_channel_diffs and pos.shape[-1] * 2 == expected_spatial:
                    pos = torch.cat([pos, pos], dim=-1)
                else:
                    raise ValueError(
                        f"Unexpected MANAS position dimension: got {pos.shape[-1]}, expected {expected_spatial}"
                    )

        patches = eeg.unfold(dimension=-1, size=self.patch_size, step=self.step)
        b, c_eff, n_patches, _ = patches.shape

        tokens = self.mae.patch_embed.linear(patches).flatten(1, 2)
        coords = self.mae.prepare_coords(pos, n_patches)
        pos_emb = self.mae.pos_enc(coords)

        x = tokens + pos_emb
        x, _ = self.mae.encoder(x)

        x = x.reshape(b, c_eff, n_patches, self.embed_dim)
        return x
