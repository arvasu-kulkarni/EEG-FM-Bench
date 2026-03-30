"""
MANAS-Long v0.1 encoder wrapper for EEG-FM-Bench.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from baseline.manas_long_v0_1.manas_long_v0_1_config import ManasLongV01ModelArgs
from baseline.manas_long_v0_1.manasfiles.model import MAE


class ManasLongV01Encoder(nn.Module):
    """Encoder that exposes MANAS-Long v0.1 features in (B, C, T, E) format."""

    def __init__(self, cfg: ManasLongV01ModelArgs, fs: int):
        super().__init__()
        self.mae = MAE(
            fs=fs,
            patch_seconds=cfg.patch_seconds,
            overlap_seconds=cfg.overlap_seconds,
            embed_dim=cfg.embed_dim,
            encoder_depth=cfg.encoder_depth,
            encoder_heads=cfg.encoder_heads,
            attn_dropout=cfg.attn_dropout,
            use_flash_attention=cfg.use_flash_attention,
            use_pairwise_channel_diffs=cfg.use_pairwise_channel_diffs,
            n_spatial_coords=cfg.n_spatial_coords,
            posenc_n_freqs=cfg.posenc_n_freqs,
            remove_bias_except_decoder_out=cfg.remove_bias_except_decoder_out,
            use_memory_tokens=cfg.use_memory_tokens,
            memory_scales_seconds=cfg.memory_scales_seconds,
            memory_active_scales_seconds=cfg.memory_active_scales_seconds,
            memory_attention_scales_seconds=cfg.memory_attention_scales_seconds,
            memory_encoder_access_start_layers=cfg.memory_encoder_access_start_layers,
            memory_conv_kernel_size=cfg.memory_conv_kernel_size,
            memory_conv_kernel_sizes=cfg.memory_conv_kernel_sizes,
            memory_patch_conv_channels=cfg.memory_patch_conv_channels,
            memory_token_dim=cfg.memory_token_dim,
            memory_token_dims=cfg.memory_token_dims,
        )

        self.patch_size = self.mae.patch_size
        self.step = self.mae.step
        self.embed_dim = cfg.embed_dim

        requested_pairwise = bool(cfg.use_pairwise_channel_diffs)
        supports_pairwise = hasattr(self.mae, "_to_pairwise_channels")
        if requested_pairwise and not supports_pairwise:
            raise ValueError(
                "model.use_pairwise_channel_diffs=true but current MANAS-Long v0.1 backend does not support pairwise channels."
            )

        self.use_pairwise_channel_diffs = requested_pairwise and supports_pairwise
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
            features = self.mae.encode(eeg, pos)
        else:
            expected_spatial = int(getattr(self.mae, "n_spatial_coords", pos.shape[-1]))
            if pos.shape[-1] != expected_spatial:
                if self.use_pairwise_channel_diffs and pos.shape[-1] * 2 == expected_spatial:
                    pos = torch.cat([pos, pos], dim=-1)
                else:
                    raise ValueError(
                        f"Unexpected MANAS-Long v0.1 position dimension: got {pos.shape[-1]}, expected {expected_spatial}"
                    )
            features = self.mae.encode(eeg, pos)

        return features
