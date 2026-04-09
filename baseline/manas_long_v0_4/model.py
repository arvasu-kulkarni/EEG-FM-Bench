"""
MANAS-Long v0.4 encoder wrapper for EEG-FM-Bench.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from baseline.manas_long_v0_4.manas_long_v0_4_config import ManasLongV04ModelArgs
from baseline.manas_long_v0_4.manasfiles.model import MAE


class ManasLongV04Encoder(nn.Module):
    """Encoder that exposes upstream v0.4 features in (B, C, T, E) format."""

    def __init__(self, cfg: ManasLongV04ModelArgs, fs: int):
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
            attn_dropout=cfg.attn_dropout,
            use_flash_attention=cfg.use_flash_attention,
            pool_from_mha_outputs=cfg.pool_from_mha_outputs,
            remove_bias_except_decoder_out=cfg.remove_bias_except_decoder_out,
            use_memory_tokens=cfg.use_memory_tokens,
            memory_scales_seconds=cfg.memory_scales_seconds,
            memory_active_scales_seconds=cfg.memory_active_scales_seconds,
            memory_attention_scales_seconds=cfg.memory_attention_scales_seconds,
            memory_encoder_access_start_layers=cfg.memory_encoder_access_start_layers,
            memory_decoder_access_start_layers=cfg.memory_decoder_access_start_layers,
            memory_conv_kernel_size=cfg.memory_conv_kernel_size,
            memory_conv_kernel_sizes=cfg.memory_conv_kernel_sizes,
            memory_patch_conv_channels=cfg.memory_patch_conv_channels,
            memory_token_dim=cfg.memory_token_dim,
            memory_token_dims=cfg.memory_token_dims,
            memory_attention_use_null_token=cfg.memory_attention_use_null_token,
            num_local_aux_tokens=cfg.num_local_aux_tokens,
        )

        self.patch_size = self.mae.patch_size
        self.step = self.mae.step
        self.embed_dim = cfg.embed_dim

        self.base_temporal_radius_black = float(getattr(self.mae, "temporal_radius_black", 0.0))
        self.base_temporal_radius_fuzzy = float(getattr(self.mae, "temporal_radius_fuzzy", 0.0))
        configured_scales = tuple(float(scale) for scale in getattr(self.mae, "memory_active_scales_seconds", tuple()))
        self.mae.configured_memory_active_scales_seconds = configured_scales
        self.memory_window_policy_enabled = bool(cfg.memory_window_policy_enabled)
        self.memory_window_scale_thresholds_seconds = {
            float(scale): float(min_window)
            for scale, min_window in cfg.memory_window_scale_thresholds_seconds.items()
        }
        self.memory_window_behavior_if_none_valid = cfg.memory_window_behavior_if_none_valid

        requested_pairwise = bool(cfg.use_pairwise_channel_diffs)
        supports_pairwise = hasattr(self.mae, "_to_pairwise_channels")
        if requested_pairwise and not supports_pairwise:
            raise ValueError(
                "model.use_pairwise_channel_diffs=true but current MANAS-Long v0.4 backend does not support pairwise channels."
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

    def resolve_active_memory_scales(self, current_window_sec: float) -> tuple[float, ...]:
        if not getattr(self.mae, "use_memory_tokens", False):
            return tuple()

        configured = tuple(getattr(self.mae, "configured_memory_active_scales_seconds", tuple()))
        if not self.memory_window_policy_enabled:
            return configured

        active = tuple(
            scale
            for scale in configured
            if float(current_window_sec) + 1e-12 >= self.memory_window_scale_thresholds_seconds.get(float(scale), float(scale))
        )
        if active:
            return active
        if self.memory_window_behavior_if_none_valid == "disable_memory":
            return tuple()
        raise ValueError(
            f"Unsupported memory_window_behavior_if_none_valid={self.memory_window_behavior_if_none_valid!r}"
        )

    def set_batch_context(
        self,
        *,
        active_memory_scales: tuple[float, ...] | list[float] | None = None,
        temporal_radius_scale: float | None = None,
    ) -> dict:
        if getattr(self.mae, "use_memory_tokens", False):
            if active_memory_scales is None:
                active = tuple(getattr(self.mae, "configured_memory_active_scales_seconds", tuple()))
            else:
                active = tuple(float(scale) for scale in active_memory_scales)
            self.mae.memory_active_scales_seconds = active
            self.mae.memory_build_order_seconds = tuple(sorted(active))
        else:
            active = tuple()

        if temporal_radius_scale is None:
            self.mae.temporal_radius_black = float(self.base_temporal_radius_black)
            self.mae.temporal_radius_fuzzy = float(self.base_temporal_radius_fuzzy)
        else:
            scale = max(0.0, float(temporal_radius_scale))
            self.mae.temporal_radius_black = float(self.base_temporal_radius_black) * scale
            self.mae.temporal_radius_fuzzy = float(self.base_temporal_radius_fuzzy) * scale

        return {
            "active_memory_scales": active,
            "temporal_radius_black": float(getattr(self.mae, "temporal_radius_black", 0.0)),
            "temporal_radius_fuzzy": float(getattr(self.mae, "temporal_radius_fuzzy", 0.0)),
        }

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
                        f"Unexpected MANAS-Long v0.4 position dimension: got {pos.shape[-1]}, expected {expected_spatial}"
                    )
            features = self.mae.encode(eeg, pos)

        return features
