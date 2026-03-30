"""
MANAS-Long v0.2 configuration for EEG-FM-Bench.
"""

from typing import Dict, List, Optional

from pydantic import Field

from baseline.abstract.config import AbstractConfig
from baseline.manas_long.manas_long_config import (
    ManasLongDataArgs,
    ManasLongLoggingArgs,
    ManasLongModelArgs,
    ManasLongTrainingArgs,
)


class ManasLongV02ModelArgs(ManasLongModelArgs):
    memory_encoder_access_start_layers: Dict[float, int] = Field(
        default_factory=lambda: {5.0: 9, 10.0: 9, 30.0: 15}
    )
    memory_decoder_access_start_layers: Dict[float, int] = Field(
        default_factory=lambda: {5.0: 3, 10.0: 2, 30.0: 1}
    )
    memory_conv_kernel_size: Optional[int] = None
    memory_conv_kernel_sizes: Dict[float, int] = Field(default_factory=lambda: {})
    memory_token_dim: Optional[int] = None
    memory_token_dims: Dict[float, int] = Field(default_factory=lambda: {})
    memory_attention_use_null_token: bool = False
    num_local_aux_tokens: int = 0


class ManasLongV02LoggingArgs(ManasLongLoggingArgs):
    experiment_name: str = "manas-long-v0.2"
    project: Optional[str] = "manas-long-v0.2"
    tags: List[str] = Field(default_factory=lambda: ["manas-long-v0.2"])


class ManasLongV02Config(AbstractConfig):
    model_type: str = "manas-long-v0.2"
    fs: int = 200

    data: ManasLongDataArgs = Field(default_factory=ManasLongDataArgs)
    model: ManasLongV02ModelArgs = Field(default_factory=ManasLongV02ModelArgs)
    training: ManasLongTrainingArgs = Field(default_factory=ManasLongTrainingArgs)
    logging: ManasLongV02LoggingArgs = Field(default_factory=ManasLongV02LoggingArgs)

    def validate_config(self) -> bool:
        if self.model.patch_seconds <= 0:
            return False
        if self.model.overlap_seconds < 0:
            return False
        if self.model.which_mask not in {"default", "fuzzy"}:
            return False
        if not 0.0 <= self.model.dropout_ratio <= 1.0:
            return False
        if self.model.n_spatial_coords <= 0:
            return False
        if self.model.posenc_n_freqs <= 0:
            return False
        if not 0.0 <= self.model.attn_dropout < 1.0:
            return False
        if self.model.memory_patch_conv_channels <= 0:
            return False
        if self.model.memory_teacher_loss_weight < 0.0:
            return False
        if self.model.memory_conv_kernel_size is not None and self.model.memory_conv_kernel_size <= 0:
            return False
        if self.model.memory_token_dim is not None and self.model.memory_token_dim <= 0:
            return False
        if self.model.num_local_aux_tokens < 0 or self.model.num_local_aux_tokens == 1:
            return False

        memory_scales = tuple(float(scale) for scale in self.model.memory_scales_seconds)
        if any(scale <= 0.0 for scale in memory_scales):
            return False

        active_scales = tuple(float(scale) for scale in self.model.memory_active_scales_seconds)
        if any(scale <= 0.0 for scale in active_scales):
            return False
        if any(scale not in memory_scales for scale in active_scales):
            return False

        attention_scales = tuple(float(scale) for scale in self.model.memory_attention_scales_seconds)
        if any(scale <= 0.0 for scale in attention_scales):
            return False
        if any(scale not in memory_scales for scale in attention_scales):
            return False

        encoder_access = {
            float(scale): int(start_layer)
            for scale, start_layer in self.model.memory_encoder_access_start_layers.items()
        }
        if any(scale not in memory_scales for scale in encoder_access):
            return False
        if any(start_layer <= 0 for start_layer in encoder_access.values()):
            return False
        if any(scale not in encoder_access for scale in attention_scales):
            return False

        decoder_access = {
            float(scale): int(start_layer)
            for scale, start_layer in self.model.memory_decoder_access_start_layers.items()
        }
        if any(scale not in memory_scales for scale in decoder_access):
            return False
        if any(start_layer <= 0 for start_layer in decoder_access.values()):
            return False
        if any(scale not in decoder_access for scale in attention_scales):
            return False

        conv_kernel_sizes = {
            float(scale): int(kernel_size)
            for scale, kernel_size in self.model.memory_conv_kernel_sizes.items()
        }
        if any(scale not in memory_scales for scale in conv_kernel_sizes):
            return False
        if any(kernel_size <= 0 for kernel_size in conv_kernel_sizes.values()):
            return False

        token_dims = {
            float(scale): int(token_dim)
            for scale, token_dim in self.model.memory_token_dims.items()
        }
        if any(scale not in memory_scales for scale in token_dims):
            return False
        if any(token_dim <= 0 for token_dim in token_dims.values()):
            return False

        if self.model.use_memory_tokens and self.model.use_pairwise_channel_diffs:
            return False

        return True
