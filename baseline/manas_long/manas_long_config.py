"""
MANAS-Long configuration for EEG-FM-Bench.
"""

from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs
from baseline.manas.manas_config import ManasTrainingArgs, ManasLoggingArgs


class ManasLongDataArgs(BaseDataArgs):
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 4096
    num_workers: int = 2
    use_legacy_mne_positions: bool = True
    positions_json_path: Optional[str] = None


class ManasLongModelArgs(BaseModelArgs):
    pretrained_path: Optional[str] = None

    patch_seconds: float = 1.0
    overlap_seconds: float = 0.1

    embed_dim: int = 512
    encoder_depth: int = 22
    encoder_heads: int = 8
    decoder_depth: int = 5
    decoder_heads: int = 8

    mask_ratio: float = 0.55
    aux_loss_weight: float = 0.5

    which_mask: Literal["default", "fuzzy"] = "default"
    fuzzy_noise_std: float = 0.0
    spatial_radius_black: float = 3.0
    spatial_radius_fuzzy: float = 4.0
    temporal_radius_black: float = 3.0
    temporal_radius_fuzzy: float = 4.0
    dropout_ratio: float = 0.1
    dropout_radius: float = 4.0
    ema_mix_ratio: float = 0.6
    ema_temperature: float = 2.0
    ema_floor_eps: float = 0.1

    use_pairwise_channel_diffs: bool = False
    pairwise_exclude_datasets: List[str] = Field(default_factory=lambda: [])
    n_spatial_coords: int = 3
    posenc_n_freqs: int = 4

    attn_dropout: float = 0.0
    use_flash_attention: bool = True
    pool_from_mha_outputs: bool = True
    remove_bias_except_decoder_out: bool = False

    use_memory_tokens: bool = True
    memory_scales_seconds: List[float] = Field(default_factory=lambda: [5.0, 10.0, 30.0])
    memory_active_scales_seconds: List[float] = Field(default_factory=lambda: [5.0, 10.0, 30.0])
    memory_attention_scales_seconds: List[float] = Field(default_factory=lambda: [5.0, 10.0, 30.0])
    memory_conv_kernel_size: int = 3
    memory_patch_conv_channels: int = 64
    memory_teacher_loss_weight: float = 0.1
    memory_teacher_loss_type: Literal["smooth_l1", "mse"] = "smooth_l1"


ManasLongTrainMethod = Literal[
    "linear_probe",
    "partial_ft",
    "full_ft",
    "mem_freeze",
    "memory_probe",
    "partial-enc",
]


class ManasLongTrainingArgs(ManasTrainingArgs):
    train_method: ManasLongTrainMethod = "linear_probe"
    enc_layer: Optional[int] = None
    freeze: Optional[Literal["after", "before"]] = None
    freeze_memory: bool = False

    @model_validator(mode="after")
    def validate_partial_enc_args(self):
        if self.enc_layer is not None and self.enc_layer <= 0:
            raise ValueError("training.enc_layer must be >= 1 when provided.")
        if self.train_method == "partial-enc":
            if self.enc_layer is None:
                raise ValueError("training.enc_layer is required when train_method='partial-enc'.")
            if self.freeze is None:
                raise ValueError("training.freeze is required when train_method='partial-enc'.")
        return self


class ManasLongLoggingArgs(ManasLoggingArgs):
    experiment_name: str = "manas-long"
    project: Optional[str] = "manas-long"
    tags: List[str] = Field(default_factory=lambda: ["manas-long"])


class ManasLongConfig(AbstractConfig):
    model_type: str = "manas-long"
    fs: int = 200

    data: ManasLongDataArgs = Field(default_factory=ManasLongDataArgs)
    model: ManasLongModelArgs = Field(default_factory=ManasLongModelArgs)
    training: ManasLongTrainingArgs = Field(default_factory=ManasLongTrainingArgs)
    logging: ManasLongLoggingArgs = Field(default_factory=ManasLongLoggingArgs)

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
        if self.model.memory_conv_kernel_size <= 0 or self.model.memory_conv_kernel_size % 2 == 0:
            return False
        if self.model.memory_patch_conv_channels <= 0:
            return False
        if self.model.memory_teacher_loss_weight < 0.0:
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

        if self.model.use_memory_tokens and self.model.use_pairwise_channel_diffs:
            return False

        return True
