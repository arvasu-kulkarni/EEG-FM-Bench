"""
MANAS configuration for EEG-FM-Bench.
"""

from typing import Dict, Optional, List, Literal

from pydantic import Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class ManasDataArgs(BaseDataArgs):
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2
    # Position source controls for MANAS adapter.
    # True: legacy MNE montage-based position generation.
    # False: read positions from positions_json_path (with MNE fallback on missing channels).
    use_legacy_mne_positions: bool = True
    positions_json_path: Optional[str] = None


class ManasModelArgs(BaseModelArgs):
    pretrained_path: Optional[str] = None
    mae_type: Literal["default", "mahir"] = "default"

    patch_seconds: float = 1.0
    overlap_seconds: float = 0.1

    embed_dim: int = 512
    encoder_depth: int = 12
    encoder_heads: int = 8
    decoder_depth: int = 4
    decoder_heads: int = 8

    mask_ratio: float = 0.55
    aux_loss_weight: float = 0.1

    which_mask: Literal["default", "fuzzy"] = "default"
    fuzzy_noise_std: float = 0.1
    spatial_radius_black: float = 3.0
    spatial_radius_fuzzy: float = 6.0
    temporal_radius_black: float = 3.0
    temporal_radius_fuzzy: float = 6.0
    dropout_ratio: float = 0.0
    dropout_radius: float = 3.0
    ema_mix_ratio: float = 0.6
    ema_temperature: float = 2.0
    ema_floor_eps: float = 0.1

    # NC2 / pairwise-channel controls (Mahir MAE backend only).
    use_pairwise_channel_diffs: bool = False
    pairwise_exclude_datasets: List[str] = Field(default_factory=lambda: [])
    n_spatial_coords: int = 3
    posenc_n_freqs: int = 4


class ManasTrainingArgs(BaseTrainingArgs):
    max_epochs: int = 30

    lr_schedule: str = "cosine"
    max_lr: float = 2.0e-4
    min_lr: float = 2.0e-5
    warmup_epochs: int = 3
    warmup_scale: float = 1.0e-1

    use_amp: bool = True
    train_method: Literal["linear_probe", "partial_ft", "full_ft"] = "linear_probe"
    dual_stage: bool = False


class ManasLoggingArgs(BaseLoggingArgs):
    experiment_name: str = "manas"
    run_dir: str = "assets/run"

    use_cloud: bool = True
    cloud_backend: str = "wandb"
    project: Optional[str] = "manas"

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: ["manas"])

    log_step_interval: int = 1
    ckpt_interval: int = 1


class ManasConfig(AbstractConfig):
    model_type: str = "manas"
    fs: int = 200

    data: ManasDataArgs = Field(default_factory=ManasDataArgs)
    model: ManasModelArgs = Field(default_factory=ManasModelArgs)
    training: ManasTrainingArgs = Field(default_factory=ManasTrainingArgs)
    logging: ManasLoggingArgs = Field(default_factory=ManasLoggingArgs)

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
        return True
