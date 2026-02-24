"""
MANAS configuration for EEG-FM-Bench.
"""

from typing import Dict, Optional, List

from pydantic import Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class ManasDataArgs(BaseDataArgs):
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class ManasModelArgs(BaseModelArgs):
    pretrained_path: Optional[str] = None

    patch_seconds: float = 1.0
    overlap_seconds: float = 0.1

    embed_dim: int = 512
    encoder_depth: int = 12
    encoder_heads: int = 8
    decoder_depth: int = 4
    decoder_heads: int = 8

    mask_ratio: float = 0.55
    aux_loss_weight: float = 0.1


class ManasTrainingArgs(BaseTrainingArgs):
    max_epochs: int = 30

    lr_schedule: str = "cosine"
    max_lr: float = 2.0e-4
    min_lr: float = 2.0e-5
    warmup_epochs: int = 3
    warmup_scale: float = 1.0e-1

    use_amp: bool = True
    freeze_encoder: bool = False


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
        return True
