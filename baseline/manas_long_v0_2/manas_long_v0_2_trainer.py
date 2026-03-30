"""
MANAS-Long v0.2 trainer for EEG-FM-Bench.
"""

import logging
from typing import Optional

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.manas_long.manas_long_trainer import ManasLongTrainer, ManasLongUnifiedModel
from baseline.manas_long_v0_2.manas_long_v0_2_adapter import ManasLongV02DataLoaderFactory
from baseline.manas_long_v0_2.manas_long_v0_2_config import ManasLongV02Config, ManasLongV02ModelArgs
from baseline.manas_long_v0_2.model import ManasLongV02Encoder

logger = logging.getLogger("baseline")


class ManasLongV02Trainer(ManasLongTrainer):
    def __init__(self, cfg: ManasLongV02Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.dataloader_factory = ManasLongV02DataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
            target_fs=self.cfg.fs,
            use_legacy_mne_positions=self.cfg.data.use_legacy_mne_positions,
            positions_json_path=self.cfg.data.positions_json_path,
        )
        self.encoder: Optional[ManasLongV02Encoder] = None

    def setup_model(self):
        logger.info("Setting up MANAS-Long v0.2 model architecture...")
        cfg: ManasLongV02ModelArgs = self.cfg.model

        self.encoder = ManasLongV02Encoder(cfg, fs=self.cfg.fs)

        embed_dim = cfg.embed_dim
        head_configs = {ds_name: info["n_class"] for ds_name, info in self.ds_info.items()}
        head_cfg = cfg.classifier_head

        patch_size = int(round(cfg.patch_seconds * self.cfg.fs))
        overlap_size = int(round(cfg.overlap_seconds * self.cfg.fs))

        ds_shape_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info["shape_info"].items():
                n_patches = self.compute_patches_num(n_timepoints, patch_size, overlap_size)
                if n_patches <= 0:
                    raise ValueError(
                        f"Dataset sample too short for MANAS-Long v0.2 patching: montage={montage_key}, "
                        f"timepoints={n_timepoints}, patch_size={patch_size}, overlap={overlap_size}"
                    )
                n_channels_eff = self.encoder.effective_num_channels(n_channels, ds_name=ds_name)
                ds_shape_info[montage_key] = (n_patches, n_channels_eff, embed_dim)
                token_count = n_channels_eff * n_patches
                pairwise_flag = self.encoder.uses_pairwise_for_dataset(ds_name)
                logger.info(
                    f"MANAS-Long v0.2 shape {montage_key}: patches={n_patches}, channels_eff={n_channels_eff}, "
                    f"tokens={token_count}, pairwise={'on' if pairwise_flag else 'off'}"
                )

        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            head_configs=head_configs,
            head_cfg=head_cfg,
            ds_shape_info=ds_shape_info,
            t_sne=cfg.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")

        if not cfg.pretrained_path:
            raise ValueError(
                "MANAS-Long v0.2 requires model.pretrained_path. "
                "Training from scratch is disabled."
            )
        self.load_checkpoint(cfg.pretrained_path)

        model = ManasLongUnifiedModel(
            encoder=self.encoder,
            classifier=self.classifier,
            grad_cam=cfg.grad_cam,
        )

        model = self.apply_lora(model)
        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model, find_unused_parameters=True)
        self.model = model

        return model
