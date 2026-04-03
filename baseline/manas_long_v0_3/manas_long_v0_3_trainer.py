"""
MANAS-Long v0.3 trainer for EEG-FM-Bench.
"""

import logging
from typing import Literal, Optional

import torch

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.manas_long.manas_long_trainer import ManasLongTrainer, ManasLongUnifiedModel
from baseline.manas_long_v0_3.manas_long_v0_3_adapter import ManasLongV03DataLoaderFactory
from baseline.manas_long_v0_3.manas_long_v0_3_config import ManasLongV03Config, ManasLongV03ModelArgs
from baseline.manas_long_v0_3.model import ManasLongV03Encoder

logger = logging.getLogger("baseline")


class ManasLongV03Trainer(ManasLongTrainer):
    def __init__(self, cfg: ManasLongV03Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.dataloader_factory = ManasLongV03DataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
            target_fs=self.cfg.fs,
            use_legacy_mne_positions=self.cfg.data.use_legacy_mne_positions,
            positions_json_path=self.cfg.data.positions_json_path,
        )
        self.encoder: Optional[ManasLongV03Encoder] = None

    def setup_model(self):
        logger.info("Setting up MANAS-Long v0.3 model architecture...")
        cfg: ManasLongV03ModelArgs = self.cfg.model

        self.encoder = ManasLongV03Encoder(cfg, fs=self.cfg.fs)

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
                        f"Dataset sample too short for MANAS-Long v0.3 patching: montage={montage_key}, "
                        f"timepoints={n_timepoints}, patch_size={patch_size}, overlap={overlap_size}"
                    )
                n_channels_eff = self.encoder.effective_num_channels(n_channels, ds_name=ds_name)
                ds_shape_info[montage_key] = (n_patches, n_channels_eff, embed_dim)
                token_count = n_channels_eff * n_patches
                pairwise_flag = self.encoder.uses_pairwise_for_dataset(ds_name)
                logger.info(
                    f"MANAS-Long v0.3 shape {montage_key}: patches={n_patches}, channels_eff={n_channels_eff}, "
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
                "MANAS-Long v0.3 requires model.pretrained_path. "
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

    def _set_encoder_trainability(self, method: Literal["linear_probe", "partial_ft", "full_ft"]):
        if self.model is None:
            return

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        if not hasattr(model, "encoder"):
            raise AttributeError("MANAS-Long v0.3 model does not have an encoder attribute")

        encoder = model.encoder

        for param in encoder.parameters():
            param.requires_grad = False

        if method == "linear_probe":
            pass
        elif method == "partial_ft":
            for param in encoder.mae.encoder.layers[-1].parameters():
                param.requires_grad = True
            for param in encoder.mae.encoder.final_norm.parameters():
                param.requires_grad = True
            if hasattr(encoder.mae.encoder, "memory_branch"):
                for param in encoder.mae.encoder.memory_branch.parameters():
                    param.requires_grad = True
            if hasattr(encoder.mae.encoder, "memory_branches"):
                for param in encoder.mae.encoder.memory_branches.parameters():
                    param.requires_grad = True
            if hasattr(encoder.mae.encoder, "memory_scale_alpha"):
                encoder.mae.encoder.memory_scale_alpha.requires_grad = True
            for param in encoder.mae.memory_patch_embeds.parameters():
                param.requires_grad = True
            for param in encoder.mae.memory_token_modules.parameters():
                param.requires_grad = True
        elif method == "full_ft":
            for param in encoder.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown train_method: {method}")

        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
