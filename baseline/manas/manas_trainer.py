"""
MANAS trainer for EEG-FM-Bench.
"""

import logging
import os
from typing import Optional, Literal
import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.manas.manas_adapter import ManasDataLoaderFactory
from baseline.manas.manas_config import ManasConfig, ManasModelArgs
from baseline.manas.model import ManasEncoder

logger = logging.getLogger('baseline')


class ManasUnifiedModel(nn.Module):
    def __init__(self, encoder: ManasEncoder, classifier: MultiHeadClassifier, grad_cam: bool = False):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

        self.grad_cam = grad_cam
        self.grad_cam_activation = None

    def forward(self, batch):
        x = batch['data']
        pos = batch['pos']
        montage = batch['montage'][0]

        features = self.encoder(x, pos)
        features = features.permute(0, 2, 1, 3)

        if self.grad_cam:
            self.grad_cam_activation = features

        logits = self.classifier(features, montage)
        return logits


class ManasTrainer(AbstractTrainer):
    def __init__(self, cfg: ManasConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.dataloader_factory = ManasDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
            target_fs=self.cfg.fs,
        )

        self.encoder: Optional[ManasEncoder] = None
        self.classifier: Optional[MultiHeadClassifier] = None
        self.loss_fn = nn.CrossEntropyLoss()
        self._last_effective_train_method: Optional[str] = None

    @staticmethod
    def compute_patches_num(n_timepoints: int, patch_size: int, overlap_size: int) -> int:
        step = patch_size - overlap_size
        if step <= 0:
            raise ValueError(
                f"Invalid patch config: patch_size={patch_size}, overlap={overlap_size} (step={step})"
            )
        if n_timepoints < patch_size:
            return 0
        return 1 + (n_timepoints - patch_size) // step

    def setup_model(self):
        logger.info("Setting up MANAS model architecture...")
        cfg: ManasModelArgs = self.cfg.model

        self.encoder = ManasEncoder(cfg, fs=self.cfg.fs)

        embed_dim = cfg.embed_dim
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        head_cfg = cfg.classifier_head

        patch_size = int(round(cfg.patch_seconds * self.cfg.fs))
        overlap_size = int(round(cfg.overlap_seconds * self.cfg.fs))

        ds_shape_info = {}
        for ds_name, info in self.ds_info.items():
            for montage_key, (n_timepoints, n_channels) in info['shape_info'].items():
                n_patches = self.compute_patches_num(n_timepoints, patch_size, overlap_size)
                if n_patches <= 0:
                    raise ValueError(
                        f"Dataset sample too short for MANAS patching: montage={montage_key}, timepoints={n_timepoints}, "
                        f"patch_size={patch_size}, overlap={overlap_size}"
                    )
                ds_shape_info[montage_key] = (n_patches, n_channels, embed_dim)

        self.classifier = MultiHeadClassifier(
            embed_dim=embed_dim,
            head_configs=head_configs,
            head_cfg=head_cfg,
            ds_shape_info=ds_shape_info,
            t_sne=cfg.t_sne,
        )
        logger.info(f"Created multi-head classifier with heads: {list(head_configs.keys())}")

        if cfg.pretrained_path:
            self.load_checkpoint(cfg.pretrained_path)
        else:
            logger.info("No pretrained path specified, starting from scratch")

        model = ManasUnifiedModel(
            encoder=self.encoder,
            classifier=self.classifier,
            grad_cam=cfg.grad_cam,
        )

        model = self.apply_lora(model)
        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model, find_unused_parameters=True)
        self.model = model

        return model

    def load_checkpoint(self, checkpoint_path: str):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return

        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(ckpt, dict):
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        else:
            state_dict = ckpt

        if self.encoder is None:
            logger.warning("Encoder not initialized; skipping checkpoint load")
            return

        missing, unexpected = self.encoder.mae.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

        logger.info("Successfully loaded pretrained MANAS weights")

    def _effective_train_method(self) -> Literal["linear_probe", "partial_ft", "full_ft"]:
        """Resolve training method for current epoch, including dual-stage scheduling."""
        method: Literal["linear_probe", "partial_ft", "full_ft"] = self.cfg.training.train_method
        if self.cfg.training.dual_stage:
            switch_epoch = self.cfg.training.max_epochs // 2
            if self.epoch < switch_epoch:
                return "linear_probe"
        return method

    def _set_encoder_trainability(self, method: Literal["linear_probe", "partial_ft", "full_ft"]):
        """Apply encoder freezing policy for MANAS according to selected train method."""
        if self.model is None:
            return

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        if not hasattr(model, "encoder"):
            raise AttributeError("MANAS model does not have an encoder attribute")

        encoder = model.encoder

        # Start from fully frozen encoder, then selectively unfreeze.
        for param in encoder.parameters():
            param.requires_grad = False

        if method == "linear_probe":
            pass
        elif method == "partial_ft":
            # Unfreeze last encoder block (+ final norm) for lightweight adaptation.
            for param in encoder.mae.encoder.layers[-1].parameters():
                param.requires_grad = True
            for param in encoder.mae.encoder.final_norm.parameters():
                param.requires_grad = True
        elif method == "full_ft":
            for param in encoder.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown train_method: {method}")

        # Keep classifier trainable in all strategies.
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

    def setup_optimizer_and_scheduler(self, model, train_loader):
        # Always build optimizer with encoder param group present.
        # Runtime trainability is controlled per epoch via requires_grad flags.
        original_freeze_encoder = self.cfg.training.freeze_encoder
        self.cfg.training.freeze_encoder = False
        try:
            super().setup_optimizer_and_scheduler(model, train_loader)
        finally:
            self.cfg.training.freeze_encoder = original_freeze_encoder

    def train_epoch(self, train_loader, train_sampler):
        effective_method = self._effective_train_method()
        self._set_encoder_trainability(effective_method)
        self.cfg.training.freeze_encoder = (effective_method == "linear_probe")

        if effective_method != self._last_effective_train_method:
            logger.info(
                f"Epoch {self.epoch}: applying train_method='{effective_method}' "
                f"(configured='{self.cfg.training.train_method}', dual_stage={self.cfg.training.dual_stage})"
            )
            self._last_effective_train_method = effective_method

        super().train_epoch(train_loader, train_sampler)
