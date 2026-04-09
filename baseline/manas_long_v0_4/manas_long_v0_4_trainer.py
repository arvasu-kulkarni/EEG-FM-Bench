"""
MANAS-Long v0.4 trainer for EEG-FM-Bench.
"""

import logging
from typing import Optional

import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.manas_long.manas_long_config import ManasLongTrainMethod
from baseline.manas_long.manas_long_trainer import ManasLongTrainer
from baseline.manas_long_v0_4.manas_long_v0_4_adapter import ManasLongV04DataLoaderFactory
from baseline.manas_long_v0_4.manas_long_v0_4_config import ManasLongV04Config, ManasLongV04ModelArgs
from baseline.manas_long_v0_4.model import ManasLongV04Encoder

logger = logging.getLogger("baseline")


class ManasLongV04UnifiedModel(nn.Module):
    def __init__(self, encoder: ManasLongV04Encoder, classifier: MultiHeadClassifier, grad_cam: bool = False):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.grad_cam = grad_cam
        self.grad_cam_activation = None
        self.last_batch_context: dict | None = None

    def forward(self, batch):
        x = batch["data"]
        pos = batch["pos"]
        montage = batch["montage"][0]

        total_samples = int(x.shape[-1])
        current_window_sec = float(total_samples) / float(getattr(self.encoder.mae, "fs", 1))
        active_memory_scales = self.encoder.resolve_active_memory_scales(current_window_sec)
        self.last_batch_context = self.encoder.set_batch_context(
            active_memory_scales=active_memory_scales,
            temporal_radius_scale=None,
        )
        self.last_batch_context["current_window_sec"] = current_window_sec
        self.last_batch_context["current_window_samples"] = total_samples

        features = self.encoder(x, pos, montage=montage)
        features = features.permute(0, 2, 1, 3)

        if self.grad_cam:
            self.grad_cam_activation = features

        logits = self.classifier(features, montage)
        return logits


class ManasLongV04Trainer(ManasLongTrainer):
    def __init__(self, cfg: ManasLongV04Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.dataloader_factory = ManasLongV04DataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
            target_fs=self.cfg.fs,
            use_legacy_mne_positions=self.cfg.data.use_legacy_mne_positions,
            positions_json_path=self.cfg.data.positions_json_path,
        )
        self.encoder: Optional[ManasLongV04Encoder] = None

    def setup_model(self):
        logger.info("Setting up MANAS-Long v0.4 model architecture...")
        cfg: ManasLongV04ModelArgs = self.cfg.model

        self.encoder = ManasLongV04Encoder(cfg, fs=self.cfg.fs)

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
                        f"Dataset sample too short for MANAS-Long v0.4 patching: montage={montage_key}, "
                        f"timepoints={n_timepoints}, patch_size={patch_size}, overlap={overlap_size}"
                    )
                n_channels_eff = self.encoder.effective_num_channels(n_channels, ds_name=ds_name)
                ds_shape_info[montage_key] = (n_patches, n_channels_eff, embed_dim)
                token_count = n_channels_eff * n_patches
                pairwise_flag = self.encoder.uses_pairwise_for_dataset(ds_name)
                logger.info(
                    f"MANAS-Long v0.4 shape {montage_key}: patches={n_patches}, channels_eff={n_channels_eff}, "
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

        prediction_types = {
            info["eval"].get("prediction_type", "classification")
            for info in self.ds_info.values()
        }
        if prediction_types == {"regression"}:
            self.loss_fn = nn.MSELoss()
            logger.info("MANAS-Long v0.4 configured with MSE loss for regression targets")
        elif prediction_types == {"classification"}:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "MANAS-Long v0.4 does not support mixing prediction types in one run: "
                f"{sorted(prediction_types)}"
            )

        if not cfg.pretrained_path:
            raise ValueError(
                "MANAS-Long v0.4 requires model.pretrained_path. "
                "Training from scratch is disabled."
            )
        self.load_checkpoint(cfg.pretrained_path)

        model = ManasLongV04UnifiedModel(
            encoder=self.encoder,
            classifier=self.classifier,
            grad_cam=cfg.grad_cam,
        )

        model = self.apply_lora(model)
        model = model.to(self.device)
        model = self.maybe_wrap_ddp(model, find_unused_parameters=True)
        self.model = model

        return model

    def _set_encoder_trainability(self, method: ManasLongTrainMethod):
        if self.model is None:
            return

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        if not hasattr(model, "encoder"):
            raise AttributeError("MANAS-Long v0.4 model does not have an encoder attribute")

        encoder = model.encoder

        for param in encoder.parameters():
            param.requires_grad = False

        if method == "linear_probe":
            pass
        elif method == "memory_probe":
            self._set_memory_trainability(encoder, True, include_decoder=True)
        elif method == "partial_ft":
            for param in encoder.mae.encoder.layers[-1].parameters():
                param.requires_grad = True
            for param in encoder.mae.encoder.final_norm.parameters():
                param.requires_grad = True
            self._set_memory_trainability(encoder, True, include_decoder=False)
        elif method == "partial-enc":
            self._set_partial_encoder_trainability(encoder)
        elif method == "full_ft":
            for param in encoder.parameters():
                param.requires_grad = True
        elif method == "mem_freeze":
            for param in encoder.parameters():
                param.requires_grad = True
            self._set_memory_trainability(encoder, False, include_decoder=True)
        else:
            raise ValueError(f"Unknown train_method: {method}")

        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
