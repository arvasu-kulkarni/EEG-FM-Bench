"""
MANAS-Long trainer for EEG-FM-Bench.
"""

import logging
import os
from typing import Optional

import torch
from torch import nn

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.trainer import AbstractTrainer
from baseline.manas_long.manas_long_adapter import ManasLongDataLoaderFactory
from baseline.manas_long.manas_long_config import ManasLongConfig, ManasLongModelArgs, ManasLongTrainMethod
from baseline.manas_long.model import ManasLongEncoder
from baseline.utils.lora import freeze_non_lora_parameters, set_lora_trainable

logger = logging.getLogger("baseline")


class ManasLongUnifiedModel(nn.Module):
    def __init__(self, encoder: ManasLongEncoder, classifier: MultiHeadClassifier, grad_cam: bool = False):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

        self.grad_cam = grad_cam
        self.grad_cam_activation = None

    def forward(self, batch):
        x = batch["data"]
        pos = batch["pos"]
        montage = batch["montage"][0]

        features = self.encoder(x, pos, montage=montage)
        features = features.permute(0, 2, 1, 3)

        if self.grad_cam:
            self.grad_cam_activation = features

        logits = self.classifier(features, montage)
        return logits


class ManasLongTrainer(AbstractTrainer):
    def __init__(self, cfg: ManasLongConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.dataloader_factory = ManasLongDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
            target_fs=self.cfg.fs,
            use_legacy_mne_positions=self.cfg.data.use_legacy_mne_positions,
            positions_json_path=self.cfg.data.positions_json_path,
        )

        self.encoder: Optional[ManasLongEncoder] = None
        self.classifier: Optional[MultiHeadClassifier] = None
        self.loss_fn = None
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
        logger.info("Setting up MANAS-Long model architecture...")
        cfg: ManasLongModelArgs = self.cfg.model

        self.encoder = ManasLongEncoder(cfg, fs=self.cfg.fs)

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
                        f"Dataset sample too short for MANAS-Long patching: montage={montage_key}, timepoints={n_timepoints}, "
                        f"patch_size={patch_size}, overlap={overlap_size}"
                    )
                n_channels_eff = self.encoder.effective_num_channels(n_channels, ds_name=ds_name)
                ds_shape_info[montage_key] = (n_patches, n_channels_eff, embed_dim)
                token_count = n_channels_eff * n_patches
                pairwise_flag = self.encoder.uses_pairwise_for_dataset(ds_name)
                logger.info(
                    f"MANAS-Long shape {montage_key}: patches={n_patches}, channels_eff={n_channels_eff}, "
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
            info['eval'].get('prediction_type', 'classification')
            for info in self.ds_info.values()
        }
        if prediction_types == {'regression'}:
            self.loss_fn = nn.MSELoss()
            logger.info("MANAS-Long configured with MSE loss for regression targets")
        elif prediction_types == {'classification'}:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"MANAS-Long does not support mixing prediction types in one run: {sorted(prediction_types)}"
            )

        if not cfg.pretrained_path:
            raise ValueError(
                "MANAS-Long requires model.pretrained_path. "
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

    def load_checkpoint(self, checkpoint_path: str):
        if not checkpoint_path:
            raise ValueError(
                "MANAS-Long checkpoint path is empty. "
                "Training from scratch is disabled."
            )
        if self.encoder is None:
            raise RuntimeError("MANAS-Long encoder is not initialized before checkpoint loading.")

        resolved_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"MANAS-Long pretrained checkpoint not found: {checkpoint_path} "
                f"(resolved: {resolved_path})"
            )

        logger.info(f"Loading pretrained weights from: {resolved_path}")
        try:
            ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to read MANAS-Long checkpoint: {resolved_path}") from exc

        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict", "model", "mae_state_dict"):
                nested = ckpt.get(key)
                if isinstance(nested, dict):
                    state_dict = nested
                    break

        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Unsupported MANAS-Long checkpoint format at {resolved_path}: "
                f"expected dict-like state_dict, got {type(state_dict).__name__}"
            )

        remapped: dict[str, torch.Tensor] = {}
        stripped_prefix_count = 0
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("module.", "mae."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    stripped_prefix_count += 1
            remapped[new_key] = value

        if stripped_prefix_count > 0:
            logger.info(
                "Applied MANAS-Long checkpoint prefix remap: "
                f"stripped outer prefixes from {stripped_prefix_count} keys."
            )

        encoder_state = self.encoder.mae.state_dict()
        filtered = {key: value for key, value in remapped.items() if key in encoder_state}
        skipped = sorted(key for key in remapped if key not in encoder_state)
        if skipped:
            logger.warning("MANAS-Long checkpoint keys skipped before strict load:")
            for key in skipped:
                logger.warning(f"  SKIPPED: {key}")

        if not filtered:
            raise RuntimeError(
                "MANAS-Long checkpoint did not contain any encoder-side keys matching the downstream backbone."
            )

        missing, unexpected = self.encoder.mae.load_state_dict(filtered, strict=False)

        allowed_missing: set[str] = set()
        if not getattr(self.encoder.mae, "use_memory_tokens", True):
            allowed_missing = {
                key
                for key in missing
                if ".memory_branch." in key or key == "decoder.pre_memory_norm.weight"
            }
            if allowed_missing:
                logger.info(
                    "Allowing missing MANAS-Long memory-specific keys because use_memory_tokens is disabled: "
                    f"{sorted(allowed_missing)}"
                )

        missing = [key for key in missing if key not in allowed_missing]

        if missing:
            logger.error("MANAS-Long encoder missing keys after strict load:")
            for key in sorted(missing):
                logger.error(f"  MISSING: {key}")
        if unexpected:
            logger.error("MANAS-Long encoder unexpected keys after strict load:")
            for key in sorted(unexpected):
                logger.error(f"  UNEXPECTED: {key}")
            raise RuntimeError(
                "MANAS-Long checkpoint load produced unexpected keys after filtering: "
                f"{unexpected}"
            )
        if missing:
            raise RuntimeError(
                "MANAS-Long checkpoint is incompatible with current encoder architecture. "
                f"missing_keys={missing}"
            )

        logger.info(
            "Successfully loaded pretrained MANAS-Long encoder weights "
            f"({len(filtered)}/{len(remapped)} checkpoint keys used)"
        )

    @staticmethod
    def _set_module_trainable(module: Optional[nn.Module], trainable: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = trainable

    @staticmethod
    def _set_parameter_trainable(param: Optional[nn.Parameter], trainable: bool):
        if isinstance(param, nn.Parameter):
            param.requires_grad = trainable

    @staticmethod
    def _get_encoder_layers(encoder: nn.Module) -> nn.ModuleList:
        mae = getattr(encoder, "mae", None)
        encoder_core = getattr(mae, "encoder", None) if mae is not None else None
        layers = getattr(encoder_core, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise AttributeError("MANAS-Long encoder does not expose encoder layers via encoder.mae.encoder.layers")
        return layers

    def _set_memory_trainability(
        self,
        encoder: nn.Module,
        trainable: bool,
        *,
        include_decoder: bool,
    ):
        mae = getattr(encoder, "mae", None)
        if mae is None:
            return

        encoder_core = getattr(mae, "encoder", None)
        decoder_core = getattr(mae, "decoder", None) if include_decoder else None

        self._set_module_trainable(getattr(encoder_core, "memory_branch", None), trainable)
        self._set_module_trainable(getattr(encoder_core, "memory_branches", None), trainable)
        self._set_parameter_trainable(getattr(encoder_core, "memory_scale_alpha", None), trainable)

        self._set_module_trainable(getattr(mae, "memory_patch_embeds", None), trainable)
        self._set_module_trainable(getattr(mae, "memory_token_modules", None), trainable)

        if include_decoder:
            self._set_module_trainable(getattr(decoder_core, "memory_branch", None), trainable)
            self._set_module_trainable(getattr(decoder_core, "memory_branches", None), trainable)
            self._set_module_trainable(getattr(decoder_core, "pre_memory_norm", None), trainable)
            self._set_parameter_trainable(getattr(decoder_core, "memory_scale_alpha", None), trainable)

    def _effective_train_method(self) -> ManasLongTrainMethod:
        method: ManasLongTrainMethod = self.cfg.training.train_method
        if self.cfg.training.dual_stage:
            switch_epoch = self.cfg.training.max_epochs // 2
            if self.epoch < switch_epoch:
                return "linear_probe"
        return method

    def _set_partial_encoder_trainability(self, encoder: nn.Module):
        layers = self._get_encoder_layers(encoder)
        num_layers = len(layers)
        enc_layer = self.cfg.training.enc_layer
        freeze_direction = self.cfg.training.freeze
        freeze_memory = self.cfg.training.freeze_memory

        if enc_layer is None or freeze_direction is None:
            raise ValueError("train_method='partial-enc' requires both training.enc_layer and training.freeze.")
        if not 1 <= enc_layer <= num_layers:
            raise ValueError(
                f"training.enc_layer={enc_layer} is out of range for encoder depth {num_layers}. "
                f"Expected a 1-based layer index between 1 and {num_layers}."
            )

        for param in encoder.parameters():
            param.requires_grad = True

        for layer_idx, layer in enumerate(layers, start=1):
            freeze_layer = layer_idx > enc_layer if freeze_direction == "after" else layer_idx < enc_layer
            if freeze_layer:
                for param in layer.parameters():
                    param.requires_grad = False

        if freeze_memory:
            self._set_memory_trainability(encoder, False, include_decoder=True)

    def _set_encoder_trainability(self, method: ManasLongTrainMethod):
        if self.model is None:
            return

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        if not hasattr(model, "encoder"):
            raise AttributeError("MANAS-Long model does not have an encoder attribute")

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

    def setup_optimizer_and_scheduler(self, model, train_loader):
        original_freeze_encoder = self.cfg.training.freeze_encoder
        self.cfg.training.freeze_encoder = False
        try:
            super().setup_optimizer_and_scheduler(model, train_loader)
        finally:
            self.cfg.training.freeze_encoder = original_freeze_encoder

    def train_epoch(self, train_loader, train_sampler):
        if self.cfg.training.lora.use_lora:
            model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            freeze_non_lora_parameters(model)
            set_lora_trainable(model, True)
            if hasattr(model, "classifier"):
                for param in model.classifier.parameters():
                    param.requires_grad = True

            self.cfg.training.freeze_encoder = True
            effective_method = f"lora::{self.cfg.training.train_method}"
        else:
            effective_method = self._effective_train_method()
            self._set_encoder_trainability(effective_method)
            self.cfg.training.freeze_encoder = (effective_method == "linear_probe")

        if effective_method != self._last_effective_train_method:
            logger.info(
                f"Epoch {self.epoch}: applying train_method='{effective_method}' "
                f"(configured='{self.cfg.training.train_method}', dual_stage={self.cfg.training.dual_stage}, "
                f"use_lora={self.cfg.training.lora.use_lora})"
            )
            self._last_effective_train_method = effective_method

        super().train_epoch(train_loader, train_sampler)
