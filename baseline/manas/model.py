"""
MANAS encoder wrapper for EEG-FM-Bench.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any

import torch
from torch import nn

from baseline.manas.manas_config import ManasModelArgs
from baseline.manas.manasfiles.model import MAE
from baseline.manas.manasfiles.mahirmodel import MAE as MahirMAE


def _resolve_run_dir(pretrained_path: str | None) -> str | None:
    if not pretrained_path:
        return None
    resolved = os.path.abspath(pretrained_path)
    if os.path.isdir(resolved):
        return resolved
    return os.path.dirname(resolved)


def _load_ndx_run_metadata(
    pretrained_path: str | None,
    run_config_path: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    candidates: list[str] = []

    if run_config_path:
        candidates.append(os.path.abspath(run_config_path))

    run_dir = _resolve_run_dir(pretrained_path)
    if run_dir:
        candidates.append(os.path.join(run_dir, "run_config.json"))

    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue

        with open(candidate, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid run_config payload at {candidate}: expected object")

        run_cfg = payload.get("config", {})
        ablation_cfg = payload.get("ablation_cfg", {})

        if not isinstance(run_cfg, dict):
            raise ValueError(f"Invalid run_config['config'] at {candidate}: expected object")
        if not isinstance(ablation_cfg, dict):
            raise ValueError(f"Invalid run_config['ablation_cfg'] at {candidate}: expected object")

        return run_cfg, ablation_cfg, candidate

    return {}, {}, None


def _load_external_mae_class(model_py_path: str):
    resolved = os.path.abspath(model_py_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"External NDX model.py not found: {resolved}")

    # Keep external model directory on sys.path so runtime local imports
    # inside the external MAE (e.g. `from stft_loss import ...`) resolve.
    module_dir = os.path.dirname(resolved)
    if module_dir and module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("ndx_external_mae_module", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import external model module from: {resolved}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mae_cls = getattr(module, "MAE", None)
    if mae_cls is None:
        raise AttributeError(f"External module does not expose MAE class: {resolved}")

    return mae_cls


class ManasEncoder(nn.Module):
    """Encoder that exposes MAE features in (B, C, T, E) format."""

    def __init__(self, cfg: ManasModelArgs, fs: int):
        super().__init__()

        self.runtime_meta: dict[str, Any] = {}

        if cfg.mae_type == "ndx":
            if not cfg.external_model_py:
                raise ValueError(
                    "For model.mae_type='ndx', you must set model.external_model_py to the source model.py path."
                )

            run_cfg, ablation_cfg, run_cfg_used = _load_ndx_run_metadata(
                cfg.pretrained_path,
                cfg.ndx_run_config_path,
            )

            mae_cls = _load_external_mae_class(cfg.external_model_py)

            fs_eff = int(run_cfg.get("fs", fs))
            patch_seconds = float(run_cfg.get("patch_seconds", cfg.patch_seconds))
            overlap_seconds = float(run_cfg.get("overlap_seconds", cfg.overlap_seconds))
            embed_dim = int(run_cfg.get("embed_dim", cfg.embed_dim))
            encoder_depth = int(run_cfg.get("encoder_depth", cfg.encoder_depth))
            encoder_heads = int(run_cfg.get("encoder_heads", cfg.encoder_heads))
            decoder_depth = int(run_cfg.get("decoder_depth", cfg.decoder_depth))
            decoder_heads = int(run_cfg.get("decoder_heads", cfg.decoder_heads))
            mask_ratio = float(run_cfg.get("mask_ratio", cfg.mask_ratio))
            aux_loss_weight = float(ablation_cfg.get("aux_loss_weight", cfg.aux_loss_weight))

            self.mae = mae_cls(
                fs=fs_eff,
                patch_seconds=patch_seconds,
                overlap_seconds=overlap_seconds,
                embed_dim=embed_dim,
                encoder_depth=encoder_depth,
                encoder_heads=encoder_heads,
                decoder_depth=decoder_depth,
                decoder_heads=decoder_heads,
                mask_ratio=mask_ratio,
                aux_loss_weight=aux_loss_weight,
                ablation_cfg=ablation_cfg,
            )

            self.runtime_meta = {
                "run_cfg_used": run_cfg_used,
                "fs": fs_eff,
                "embed_dim": embed_dim,
                "encoder_depth": encoder_depth,
                "encoder_heads": encoder_heads,
                "decoder_depth": decoder_depth,
                "decoder_heads": decoder_heads,
                "mask_ratio": mask_ratio,
                "aux_loss_weight": aux_loss_weight,
            }
        else:
            mae_cls = MahirMAE if cfg.mae_type == "mahir" else MAE
            self.mae = mae_cls(
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
            )

        self.patch_size = int(self.mae.patch_size)
        self.step = int(self.mae.step)
        self.embed_dim = int(getattr(self.mae, "embed_dim", cfg.embed_dim))

    def forward(self, eeg: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        eeg = eeg.float()
        patches = eeg.unfold(dimension=-1, size=self.patch_size, step=self.step)
        b, c, n_patches, _ = patches.shape

        tokens = self.mae.patch_embed.linear(patches).flatten(1, 2)
        coords = self.mae.prepare_coords(pos, n_patches)
        pos_emb = self.mae.pos_enc(coords)

        x = tokens + pos_emb
        encoded = self.mae.encoder(x)
        if isinstance(encoded, tuple):
            x = encoded[0]
        else:
            x = encoded

        x = x.reshape(b, c, n_patches, self.embed_dim)
        return x
