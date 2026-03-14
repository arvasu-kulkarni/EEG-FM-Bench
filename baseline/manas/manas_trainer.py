"""
MANAS trainer for EEG-FM-Bench.
"""

import collections
import glob
import logging
import os
import re
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List
from collections.abc import Iterable
import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline.abstract.classifier import MultiHeadClassifier
from baseline.abstract.config import BaseLoggingArgs
from baseline.abstract.trainer import AbstractTrainer, MASTER_RESULTS_COLUMNS
from baseline.manas.manas_adapter import ManasDataLoaderFactory
from baseline.manas.manas_config import ManasConfig, ManasModelArgs
from baseline.manas.model import ManasEncoder
from common.distributed.env import get_is_master

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

    @staticmethod
    def resolve_checkpoint_path(checkpoint_path: str, pretrained_epoch: Optional[int] = None) -> str:
        resolved = os.path.abspath(checkpoint_path)

        if os.path.isdir(resolved):
            if pretrained_epoch is not None:
                # Load a specific epoch checkpoint
                specific = os.path.join(resolved, f"mae_epoch_{pretrained_epoch}.pt")
                if not os.path.isfile(specific):
                    raise FileNotFoundError(
                        f"Requested epoch checkpoint not found: {specific}"
                    )
                logger.info(f"Resolved to specific epoch checkpoint: {specific}")
                return specific

            candidates = glob.glob(os.path.join(resolved, "mae_epoch_*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No epoch checkpoints found in directory: {resolved}. "
                    "Expected files like mae_epoch_<N>.pt"
                )

            def _epoch_of(path: str) -> int:
                match = re.search(r"mae_epoch_(\d+)\.pt$", os.path.basename(path))
                return int(match.group(1)) if match else -1

            resolved = max(candidates, key=lambda p: (_epoch_of(p), p))
            logger.info(f"Resolved checkpoint directory to latest epoch file: {resolved}")

        return resolved

    @staticmethod
    def list_epoch_checkpoints(pretrained_dir: str) -> list[int]:
        """Return sorted list of epoch numbers available in a pretrained directory."""
        resolved = os.path.abspath(pretrained_dir)
        if not os.path.isdir(resolved):
            return []
        candidates = glob.glob(os.path.join(resolved, "mae_epoch_*.pt"))
        epochs = []
        for c in candidates:
            match = re.search(r"mae_epoch_(\d+)\.pt$", os.path.basename(c))
            if match:
                epochs.append(int(match.group(1)))
        return sorted(epochs)

    @staticmethod
    def _strip_state_dict_prefix(state_dict: dict, prefixes: Iterable[str]) -> dict:
        stripped = dict(state_dict)
        for prefix in prefixes:
            if stripped and all(k.startswith(prefix) for k in stripped.keys()):
                stripped = {k[len(prefix):]: v for k, v in stripped.items()}
        return stripped

    def setup_model(self):
        logger.info("Setting up MANAS model architecture...")
        cfg: ManasModelArgs = self.cfg.model

        self.encoder = ManasEncoder(cfg, fs=self.cfg.fs)

        if cfg.mae_type == "ndx" and self.encoder.runtime_meta:
            logger.info(
                "Loaded NDX model metadata: "
                f"{self.encoder.runtime_meta}"
            )

        embed_dim = int(self.encoder.embed_dim)
        head_configs = {ds_name: info['n_class'] for ds_name, info in self.ds_info.items()}
        head_cfg = cfg.classifier_head

        patch_size = int(self.encoder.patch_size)
        overlap_size = int(patch_size - self.encoder.step)

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

        if not cfg.pretrained_path:
            raise ValueError(
                "MANAS requires model.pretrained_path. "
                "Training from scratch is disabled."
            )
        self.load_checkpoint(cfg.pretrained_path)

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
        if not checkpoint_path:
            raise ValueError(
                "MANAS checkpoint path is empty. "
                "Training from scratch is disabled."
            )
        if self.encoder is None:
            raise RuntimeError("MANAS encoder is not initialized before checkpoint loading.")

        resolved_path = self.resolve_checkpoint_path(checkpoint_path, self.cfg.model.pretrained_epoch)
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"MANAS pretrained checkpoint not found: {checkpoint_path} "
                f"(resolved: {resolved_path})"
            )

        logger.info(f"Loading pretrained weights from: {resolved_path}")
        try:
            ckpt = torch.load(resolved_path, map_location='cpu', weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to read MANAS checkpoint: {resolved_path}") from exc

        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ('model_state_dict', 'state_dict', 'model', 'mae_state_dict'):
                nested = ckpt.get(key)
                if isinstance(nested, dict):
                    state_dict = nested
                    break

        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Unsupported MANAS checkpoint format at {resolved_path}: "
                f"expected dict-like state_dict, got {type(state_dict).__name__}"
            )

        candidate_state_dicts = [
            state_dict,
            self._strip_state_dict_prefix(state_dict, ("module.",)),
            self._strip_state_dict_prefix(state_dict, ("mae.",)),
            self._strip_state_dict_prefix(state_dict, ("module.", "mae.")),
        ]
        best_result: tuple[list[str], list[str]] | None = None

        for candidate in candidate_state_dicts:
            missing, unexpected = self.encoder.mae.load_state_dict(candidate, strict=False)
            if not missing and not unexpected:
                best_result = (missing, unexpected)
                break
            if best_result is None or (len(missing) + len(unexpected) < len(best_result[0]) + len(best_result[1])):
                best_result = (missing, unexpected)

        if best_result is None or best_result[0] or best_result[1]:
            missing, unexpected = best_result if best_result is not None else ([], [])
            raise RuntimeError(
                "MANAS checkpoint is incompatible with current encoder architecture. "
                f"missing_keys={missing}, unexpected_keys={unexpected}"
            )

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

    # ------------------------------------------------------------------ #
    #  Sweep-mode overrides: flat directory layout
    # ------------------------------------------------------------------ #

    def setup_logging(self):
        """In sweep mode, skip creating manas_trainer.log file."""
        if not self.cfg.sweep_mode:
            return super().setup_logging()

        log_dir, ckpt_dir = self.get_train_io_path(self.cfg.logging)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ckpt_dir = self.broadcast_str(ckpt_dir, max_length=512, rank=self.rank)
            log_dir = self.broadcast_str(log_dir, max_length=512, rank=self.rank)

        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        # No log file in sweep mode — stdout/stderr goes to sweep.log via shell redirect
        logger.info(f"[sweep] log dir: {self.log_dir}, checkpoint dir: {self.ckpt_dir}")
        logger.info(f"Starting {self.cfg.model_type} training with "
                     f"{self.num_ds} dataset(s): {list(self.ds_conf.keys())}")

    def init_cloud_logging(self):
        """In sweep mode, override wandb run name to include model + method."""
        if not self.cfg.sweep_mode:
            return super().init_cloud_logging()

        if not self.cfg.logging.use_cloud or not get_is_master():
            return

        backend = self.cfg.logging.cloud_backend.lower()
        if backend in ['none', 'off', 'disabled']:
            return

        if backend in ['wandb', 'both']:
            self._init_sweep_wandb()

        if backend in ['comet', 'both']:
            self._init_comet()

    def _init_sweep_wandb(self):
        """Initialize wandb with descriptive run name: e.g. 20260227_093832_epoch3_ft"""
        try:
            wandb = self._safe_import_wandb()
            if wandb is None:
                return

            # Build descriptive name from experiment_name
            # experiment_name is like "manas_ndx_20260225_221743_epoch1_ft"
            run_name = self.cfg.logging.experiment_name

            wandb_dir = os.path.join(self.cfg.logging.run_dir, 'wandb')
            os.makedirs(wandb_dir, exist_ok=True)

            wandb_config = {
                'dir': wandb_dir,
                'project': self.cfg.logging.project or self.cfg.logging.experiment_name,
                'name': run_name,
                'config': self.cfg.model_dump(),
                'tags': self.cfg.logging.tags,
                'mode': 'offline' if self.cfg.logging.offline else 'online',
            }

            if self.cfg.logging.entity:
                wandb_config['entity'] = self.cfg.logging.entity

            if self.cfg.logging.api_key:
                os.environ['WANDB_API_KEY'] = self.cfg.logging.api_key

            wandb.init(**wandb_config)

            # Define step metrics
            wandb_metrics = []
            for ds_name in self.ds_conf.keys():
                if not self.multitask:
                    wandb_metrics.append(f"{ds_name}/train/step")
                wandb_metrics.extend([
                    f"{ds_name}/eval/epoch",
                    f"{ds_name}/test/epoch"
                ])

            for metric in wandb_metrics:
                idx = metric.rfind('/')
                step_metric = metric[:idx] + '/step' if '/train/' in metric else metric[:idx] + '/epoch'
                wandb.define_metric(metric, step_metric=step_metric)

            logger.info(f"[sweep] wandb initialized: project={wandb_config['project']}, name={run_name}")

        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")

    def get_train_io_path(self, args: BaseLoggingArgs) -> tuple[str, str]:
        """In sweep mode, use run_dir directly as log and ckpt root (flat layout)."""
        if not self.cfg.sweep_mode:
            return super().get_train_io_path(args)

        if not get_is_master():
            return '', ''

        run_dir = args.run_dir
        log_path = run_dir
        ckpt_path = os.path.join(run_dir, 'ckpt')

        os.makedirs(log_path, exist_ok=True)
        os.makedirs(ckpt_path, exist_ok=True)

        return log_path, ckpt_path

    def _run_results_dir(self) -> Path:
        """In sweep mode, put results directly in run_dir (no nested best_results/)."""
        if not self.cfg.sweep_mode:
            return super()._run_results_dir()
        if self.log_dir:
            return Path(self.log_dir)
        return Path(self.cfg.logging.run_dir)

    def save_checkpoint(
            self,
            ds_name: Optional[str] = None,
            is_milestone: bool = False,
            **kwargs
    ):
        """In sweep mode, save to ckpt/<ds_name>/ (flat, no 'seperated' level).
        If save_checkpoints=False, skip saving entirely."""
        if not self.cfg.save_checkpoints:
            return

        if not self.cfg.sweep_mode:
            return super().save_checkpoint(ds_name=ds_name, is_milestone=is_milestone, **kwargs)

        if not get_is_master():
            return

        rep_parts = [f"rep_{self.current_repetition:02d}"] if self.num_repetitions > 1 else []

        if ds_name is None:
            ds_name = 'unified'
        checkpoint_dir = Path(self.ckpt_dir, *rep_parts, ds_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.cfg.model_dump(mode='json'),
            'dataset_name': ds_name,
        }

        suffix = 'last' if is_milestone else f'epoch_{self.epoch}'
        checkpoint_path = checkpoint_dir / f'{self.model_type}_{ds_name}_{suffix}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {ds_name}: {checkpoint_path}")

        if self.cfg.training.lora.use_lora:
            self.save_lora_checkpoint(checkpoint_dir, ds_name, suffix)

    # ------------------------------------------------------------------ #
    #  Sweep-mode model-level aggregation (majority-vote best epoch)
    # ------------------------------------------------------------------ #

    def eval_epoch(self, dataloaders: list[DataLoader], prefix: str):
        """Run parent eval_epoch, then update sweep-level aggregation if applicable."""
        result = super().eval_epoch(dataloaders, prefix)
        if self.cfg.sweep_mode and self.cfg.sweep_root and prefix == 'test':
            try:
                self._update_sweep_model_results()
            except Exception as exc:
                logger.warning(f"Failed to update sweep model results: {exc}")
        return result

    def _update_sweep_model_results(self):
        """Aggregate all epoch-level best_test_results.txt into model-level best_results.txt."""
        if not get_is_master():
            return

        sweep_root = Path(self.cfg.sweep_root)
        if not sweep_root.is_dir():
            return

        # Discover all epoch-level result files
        # Expected layout: <sweep_root>/epoch_<N>/{ft,lp}/best_test_results.txt
        records = self._collect_sweep_records(sweep_root)
        if not records:
            return

        # Separate by method (ft / lp)
        for method in ['ft', 'lp']:
            method_records = [r for r in records if r['method'] == method]
            if not method_records:
                continue
            self._write_sweep_summary(sweep_root, method, method_records)

        # Also write a combined summary
        self._write_sweep_summary(sweep_root, 'combined', records)

    @staticmethod
    def _collect_sweep_records(sweep_root: Path) -> list[dict]:
        """Read all epoch-level best_test_results.txt files under sweep_root."""
        records = []
        for epoch_dir in sorted(sweep_root.glob('epoch_*')):
            epoch_match = re.match(r'epoch_(\d+)$', epoch_dir.name)
            if not epoch_match:
                continue
            pretrained_epoch = int(epoch_match.group(1))

            for method in ['ft', 'lp']:
                results_file = epoch_dir / method / 'best_test_results.txt'
                if not results_file.is_file():
                    continue

                rows = ManasTrainer._parse_best_test_results(results_file)
                for row in rows:
                    row['pretrained_epoch'] = pretrained_epoch
                    row['method'] = method
                    records.append(row)
        return records

    @staticmethod
    def _parse_best_test_results(path: Path) -> list[dict]:
        """Parse a best_test_results.txt TSV file into list of dicts."""
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if parts == MASTER_RESULTS_COLUMNS:
                    continue  # skip header
                if len(parts) != len(MASTER_RESULTS_COLUMNS):
                    continue
                row = dict(zip(MASTER_RESULTS_COLUMNS, parts))
                rows.append(row)
        return rows

    @staticmethod
    def _write_sweep_summary(sweep_root: Path, method: str, records: list[dict]):
        """Write model-level summary with majority-vote best epoch selection."""
        import datetime

        # Group: dataset -> list of (pretrained_epoch, score, row)
        ds_best: Dict[str, List[tuple[int, float, dict]]] = collections.defaultdict(list)
        for r in records:
            ds_name = r.get('dataset', '')
            try:
                score = float(r.get('score', '-inf'))
            except (ValueError, TypeError):
                score = float('-inf')
            ds_best[ds_name].append((r['pretrained_epoch'], score, r))

        # For each dataset, find which pretrained epoch had the best score
        epoch_wins: Dict[int, int] = collections.Counter()
        epoch_scores: Dict[int, list[float]] = collections.defaultdict(list)
        ds_winner: Dict[str, tuple[int, float, dict]] = {}

        for ds_name, entries in ds_best.items():
            best_entry = max(entries, key=lambda e: e[1])
            ds_winner[ds_name] = best_entry
            epoch_wins[best_entry[0]] += 1
            for ep, sc, _ in entries:
                epoch_scores[ep].append(sc)

        # Majority vote: epoch with most wins; tiebreak by avg score
        if not epoch_wins:
            return

        best_epoch = max(
            epoch_wins.keys(),
            key=lambda ep: (
                epoch_wins[ep],
                sum(epoch_scores.get(ep, [])) / max(len(epoch_scores.get(ep, [])), 1)
            )
        )

        # Write summary
        out_path = sweep_root / f'best_results_{method}.txt'
        tmp_path = out_path.with_suffix('.tmp')
        updated_utc = datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z'

        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(f"# EEG-FM-Bench Sweep Model-Level Best Results ({method.upper()})\n")
            f.write(f"# Updated: {updated_utc}\n")
            f.write(f"# Sweep root: {sweep_root}\n")
            f.write(f"#\n")
            f.write(f"# === MAJORITY-VOTE BEST EPOCH: {best_epoch} ===")
            f.write(f"  (wins: {epoch_wins[best_epoch]}/{sum(epoch_wins.values())} downstream tasks)\n")
            avg_score = sum(epoch_scores.get(best_epoch, [])) / max(len(epoch_scores.get(best_epoch, [])), 1)
            f.write(f"# Average score at epoch {best_epoch}: {avg_score:.4f}\n")
            f.write(f"#\n")

            # Vote summary
            f.write(f"# --- Epoch vote counts ---\n")
            for ep in sorted(epoch_wins.keys()):
                ep_avg = sum(epoch_scores.get(ep, [])) / max(len(epoch_scores.get(ep, [])), 1)
                f.write(f"#   epoch_{ep}: {epoch_wins[ep]} wins, avg_score={ep_avg:.4f}\n")
            f.write(f"#\n")

            # Per-dataset detail
            f.write(f"# --- Per-dataset best pretrained epoch ---\n")
            f.write(f"# {'dataset':<20s} {'best_pretrained_epoch':>22s} {'score':>10s} {'ft_epoch':>10s}\n")
            for ds_name in sorted(ds_winner.keys()):
                ep, sc, row = ds_winner[ds_name]
                ft_epoch = row.get('epoch', '')
                f.write(f"# {ds_name:<20s} {ep:>22d} {sc:>10.4f} {ft_epoch:>10s}\n")
            f.write(f"#\n")

            # Full TSV of all records for this method
            f.write("\t".join(['pretrained_epoch'] + MASTER_RESULTS_COLUMNS) + "\n")
            for r in sorted(records, key=lambda x: (x['pretrained_epoch'], x.get('dataset', ''))):
                f.write(str(r['pretrained_epoch']) + "\t")
                f.write("\t".join(r.get(col, '') for col in MASTER_RESULTS_COLUMNS) + "\n")

        os.replace(tmp_path, out_path)
        logger.info(f"[Sweep] Updated model-level results: {out_path} (best_epoch={best_epoch})")
