"""
MANAS adapter for EEG-FM-Bench.
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional

import mne
import numpy as np
import torch
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory
from common.path import PROJECT_ROOT
from common.utils import ElectrodeSet

logger = logging.getLogger("baseline")


IGNORE_CHANS = {
    'TPP9H', 'TPP10H', 'AFF1', 'AFF2', 'FFC5H', 'FFC3H', 'FFC4H', 'FFC6H', 'FCC5H', 'FCC3H',
    'FCC4H', 'FCC6H', 'CCP5H', 'CCP3H', 'CCP4H', 'CCP6H', 'CPP5H', 'CPP3H', 'CPP4H', 'CPP6H',
    'PPO1', 'PPO2', 'I1', 'I2', 'AFP3H', 'AFP4H', 'AFF5H', 'AFF6H', 'FFT7H', 'FFC1H', 'FFC2H',
    'FFT8H', 'FTT9H', 'FTT7H', 'FCC1H', 'FCC2H', 'FTT8H', 'FTT10H', 'TTP7H', 'CCP1H', 'CCP2H',
    'TTP8H', 'TPP7H', 'CPP1H', 'CPP2H', 'TPP8H', 'PPO9H', 'PPO5H', 'PPO6H', 'PPO10H', 'POO9H',
    'POO3H', 'POO4H', 'POO10H', 'OI1H', 'OI2H',
}


class ManasDatasetAdapter(AbstractDatasetAdapter):
    """MANAS dataset adapter that attaches channel positions from MNE montage."""
    MNE_POSITION_SCALE: float = 100.0
    JSON_POSITION_SCALE: float = 1.0

    def __init__(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
        target_fs: int = 200,
        use_legacy_mne_positions: bool = True,
        positions_json_path: Optional[str] = None,
    ):
        self.electrode_set: ElectrodeSet = ElectrodeSet()
        self.target_fs = target_fs
        self.use_legacy_mne_positions = bool(use_legacy_mne_positions)
        self.positions_json_path = positions_json_path
        self._position_cache: Dict[tuple[str, ...], torch.Tensor] = {}
        self._missing_json_pos_warned: set[tuple[str, ...]] = set()
        self._json_positions: Dict[str, torch.Tensor] = {}
        if not self.use_legacy_mne_positions:
            self._json_positions = self._load_positions_json(positions_json_path)
        super().__init__(dataset, dataset_names, dataset_configs)

    def _setup_adapter(self):
        self.model_name = 'manas'
        self._build_montage_mappings()
        self._log_adapter_info()

    def get_supported_channels(self) -> List[str]:
        return self.electrode_set.Electrodes

    @staticmethod
    def _normalize_channel_name(channel_name: str) -> str:
        name = channel_name.strip().upper()
        if name.startswith("EEG "):
            name = name[4:]
        name = name.replace(" ", "").replace("-", "").replace("_", "")
        return name

    @staticmethod
    def _resolve_positions_json_path(positions_json_path: Optional[str]) -> str:
        default_path = os.path.join(PROJECT_ROOT, "positions.json")
        if positions_json_path is None:
            if not os.path.isfile(default_path):
                raise FileNotFoundError(f"positions.json not found at default path: {default_path}")
            return default_path

        if os.path.isabs(positions_json_path):
            resolved = positions_json_path
            if not os.path.isfile(resolved):
                raise FileNotFoundError(f"positions_json_path does not exist: {resolved}")
            return resolved

        cwd_candidate = os.path.abspath(positions_json_path)
        if os.path.isfile(cwd_candidate):
            return cwd_candidate

        project_candidate = os.path.join(PROJECT_ROOT, positions_json_path)
        if os.path.isfile(project_candidate):
            return project_candidate

        raise FileNotFoundError(
            f"Relative positions_json_path not found: {positions_json_path} "
            f"(checked {cwd_candidate} and {project_candidate})"
        )

    def _load_positions_json(self, positions_json_path: Optional[str]) -> Dict[str, torch.Tensor]:
        resolved_path = self._resolve_positions_json_path(positions_json_path)
        with open(resolved_path, "r", encoding="utf-8") as fp:
            raw_obj = json.load(fp)

        if isinstance(raw_obj, dict) and "positions" in raw_obj and isinstance(raw_obj["positions"], dict):
            raw_obj = raw_obj["positions"]

        if not isinstance(raw_obj, dict):
            raise ValueError(f"Unsupported positions JSON format in {resolved_path}: expected object mapping.")

        parsed: Dict[str, torch.Tensor] = {}
        for ch_name, coords in raw_obj.items():
            if not isinstance(ch_name, str):
                continue
            if not isinstance(coords, (list, tuple)) or len(coords) < 3:
                continue

            try:
                xyz = torch.tensor([float(coords[0]), float(coords[1]), float(coords[2])], dtype=torch.float32)
            except (TypeError, ValueError):
                continue

            # Keep JSON coordinates as-is (no scaling).
            parsed[self._normalize_channel_name(ch_name)] = self.JSON_POSITION_SCALE * xyz

        if not parsed:
            raise ValueError(f"No valid channel coordinates found in {resolved_path}")

        logger.info(f"MANAS loaded {len(parsed)} channel positions from {resolved_path}")
        return parsed

    def _make_positions_mne(self, channel_names: List[str]) -> torch.Tensor:
        cache_key = tuple(ch.upper() for ch in channel_names)
        cached = self._position_cache.get(cache_key)
        if cached is not None:
            return cached

        mne_info = mne.create_info(list(cache_key), sfreq=100, ch_types="eeg")
        raw_obj = mne.io.RawArray(np.zeros((len(channel_names), 100)), mne_info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_obj.set_montage(montage, match_case=False, on_missing="raise")
        pos = raw_obj.get_montage().get_positions()["ch_pos"].values()
        pos_np = np.asarray(list(pos), dtype=np.float32)
        # Legacy MNE path uses scaled coordinates for historical compatibility.
        positions = self.MNE_POSITION_SCALE * torch.from_numpy(pos_np)
        self._position_cache[cache_key] = positions
        return positions

    def _make_positions_json(self, channel_names: List[str]) -> Optional[torch.Tensor]:
        cache_key = tuple(ch.upper() for ch in channel_names)
        cached = self._position_cache.get(cache_key)
        if cached is not None:
            return cached

        coords: List[torch.Tensor] = []
        missing: List[str] = []
        for ch_name in channel_names:
            key = self._normalize_channel_name(ch_name)
            pos = self._json_positions.get(key)
            if pos is None:
                missing.append(ch_name)
            else:
                coords.append(pos)

        if missing:
            missing_key = tuple(m.upper() for m in missing)
            if missing_key not in self._missing_json_pos_warned:
                logger.warning(
                    "MANAS positions.json missing channels %s; falling back to legacy MNE generation for this montage.",
                    missing,
                )
                self._missing_json_pos_warned.add(missing_key)
            return None

        positions = torch.stack(coords, dim=0).to(torch.float32)
        self._position_cache[cache_key] = positions
        return positions

    def _make_positions(self, channel_names: List[str]) -> torch.Tensor:
        if self.use_legacy_mne_positions:
            return self._make_positions_mne(channel_names)

        json_positions = self._make_positions_json(channel_names)
        if json_positions is not None:
            return json_positions

        # Keep compatibility when JSON does not include some channels.
        return self._make_positions_mne(channel_names)

    def _resample(self, data: torch.Tensor, orig_fs: int | None) -> torch.Tensor:
        if orig_fs is None or orig_fs == self.target_fs:
            return data

        data_np = data.detach().cpu().numpy().astype(np.float64, copy=False)
        if orig_fs < self.target_fs:
            resampled = mne.filter.resample(data_np, up=self.target_fs / orig_fs, verbose=False)
        else:
            resampled = mne.filter.resample(data_np, down=orig_fs / self.target_fs, verbose=False)

        return torch.as_tensor(resampled, dtype=torch.float32)

    @staticmethod
    def _zscore(data: torch.Tensor) -> torch.Tensor:
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
        return (data - mean) / (std + 1e-6)

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        result = super()._process_sample(sample)

        ch_indices = result['chs'].tolist()
        channel_names = self.electrode_set.get_electrodes_name(ch_indices)
        keep_mask = [name.upper() not in IGNORE_CHANS for name in channel_names]

        if not any(keep_mask):
            raise ValueError("No channels left after applying MANAS ignore list")

        keep_mask_np = np.asarray(keep_mask, dtype=np.bool_)
        keep_tensor = torch.from_numpy(keep_mask_np)
        data = result['data'][keep_tensor, :]

        orig_fs = sample.get('fs') or sample.get('sampling_frequency')
        if isinstance(orig_fs, torch.Tensor):
            orig_fs = int(orig_fs.item())
        elif isinstance(orig_fs, np.ndarray):
            orig_fs = int(orig_fs.item())

        data = self._resample(data, orig_fs)
        data = self._zscore(data)

        kept_names = [name for name, keep in zip(channel_names, keep_mask) if keep]
        result['pos'] = self._make_positions(kept_names)

        result['data'] = data

        result.pop('chans_id')
        return result


class ManasDataLoaderFactory(AbstractDataLoaderFactory):
    """MANAS DataLoader factory that attaches position information."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 42,
        target_fs: int = 200,
        use_legacy_mne_positions: bool = True,
        positions_json_path: Optional[str] = None,
    ):
        super().__init__(batch_size, num_workers, seed)
        self.target_fs = target_fs
        self.use_legacy_mne_positions = bool(use_legacy_mne_positions)
        self.positions_json_path = positions_json_path

    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> ManasDatasetAdapter:
        return ManasDatasetAdapter(
            dataset,
            dataset_names,
            dataset_configs,
            target_fs=self.target_fs,
            use_legacy_mne_positions=self.use_legacy_mne_positions,
            positions_json_path=self.positions_json_path,
        )
