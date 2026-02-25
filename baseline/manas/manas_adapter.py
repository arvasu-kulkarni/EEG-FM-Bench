"""
MANAS adapter for EEG-FM-Bench.
"""

import logging
from typing import List, Dict, Any

import mne
import numpy as np
import torch
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory
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

    def __init__(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
        target_fs: int = 200,
    ):
        self.electrode_set: ElectrodeSet = ElectrodeSet()
        self.target_fs = target_fs
        super().__init__(dataset, dataset_names, dataset_configs)

    def _setup_adapter(self):
        self.model_name = 'manas'
        self._build_montage_mappings()
        self._log_adapter_info()

    def get_supported_channels(self) -> List[str]:
        return self.electrode_set.Electrodes

    @staticmethod
    def _make_positions(channel_names: List[str]) -> torch.Tensor:
        mne_info = mne.create_info(channel_names, sfreq=100, ch_types="eeg")
        raw_obj = mne.io.RawArray(np.zeros((len(channel_names), 100)), mne_info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_obj.set_montage(montage, match_case=False, on_missing="raise")
        pos = raw_obj.get_montage().get_positions()["ch_pos"].values()
        pos_np = np.asarray(list(pos), dtype=np.float32)
        return 100 * torch.from_numpy(pos_np)

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
    ):
        super().__init__(batch_size, num_workers, seed)
        self.target_fs = target_fs

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
        )
