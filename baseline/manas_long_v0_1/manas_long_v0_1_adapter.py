"""
MANAS-Long v0.1 adapter for EEG-FM-Bench.
"""

from typing import List

from datasets import Dataset as HFDataset

from baseline.manas_long.manas_long_adapter import (
    ManasLongDataLoaderFactory,
    ManasLongDatasetAdapter,
)


class ManasLongV01DatasetAdapter(ManasLongDatasetAdapter):
    """MANAS-Long v0.1 reuses the MANAS-Long preprocessing pipeline."""

    def _setup_adapter(self):
        self.model_name = "manas-long-v0.1"
        self._build_montage_mappings()
        self._log_adapter_info()


class ManasLongV01DataLoaderFactory(ManasLongDataLoaderFactory):
    """DataLoader factory for MANAS-Long v0.1."""

    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> ManasLongV01DatasetAdapter:
        return ManasLongV01DatasetAdapter(
            dataset,
            dataset_names,
            dataset_configs,
            target_fs=self.target_fs,
            use_legacy_mne_positions=self.use_legacy_mne_positions,
            positions_json_path=self.positions_json_path,
        )
