"""
MANAS-Long adapter for EEG-FM-Bench.
"""

from typing import List, Optional

from datasets import Dataset as HFDataset

from baseline.manas.manas_adapter import ManasDatasetAdapter, ManasDataLoaderFactory


class ManasLongDatasetAdapter(ManasDatasetAdapter):
    """MANAS-Long reuses the MANAS preprocessing/position pipeline."""

    def _setup_adapter(self):
        self.model_name = "manas-long"
        self._build_montage_mappings()
        self._log_adapter_info()


class ManasLongDataLoaderFactory(ManasDataLoaderFactory):
    """MANAS-Long DataLoader factory that attaches position information."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        seed: int = 42,
        target_fs: int = 200,
        use_legacy_mne_positions: bool = True,
        positions_json_path: Optional[str] = None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            target_fs=target_fs,
            use_legacy_mne_positions=use_legacy_mne_positions,
            positions_json_path=positions_json_path,
        )

    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str],
    ) -> ManasLongDatasetAdapter:
        return ManasLongDatasetAdapter(
            dataset,
            dataset_names,
            dataset_configs,
            target_fs=self.target_fs,
            use_legacy_mne_positions=self.use_legacy_mne_positions,
            positions_json_path=self.positions_json_path,
        )
