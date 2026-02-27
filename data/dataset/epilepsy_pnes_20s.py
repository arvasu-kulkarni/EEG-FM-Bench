import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import datasets
import mne
import numpy as np
from mne.io import BaseRaw
from pandas import DataFrame

from common.type import DatasetTaskType
from data.processor.builder import EEGConfig, EEGDatasetBuilder


logger = logging.getLogger("preproc")


@dataclass
class EpilepsyPnes20sConfig(EEGConfig):
    name: str = "pretrain"
    version: Optional[Union[datasets.utils.Version, str]] = datasets.utils.Version("1.0.0")
    description: Optional[str] = (
        "Preprocessed 20-second EEG segments for epileptic vs PNES (mimicker) classification. "
        "Data are already segmented, filtered, and normalized per recording at 200 Hz."
    )
    citation: Optional[str] = None

    dataset_name: Optional[str] = "epilepsy_pnes_20s"
    task_type: DatasetTaskType = DatasetTaskType.CLINICAL
    file_ext: str = "npy"
    unit: str = "V"
    is_notched: bool = True
    montage: dict[str, list[str]] = field(default_factory=lambda: {
        "10_20": [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T3",
            "C3",
            "Cz",
            "C4",
            "T4",
            "T5",
            "P3",
            "Pz",
            "P4",
            "T6",
            "O1",
            "O2",
        ]
    })

    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    wnd_div_sec: int = 20
    suffix_path: str = "/share/data/finaldata/data_eeg_mim_prep_20s"
    scan_sub_dir: str = "data"

    category: list[str] = field(default_factory=lambda: ["epileptic", "mimicker"])
    target_subjects_per_class: int = 100
    train_subjects_per_class: int = 85
    valid_subjects_per_class: int = 5
    test_subjects_per_class: int = 10
    include_sources: list[str] = field(default_factory=lambda: ["data_ihbas"])
    source_sfreq: float = 200.0


class EpilepsyPnes20sBuilder(EEGDatasetBuilder):
    BUILDER_CONFIG_CLASS = EpilepsyPnes20sConfig
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name="pretrain"),
        BUILDER_CONFIG_CLASS(name="finetune", is_finetune=True),
    ]

    def __init__(self, config_name="pretrain", **kwargs):
        super().__init__(config_name, **kwargs)
        self._selected_subject_dirs: Optional[list[str]] = None
        self._selected_subject_split_map: Optional[dict[tuple[str, str], str]] = None

    @staticmethod
    def _normalize_label(folder_name: str) -> Optional[str]:
        name = folder_name.strip().lower()
        if name.startswith("epile"):
            return "epileptic"
        if name.startswith("mim"):
            return "mimicker"
        return None

    def _scan_subject_records(self) -> dict[str, dict[str, dict[str, Any]]]:
        base_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Data directory does not exist: {base_path}")

        # label -> subject_key -> subject_record
        records: dict[str, dict[str, dict[str, Any]]] = {
            "epileptic": {},
            "mimicker": {},
        }
        allowed_sources = {src.lower() for src in self.config.include_sources}

        for source in sorted(os.listdir(base_path)):
            if allowed_sources and source.lower() not in allowed_sources:
                continue
            source_path = os.path.join(base_path, source)
            if not os.path.isdir(source_path):
                continue

            for split in sorted(os.listdir(source_path)):
                split_path = os.path.join(source_path, split)
                if not os.path.isdir(split_path):
                    continue

                for label_dir in sorted(os.listdir(split_path)):
                    label = self._normalize_label(label_dir)
                    if label is None:
                        continue

                    label_path = os.path.join(split_path, label_dir)
                    if not os.path.isdir(label_path):
                        continue

                    for subject in sorted(os.listdir(label_path)):
                        subject_path = os.path.join(label_path, subject)
                        if not os.path.isdir(subject_path):
                            continue

                        key = f"{source}:{subject}"
                        if key not in records[label]:
                            records[label][key] = {
                                "source": source,
                                "subject": subject,
                                "dirs": [],
                                "source_splits": set(),
                            }
                        records[label][key]["dirs"].append(subject_path)
                        records[label][key]["source_splits"].add(split.lower())

        return records

    def _select_subject_dirs(self) -> list[str]:
        if self._selected_subject_dirs is not None:
            return self._selected_subject_dirs

        subject_records = self._scan_subject_records()
        rng = np.random.default_rng(self.config.seed)

        selected_dirs: list[str] = []
        selected_subject_split: dict[tuple[str, str], str] = {}
        for label in self.config.category:
            candidates = sorted(
                subject_records[label].values(),
                key=lambda x: (x["source"], x["subject"]),
            )

            if self.config.is_finetune:
                train_target = self.config.train_subjects_per_class
                valid_target = self.config.valid_subjects_per_class
                test_target = self.config.test_subjects_per_class
                target_per_class = train_target + valid_target + test_target

                if target_per_class != self.config.target_subjects_per_class:
                    raise ValueError(
                        "Inconsistent subject targets: "
                        f"train+valid+test={target_per_class}, "
                        f"target_subjects_per_class={self.config.target_subjects_per_class}."
                    )

                if len(candidates) < target_per_class:
                    raise ValueError(
                        f"Not enough subjects for class '{label}': "
                        f"found {len(candidates)}, required {target_per_class}."
                    )

                train_pool = [item for item in candidates if item["source_splits"] == {"train"}]
                test_pool = [item for item in candidates if item["source_splits"] == {"test"}]
                mixed_pool = [
                    item for item in candidates
                    if item["source_splits"] not in ({"train"}, {"test"})
                ]

                if mixed_pool:
                    raise ValueError(
                        f"Found subjects with mixed/unknown source splits for class '{label}': "
                        f"{len(mixed_pool)}."
                    )
                if len(test_pool) < test_target:
                    raise ValueError(
                        f"Not enough source Test subjects for class '{label}': "
                        f"found {len(test_pool)}, required {test_target}."
                    )

                train_valid_target = train_target + valid_target
                if len(train_pool) < train_valid_target:
                    raise ValueError(
                        f"Not enough source Train subjects for class '{label}': "
                        f"found {len(train_pool)}, required {train_valid_target}."
                    )

                chosen_test_idx = rng.choice(len(test_pool), size=test_target, replace=False)
                chosen_train_valid_idx = rng.choice(
                    len(train_pool), size=train_valid_target, replace=False
                )
                valid_rel_idx = set()
                if valid_target > 0:
                    valid_rel_idx = set(
                        rng.choice(train_valid_target, size=valid_target, replace=False).tolist()
                    )

                for rel_idx, pool_idx in enumerate(chosen_train_valid_idx.tolist()):
                    record = train_pool[pool_idx]
                    subject_key = f"{record['source']}:{record['subject']}"
                    split_name = "valid" if rel_idx in valid_rel_idx else "train"
                    selected_subject_split[(label, subject_key)] = split_name
                    selected_dirs.extend(record["dirs"])

                for pool_idx in chosen_test_idx.tolist():
                    record = test_pool[pool_idx]
                    subject_key = f"{record['source']}:{record['subject']}"
                    selected_subject_split[(label, subject_key)] = "test"
                    selected_dirs.extend(record["dirs"])

                logger.info(
                    f"Selected class '{label}': "
                    f"{train_target} train, {valid_target} valid, {test_target} test subjects."
                )
            else:
                target_per_class = self.config.target_subjects_per_class
                if len(candidates) < target_per_class:
                    raise ValueError(
                        f"Not enough subjects for class '{label}': "
                        f"found {len(candidates)}, required {target_per_class}."
                    )

                chosen_idx = rng.choice(len(candidates), size=target_per_class, replace=False)
                for idx in chosen_idx.tolist():
                    record = candidates[idx]
                    subject_key = f"{record['source']}:{record['subject']}"
                    selected_subject_split[(label, subject_key)] = "train"
                    selected_dirs.extend(record["dirs"])

                logger.info(
                    f"Selected {target_per_class}/{len(candidates)} subjects for class '{label}'."
                )

        self._selected_subject_dirs = sorted(selected_dirs)
        self._selected_subject_split_map = selected_subject_split
        logger.info(
            f"Selected total subject directories: {len(self._selected_subject_dirs)}."
        )
        return self._selected_subject_dirs

    def _walk_raw_data_files(self):
        logger.info("Collecting selected preprocessed segment files...")
        raw_data_files: list[str] = []
        for subject_dir in self._select_subject_dirs():
            for file_name in sorted(os.listdir(subject_dir)):
                if file_name.endswith(self.config.file_ext):
                    raw_data_files.append(os.path.join(subject_dir, file_name))

        logger.info(f"Selected segment files: {len(raw_data_files)}")
        return raw_data_files

    def _resolve_file_name(self, file_path: str) -> dict[str, Any]:
        base_path = os.path.join(self.config.raw_path, self.config.scan_sub_dir)
        rel_path = os.path.relpath(file_path, base_path)
        parts = rel_path.split(os.sep)
        if len(parts) < 5:
            raise ValueError(f"Unexpected file path format: {file_path}")

        source, split, label_dir, subject = parts[:4]
        segment_name = os.path.splitext(parts[-1])[0]
        label = self._normalize_label(label_dir)
        if label is None:
            raise ValueError(f"Unknown class folder '{label_dir}' in: {file_path}")

        segment = -1
        if segment_name.startswith("seg_"):
            try:
                segment = int(segment_name.split("_")[-1])
            except ValueError:
                segment = -1

        return {
            "subject": f"{source}:{subject}",
            "session": split.lower(),
            "source": source,
            "subject_id": subject,
            "segment": segment,
            "label_name": label,
        }

    def _resolve_exp_meta_info(self, file_path: str) -> dict[str, Any]:
        info = self._resolve_file_name(file_path)
        info.update(
            {
                "montage": "10_20",
                "time": float(self.config.wnd_div_sec),
            }
        )
        return info

    def _resolve_exp_events(self, file_path: str, info: dict[str, Any]):
        if not self.config.is_finetune:
            return [("default", 0, -1)]
        return [(info["label_name"], 0, -1)]

    def _exclude_wrong_data(self, df: DataFrame, n_proc: Optional[int] = None):
        # Data are already preprocessed and fixed-length segments; keep only length check.
        return self._check_data_length(df)

    def _divide_split(self, df: DataFrame) -> DataFrame:
        if not self.config.is_finetune:
            return self._divide_all_split_by_sub(df)

        if self._selected_subject_split_map is None:
            self._select_subject_dirs()

        if self._selected_subject_split_map is None:
            raise RuntimeError("Subject split map was not initialized.")

        key_pairs = list(zip(df["label_name"], df["subject"]))
        df["split"] = [self._selected_subject_split_map.get(key) for key in key_pairs]
        missing_subjects = sorted(df.loc[df["split"].isna(), "subject"].unique().tolist())
        if missing_subjects:
            missing_preview = ", ".join(missing_subjects[:5])
            raise ValueError(
                f"Missing split assignment for {len(missing_subjects)} subjects. "
                f"Examples: {missing_preview}"
            )
        return df

    def standardize_chs_names(self, montage: str):
        if montage in self._std_chs_cache:
            return self._std_chs_cache[montage]

        chs = self.config.montage[montage]
        chs_std = [ch.upper() for ch in chs]
        chs_std = [self.montage_10_20_replace_dict.get(ch, ch) for ch in chs_std]
        self._std_chs_cache[montage] = chs_std
        return chs_std

    def _read_raw_data(self, file_path: str, preload: bool = False, verbose: bool = False) -> BaseRaw:
        data = np.load(file_path, allow_pickle=False)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D [channels, time] array, got shape {data.shape} at {file_path}")

        chs = self.config.montage["10_20"]
        if data.shape[0] != len(chs):
            raise ValueError(
                f"Unexpected channel count for {file_path}: "
                f"got {data.shape[0]}, expected {len(chs)}"
            )

        info = mne.create_info(
            ch_names=chs,
            sfreq=self.config.source_sfreq,
            ch_types=["eeg"] * len(chs),
        )
        raw = mne.io.RawArray(data.astype(np.float32), info, verbose=verbose)
        return raw

    def _resample_and_filter(self, data: BaseRaw):
        # Keep preprocessed signals untouched; only resample if caller explicitly requests another fs.
        orig_fs = data.info["sfreq"]
        if orig_fs != self.config.fs:
            data = data.resample(sfreq=self.config.fs, verbose=False)
        return data


if __name__ == "__main__":
    builder = EpilepsyPnes20sBuilder("finetune")
    builder.preproc(n_proc=1)
    builder.download_and_prepare(num_proc=1)
    dataset = builder.as_dataset()
    print(dataset)
