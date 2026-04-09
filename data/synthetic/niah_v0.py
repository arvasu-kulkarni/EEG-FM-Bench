from __future__ import annotations

import hashlib
from typing import Any, Literal

import datasets
import numpy as np
from datasets import Dataset
from pydantic import BaseModel, Field, model_validator

from common.type import DatasetTaskType


def _range_to_tuple(values: list[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must have exactly two values, got {values!r}")
    lo, hi = float(values[0]), float(values[1])
    if hi < lo:
        raise ValueError(f"{name} must be in ascending order, got {values!r}")
    return lo, hi


def _normalize_weights(gaussian_weight: float, lobe_weight: float) -> tuple[float, float]:
    total = float(gaussian_weight) + float(lobe_weight)
    if total <= 0.0:
        raise ValueError("gaussian_weight + lobe_weight must be > 0.")
    return float(gaussian_weight) / total, float(lobe_weight) / total


class NIAHGaussianArtifactConfig(BaseModel):
    duration_sec_range: list[float] = Field(default_factory=lambda: [0.12, 0.35])
    sigma_ratio_range: list[float] = Field(default_factory=lambda: [0.08, 0.12])
    noise_fraction_range: list[float] = Field(default_factory=lambda: [0.02, 0.05])

    @model_validator(mode="after")
    def validate_config(self) -> "NIAHGaussianArtifactConfig":
        _range_to_tuple(self.duration_sec_range, "gaussian.duration_sec_range")
        sigma_lo, sigma_hi = _range_to_tuple(self.sigma_ratio_range, "gaussian.sigma_ratio_range")
        if sigma_lo <= 0.0:
            raise ValueError("gaussian.sigma_ratio_range must be > 0.")
        _range_to_tuple(self.noise_fraction_range, "gaussian.noise_fraction_range")
        return self


class NIAHLobeArtifactConfig(BaseModel):
    duration_sec_range: list[float] = Field(default_factory=lambda: [0.35, 0.90])
    second_half_scale_range: list[float] = Field(default_factory=lambda: [0.65, 0.85])
    noise_fraction_range: list[float] = Field(default_factory=lambda: [0.02, 0.06])

    @model_validator(mode="after")
    def validate_config(self) -> "NIAHLobeArtifactConfig":
        _range_to_tuple(self.duration_sec_range, "lobe.duration_sec_range")
        second_lo, second_hi = _range_to_tuple(
            self.second_half_scale_range,
            "lobe.second_half_scale_range",
        )
        if second_lo < 0.0:
            raise ValueError("lobe.second_half_scale_range must be >= 0.")
        _range_to_tuple(self.noise_fraction_range, "lobe.noise_fraction_range")
        return self


class NIAHV0TaskConfig(BaseModel):
    kind: Literal["niah_v0"] = "niah_v0"
    source_dataset: str
    source_config: str = "finetune"
    seed: int = 42
    normalize_targets: bool = True
    target_names: list[str] = Field(default_factory=lambda: ["center", "duration"])

    amplitude_scale_range: list[float] = Field(default_factory=lambda: [4.0, 8.0])
    gaussian_weight: float = 0.7
    lobe_weight: float = 0.3
    channel_gain_jitter_std: float = 0.05
    center_margin_sec: float = 0.0
    allow_negative_polarity: bool = True

    gaussian: NIAHGaussianArtifactConfig = Field(default_factory=NIAHGaussianArtifactConfig)
    lobe: NIAHLobeArtifactConfig = Field(default_factory=NIAHLobeArtifactConfig)

    @model_validator(mode="after")
    def validate_config(self) -> "NIAHV0TaskConfig":
        if self.kind != "niah_v0":
            raise ValueError(f"Unsupported synthetic task kind: {self.kind}")
        _range_to_tuple(self.amplitude_scale_range, "amplitude_scale_range")
        if self.channel_gain_jitter_std < 0.0:
            raise ValueError("channel_gain_jitter_std must be >= 0.")
        if self.center_margin_sec < 0.0:
            raise ValueError("center_margin_sec must be >= 0.")
        if len(self.target_names) != 2:
            raise ValueError("target_names must contain exactly two items for center and duration.")
        _normalize_weights(self.gaussian_weight, self.lobe_weight)
        return self


def get_niah_v0_eval_info(config: dict[str, Any] | NIAHV0TaskConfig) -> dict[str, Any]:
    cfg = config if isinstance(config, NIAHV0TaskConfig) else NIAHV0TaskConfig.model_validate(config)
    return {
        "aggregate_by_subject": False,
        "subject_score_aggregation": "mean_logits",
        "prediction_type": "regression",
        "output_dim": 2,
        "target_names": list(cfg.target_names),
        "selection_metric": "rmse",
        "selection_metric_higher_is_better": False,
    }


def _split_to_offset(split: datasets.NamedSplit) -> int:
    if split == datasets.Split.TRAIN:
        return 0
    if split == datasets.Split.VALIDATION:
        return 10_000_003
    if split == datasets.Split.TEST:
        return 20_000_033
    return 30_000_067


def _seed_for_index(task_name: str, seed: int, split: datasets.NamedSplit, idx: int) -> int:
    digest = hashlib.sha256(f"{task_name}|{seed}|{split}|{idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _sample_uniform(rng: np.random.Generator, bounds: list[float], name: str) -> float:
    lo, hi = _range_to_tuple(bounds, name)
    if hi == lo:
        return lo
    return float(rng.uniform(lo, hi))


def _gaussian_template(duration_samples: int, sigma_ratio: float) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, duration_samples, dtype=np.float32)
    sigma = max(1e-4, float(sigma_ratio))
    wave = np.exp(-0.5 * (t / sigma) ** 2)
    wave /= max(float(np.max(np.abs(wave))), 1e-6)
    return wave.astype(np.float32, copy=False)


def _lobe_template(duration_samples: int, second_half_scale: float) -> np.ndarray:
    mid = max(1, duration_samples // 2)
    first = np.sin(np.linspace(0.0, np.pi, mid, endpoint=False, dtype=np.float32))
    second = -float(second_half_scale) * np.sin(
        np.linspace(0.0, np.pi, duration_samples - mid, endpoint=True, dtype=np.float32)
    )
    wave = np.concatenate([first, second], axis=0)
    if wave.shape[0] != duration_samples:
        wave = wave[:duration_samples]
    wave /= max(float(np.max(np.abs(wave))), 1e-6)
    return wave.astype(np.float32, copy=False)


def _sample_artifact(
    data: np.ndarray,
    cfg: NIAHV0TaskConfig,
    fs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[float], str]:
    if data.ndim != 2:
        raise ValueError(f"NIAH expects EEG data with shape (channels, time), got {data.shape!r}")

    n_channels, n_timepoints = int(data.shape[0]), int(data.shape[1])
    if n_timepoints < 8:
        raise ValueError(f"NIAH source window is too short: {n_timepoints} samples")

    gaussian_prob, _ = _normalize_weights(cfg.gaussian_weight, cfg.lobe_weight)
    artifact_kind = "gaussian" if float(rng.random()) < gaussian_prob else "lobe"

    amplitude_scale = _sample_uniform(rng, cfg.amplitude_scale_range, "amplitude_scale_range")
    signal_scale = float(np.median(np.std(data, axis=1)))
    if not np.isfinite(signal_scale) or signal_scale <= 1e-6:
        signal_scale = max(float(np.std(data)), 1e-3)
    amplitude = amplitude_scale * signal_scale
    if cfg.allow_negative_polarity and bool(rng.integers(0, 2)):
        amplitude *= -1.0

    if artifact_kind == "gaussian":
        duration_sec = _sample_uniform(rng, cfg.gaussian.duration_sec_range, "gaussian.duration_sec_range")
        sigma_ratio = _sample_uniform(rng, cfg.gaussian.sigma_ratio_range, "gaussian.sigma_ratio_range")
        noise_fraction = _sample_uniform(
            rng,
            cfg.gaussian.noise_fraction_range,
            "gaussian.noise_fraction_range",
        )
    else:
        duration_sec = _sample_uniform(rng, cfg.lobe.duration_sec_range, "lobe.duration_sec_range")
        second_half_scale = _sample_uniform(
            rng,
            cfg.lobe.second_half_scale_range,
            "lobe.second_half_scale_range",
        )
        noise_fraction = _sample_uniform(
            rng,
            cfg.lobe.noise_fraction_range,
            "lobe.noise_fraction_range",
        )

    duration_samples = max(3, int(round(duration_sec * float(fs))))
    duration_samples = min(duration_samples, n_timepoints)
    if duration_samples % 2 == 0 and duration_samples < n_timepoints:
        duration_samples += 1

    half = duration_samples // 2
    center_margin_samples = max(half, int(round(cfg.center_margin_sec * float(fs))))
    center_low = center_margin_samples
    center_high = n_timepoints - center_margin_samples
    if center_high <= center_low:
        center_low = half
        center_high = n_timepoints - half
    if center_high <= center_low:
        center_low = duration_samples // 2
        center_high = center_low + 1

    center_idx = int(rng.integers(center_low, center_high))
    start_idx = max(0, center_idx - duration_samples // 2)
    end_idx = min(n_timepoints, start_idx + duration_samples)
    start_idx = max(0, end_idx - duration_samples)
    duration_samples = int(end_idx - start_idx)
    center_idx = start_idx + duration_samples // 2

    if artifact_kind == "gaussian":
        base_wave = _gaussian_template(duration_samples, sigma_ratio=sigma_ratio)
    else:
        base_wave = _lobe_template(duration_samples, second_half_scale=second_half_scale)

    noise = rng.normal(
        loc=0.0,
        scale=max(abs(amplitude), 1e-6) * noise_fraction,
        size=duration_samples,
    ).astype(np.float32, copy=False)
    wave = amplitude * base_wave + noise

    if cfg.channel_gain_jitter_std > 0.0:
        gains = rng.normal(
            loc=1.0,
            scale=cfg.channel_gain_jitter_std,
            size=(n_channels, 1),
        ).astype(np.float32, copy=False)
    else:
        gains = np.ones((n_channels, 1), dtype=np.float32)

    artifact = gains * wave.reshape(1, -1)
    augmented = np.array(data, dtype=np.float32, copy=True)
    augmented[:, start_idx:end_idx] += artifact[:, : end_idx - start_idx]

    center_value = float(center_idx) / float(n_timepoints) if cfg.normalize_targets else float(center_idx) / float(fs)
    duration_value = (
        float(duration_samples) / float(n_timepoints)
        if cfg.normalize_targets
        else float(duration_samples) / float(fs)
    )
    label = [center_value, duration_value]
    return augmented, label, artifact_kind


def _replace_montage_prefix(montage: str, dataset_name: str) -> str:
    if "/" not in montage:
        return dataset_name
    _, suffix = montage.split("/", 1)
    return f"{dataset_name}/{suffix}"


def build_niah_v0_dataset(
    base_dataset: Dataset,
    dataset_name: str,
    config: dict[str, Any] | NIAHV0TaskConfig,
    split: datasets.NamedSplit,
    fs: int,
) -> Dataset:
    cfg = config if isinstance(config, NIAHV0TaskConfig) else NIAHV0TaskConfig.model_validate(config)
    split_offset = _split_to_offset(split)

    def _map_example(example: dict[str, Any], idx: int) -> dict[str, Any]:
        seed = _seed_for_index(dataset_name, cfg.seed + split_offset, split, idx)
        rng = np.random.default_rng(seed)
        data = np.asarray(example["data"], dtype=np.float32)
        augmented, label, artifact_kind = _sample_artifact(data, cfg, fs=fs, rng=rng)

        return {
            **example,
            "data": augmented.tolist(),
            "label": label,
            "task": int(DatasetTaskType.ARTIFACT.value),
            "montage": _replace_montage_prefix(str(example["montage"]), dataset_name),
            "niah_artifact_kind": artifact_kind,
        }

    mapped = base_dataset.map(
        _map_example,
        with_indices=True,
        load_from_cache_file=False,
        desc=f"Generating {dataset_name} synthetic NIAH samples",
    )
    mapped = mapped.cast_column("label", datasets.Sequence(datasets.Value("float32")))
    mapped = mapped.cast_column("task", datasets.Value("int32"))
    return mapped
