import logging
import re
from typing import Any, Type, Optional

import torch
import datasets
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets, Value

from data.dataset.adftd import AdftdBuilder
from data.dataset.bcic.bcic_1a import BCIC1ABuilder
from data.dataset.bcic.bcic_2020_3 import BCIC2020ImagineBuilder
from data.dataset.bcic.bcic_2a import BCIC2ABuilder
from data.dataset.brain_lat import BrainLatBuilder
from data.dataset.chisco import ChiscoBuilder
from data.dataset.emobrain import EmobrainBuilder
from data.dataset.epilepsy_pnes_20s import EpilepsyPnes20sBuilder
from data.dataset.grasp_and_lift import GraspAndLiftBuilder
from data.dataset.hbn import HBNBuilder
from data.dataset.hmc import HMCBuilder
from data.dataset.inner_speech import InnerSpeechBuilder
from data.dataset.inria_bci import InriaBciBuilder
from data.dataset.mimul_11 import Mimul11Builder
from data.dataset.motor_mv_img import MotorMoveImagineBuilder
from data.dataset.openmiir import OpenMiirBuilder
from data.dataset.seeds.seed import SeedBuilder
from data.dataset.seeds.seed_fra import SeedFraBuilder
from data.dataset.seeds.seed_ger import SeedGerBuilder
from data.dataset.seeds.seed_iv import SeedIVBuilder
from data.dataset.seeds.seed_v import SeedVBuilder
from data.dataset.seeds.seed_vii import SeedVIIBuilder
from data.dataset.siena_scalp import SienaScalpBuilder
from data.dataset.spis_resting_state import SpisRestingStateBuilder
from data.dataset.target_versus_non import TargetVersusNonBuilder
from data.dataset.things_eeg import ThingsEEGBuilder
from data.dataset.things_eeg_2 import ThingsEEG2Builder
from data.dataset.trujillo_2017 import Trujillo2017Builder
from data.dataset.trujillo_2019 import Trujillo2019Builder
from data.dataset.tue.tuab import TuabBuilder
from data.dataset.tue.tuar import TuarBuilder
from data.dataset.tue.tueg import TuegBuilder
from data.dataset.tue.tuep import TuepBuilder
from data.dataset.tue.tuev import TuevBuilder
from data.dataset.tue.tusl import TuslBuilder
from data.dataset.tue.tusz import TuszBuilder
from data.dataset.workload import WorkloadBuilder
from data.processor.builder import EEGDatasetBuilder, EEGConfig
from data.synthetic.niah_v0 import NIAHV0TaskConfig, build_niah_v0_dataset, get_niah_v0_eval_info
from data.synthetic.runtime import get_runtime_synthetic_dataset, is_runtime_synthetic_dataset


log = logging.getLogger()

WINDOW_ALIAS_PATTERN = re.compile(r"^(?P<base>hmc|siena|adftd)_(?P<window_sec>\d+)(?:s)?$")


DATASET_SELECTOR: dict[str, Type[EEGDatasetBuilder]] = {
    'tuab': TuabBuilder,
    'tuar': TuarBuilder,
    'tueg': TuegBuilder,
    'tuep': TuepBuilder,
    'tuev': TuevBuilder,
    'tusl': TuslBuilder,
    'tusz': TuszBuilder,
    'seed': SeedBuilder,
    'seed_fra': SeedFraBuilder,
    'seed_ger': SeedGerBuilder,
    'seed_iv': SeedIVBuilder,
    'seed_v': SeedVBuilder,
    'seed_vii': SeedVIIBuilder,
    'bcic_1a': BCIC1ABuilder,
    'bcic_2a': BCIC2ABuilder,
    'bcic_2020_3': BCIC2020ImagineBuilder,
    'emobrain': EmobrainBuilder,
    'grasp_and_lift': GraspAndLiftBuilder,
    'hmc': HMCBuilder,
    'inria_bci': InriaBciBuilder,
    'motor_mv_img': MotorMoveImagineBuilder,
    'siena_scalp': SienaScalpBuilder,
    'spis_resting_state': SpisRestingStateBuilder,
    'target_versus_non': TargetVersusNonBuilder,
    'trujillo_2017': Trujillo2017Builder,
    'trujillo_2019': Trujillo2019Builder,
    'workload': WorkloadBuilder,
    'hbn': HBNBuilder,
    'adftd': AdftdBuilder,
    'brain_lat': BrainLatBuilder,
    'things_eeg': ThingsEEGBuilder,
    'things_eeg_2': ThingsEEG2Builder,
    'mimul_11': Mimul11Builder,
    'inner_speech': InnerSpeechBuilder,
    'chisco': ChiscoBuilder,
    'open_miir': OpenMiirBuilder,
    'epilepsy_pnes_20s': EpilepsyPnes20sBuilder,
}

WINDOW_ALIAS_BASE_TO_DATASET = {
    'hmc': 'hmc',
    'siena': 'siena_scalp',
    'adftd': 'adftd',
}


def resolve_dataset_request(dataset_name: str) -> tuple[str, Type[EEGDatasetBuilder], dict[str, int | str]]:
    if dataset_name in DATASET_SELECTOR:
        return dataset_name, DATASET_SELECTOR[dataset_name], {}

    match = WINDOW_ALIAS_PATTERN.fullmatch(dataset_name)
    if match is None:
        raise KeyError(dataset_name)

    base_name = match.group('base')
    window_sec = int(match.group('window_sec'))
    dataset_key = WINDOW_ALIAS_BASE_TO_DATASET[base_name]
    builder_cls = DATASET_SELECTOR[dataset_key]
    return dataset_name, builder_cls, {
        'dataset_name': dataset_name,
        'wnd_div_sec': window_sec,
    }


def _replace_montage_prefix(montage_key: str, new_prefix: str) -> str:
    if '/' not in montage_key:
        return new_prefix
    _, suffix = montage_key.split('/', 1)
    return f'{new_prefix}/{suffix}'


def _get_builder(dataset_name: str, config_name: str, fs: Optional[int] = None) -> EEGDatasetBuilder:
    _, builder_cls, config_overrides = resolve_dataset_request(dataset_name)
    if fs is None:
        return builder_cls(config_name=config_name, **config_overrides)
    return builder_cls(config_name=config_name, fs=fs, **config_overrides)


def _get_runtime_synthetic_cfg(dataset_name: str) -> Optional[NIAHV0TaskConfig]:
    cfg = get_runtime_synthetic_dataset(dataset_name)
    if cfg is None:
        return None
    kind = cfg.get('kind')
    if kind != 'niah_v0':
        raise ValueError(f"Unsupported runtime synthetic dataset kind for {dataset_name}: {kind!r}")
    return NIAHV0TaskConfig.model_validate(cfg)


def _load_single_eeg_dataset(
        dataset_name: str,
        builder_config: str,
        split: datasets.NamedSplit,
        fs: int,
) -> Dataset:
    builder = _get_builder(dataset_name, builder_config, fs=fs)
    log.info(f'Loading {dataset_name}-{builder_config} at fs={fs}Hz from {builder.cache_dir}')
    return builder.as_dataset(split=split)

def get_dataset_patch_len(dataset_name: str, config_name: str) -> int:
    synthetic_cfg = _get_runtime_synthetic_cfg(dataset_name)
    if synthetic_cfg is not None:
        return get_dataset_patch_len(synthetic_cfg.source_dataset, synthetic_cfg.source_config)
    _, builder_cls, config_overrides = resolve_dataset_request(dataset_name)
    builder = builder_cls(config_name=config_name, **config_overrides)
    return builder.config.wnd_div_sec


def get_dataset_shape_info(dataset_name: str, config_name: str, fs: int) -> dict[str, tuple[int, int]]:
    """
    Get shape information for each montage in a dataset.

    Args:
        dataset_name: Name of the dataset
        config_name: Configuration name
        fs: Sampling frequency

    Returns:
        Dict mapping montage_key -> (n_timepoints, n_channels)
    """
    synthetic_cfg = _get_runtime_synthetic_cfg(dataset_name)
    if synthetic_cfg is not None:
        source_shapes = get_dataset_shape_info(
            synthetic_cfg.source_dataset,
            synthetic_cfg.source_config,
            fs,
        )
        return {
            _replace_montage_prefix(montage_key, dataset_name): shape
            for montage_key, shape in source_shapes.items()
        }

    resolved_name, builder_cls, config_overrides = resolve_dataset_request(dataset_name)
    builder: EEGDatasetBuilder = builder_cls(config_name=config_name, **config_overrides)

    config: EEGConfig = builder.config
    n_timepoints = int(config.wnd_div_sec * fs)

    shape_info: dict[str, tuple[int, int]] = {}
    for montage_name in config.montage.keys():
        montage_key = f'{resolved_name}/{montage_name}'
        chs = builder.standardize_chs_names(montage_name)
        n_channels = len(chs)
        shape_info[montage_key] = (n_timepoints, n_channels)

    return shape_info


def get_dataset_n_class(dataset_name: str, config_name: str) -> int:
    synthetic_cfg = _get_runtime_synthetic_cfg(dataset_name)
    if synthetic_cfg is not None:
        return int(get_niah_v0_eval_info(synthetic_cfg)['output_dim'])
    _, builder_cls, config_overrides = resolve_dataset_request(dataset_name)
    builder = builder_cls(config_name=config_name, **config_overrides)
    if getattr(builder.config, 'output_dim', None) is not None:
        return int(builder.config.output_dim)
    return len(builder.config.category)

def get_dataset_category(dataset_name: str, config_name: str) -> list[str]:
    synthetic_cfg = _get_runtime_synthetic_cfg(dataset_name)
    if synthetic_cfg is not None:
        return list(synthetic_cfg.target_names)
    _, builder_cls, config_overrides = resolve_dataset_request(dataset_name)
    builder = builder_cls(config_name=config_name, **config_overrides)
    return builder.config.category

def get_dataset_eval_info(dataset_name: str, config_name: str) -> dict[str, Any]:
    synthetic_cfg = _get_runtime_synthetic_cfg(dataset_name)
    if synthetic_cfg is not None:
        return get_niah_v0_eval_info(synthetic_cfg)
    _, builder_cls, config_overrides = resolve_dataset_request(dataset_name)
    builder = builder_cls(config_name=config_name, **config_overrides)
    return {
        'aggregate_by_subject': builder.config.eval_aggregate_by_subject,
        'subject_score_aggregation': builder.config.subject_score_aggregation,
        'prediction_type': getattr(builder.config, 'prediction_type', 'classification'),
        'output_dim': getattr(builder.config, 'output_dim', len(builder.config.category)),
        'target_names': list(getattr(builder.config, 'target_names', builder.config.category)),
        'selection_metric': getattr(builder.config, 'selection_metric', 'balanced_acc'),
        'selection_metric_higher_is_better': getattr(builder.config, 'selection_metric_higher_is_better', True),
    }

def get_dataset_montage(dataset_name: str, config_name: str) -> dict[str, list[str]]:
    # Note: This function needs builder instance to call standardize_chs_names()
    synthetic_cfg = _get_runtime_synthetic_cfg(dataset_name)
    if synthetic_cfg is not None:
        source_montages = get_dataset_montage(
            synthetic_cfg.source_dataset,
            synthetic_cfg.source_config,
        )
        return {
            _replace_montage_prefix(montage_key, dataset_name): channels
            for montage_key, channels in source_montages.items()
        }

    resolved_name, builder_cls, config_overrides = resolve_dataset_request(dataset_name)
    builder: EEGDatasetBuilder = builder_cls(config_name=config_name, **config_overrides)
    montage_names = builder.config.montage.keys()

    montages: dict[str, list[str]] = dict()
    for montage_name in montage_names:
        montages[f'{resolved_name}/{montage_name}'] = builder.standardize_chs_names(montage_name)

    return montages


def load_concat_eeg_datasets(
        dataset_names: list[str],
        builder_configs: list[str],
        split: datasets.NamedSplit = datasets.Split.TRAIN,
        weight_option: str = 'statistics',
        add_ds_name: bool = False,
        cast_label: bool = False,
        fs: Optional[int] = None,
) -> tuple[Dataset, list[Tensor]]:
    """
    Load and concatenate multiple EEG datasets.

    :param dataset_names: List of dataset names to load
    :param builder_configs: List of builder config names (e.g., 'pretrain', 'finetune')
    :param split: Dataset split to load (TRAIN, VALIDATION, TEST)
    :param weight_option: Weight calculation option for class imbalance
    :param add_ds_name: Whether to add dataset name column
    :param cast_label: Whether to cast label to int64
    :param fs: Target sampling rate (must match preprocessed data)
    :return: Tuple of concatenated dataset and weight list
    """
    dataset_list = []
    weight_list = []

    if fs is None:
        raise ValueError('fs for dataset loader must be specified')

    prediction_types_seen: set[str] = set()
    output_dims_seen: set[int] = set()

    for ds_name, ds_config in zip(dataset_names, builder_configs):
        try:
            synthetic_cfg = _get_runtime_synthetic_cfg(ds_name)
            if synthetic_cfg is not None:
                dataset = _load_single_eeg_dataset(
                    synthetic_cfg.source_dataset,
                    synthetic_cfg.source_config,
                    split=split,
                    fs=fs,
                )
                dataset = build_niah_v0_dataset(
                    base_dataset=dataset,
                    dataset_name=ds_name,
                    config=synthetic_cfg,
                    split=split,
                    fs=fs,
                )
            else:
                dataset = _load_single_eeg_dataset(ds_name, ds_config, split=split, fs=fs)

            if add_ds_name:
                dataset = dataset.add_column('ds_name', [ds_name for _ in range(len(dataset))])

            eval_info = get_dataset_eval_info(ds_name, ds_config)
            prediction_type = str(eval_info.get('prediction_type', 'classification'))
            output_dim = int(eval_info.get('output_dim', get_dataset_n_class(ds_name, ds_config)))
            prediction_types_seen.add(prediction_type)
            output_dims_seen.add(output_dim)

            if 'label' in dataset.column_names:
                if prediction_type == 'classification':
                    n_class = get_dataset_n_class(ds_name, ds_config)
                    label = torch.tensor(dataset['label'], dtype=torch.int32)
                    label_cnt = torch.bincount(label, minlength=n_class)
                    log.info(f'Sample distribution for {ds_name}-{ds_config} {split}: {label_cnt}')
                    weight = calc_distribution_weight(len(dataset), label_cnt, weight_option)
                    weight_list.append(weight)

                    if cast_label:
                        dataset = dataset.cast_column('label', Value('int64'))
                else:
                    log.info(
                        f'Sample distribution for {ds_name}-{ds_config} {split}: '
                        f'{prediction_type} (output_dim={output_dim}), skip bincount'
                    )
                    weight_list.append(torch.ones(1, dtype=torch.int64))

            dataset_list.append(dataset)
        except KeyError:
            log.error(f'Dataset {ds_name} not found')

    if len(prediction_types_seen) > 1:
        raise ValueError(
            f"Cannot concatenate datasets with mixed prediction types: {sorted(prediction_types_seen)}"
        )
    if prediction_types_seen == {'regression'} and len(output_dims_seen) > 1:
        raise ValueError(
            f"Cannot concatenate regression datasets with different output dims: {sorted(output_dims_seen)}"
        )

    combined_dataset: Dataset = concatenate_datasets(dataset_list)
    return combined_dataset.with_format('torch'), weight_list

def calc_distribution_weight(n: int, label_cnt: Tensor, option: str):
    if option == 'statistics':
        return label_cnt
    elif option == 'sqrt':
        return n / torch.sqrt(label_cnt.float() + 1)
    elif option == 'log':
        return n / torch.log(label_cnt.float() + 1)
    elif option == 'absolute':
        return n / label_cnt.float()
    else:
        raise ValueError(f'Unknown option {option}')



if __name__ == '__main__':
    # data = load_concat_eeg_datasets(['seed_v', 'tuab'])
    data, distribution = load_concat_eeg_datasets(['tuab'], ['finetune'], fs=256)
    loader = DataLoader(data, batch_size=32)

    for batch in loader:
        pass
