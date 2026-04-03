from baseline.abstract.factory import ModelRegistry
from baseline.cbramod.cbramod_adapter import CBraModDataLoaderFactory
from baseline.cbramod.cbramod_config import CBraModConfig
from baseline.cbramod.cbramod_trainer import CBraModTrainer
# from baseline.conformer.conformer_config import ConformerConfig
# from baseline.conformer.conformer_trainer import ConformerTrainer
from baseline.csbrain.csbrain_adapter import CSBrainDataLoaderFactory
from baseline.csbrain.csbrain_config import CSBrainConfig
from baseline.csbrain.csbrain_trainer import CSBrainTrainer
# from baseline.eegnet.eegnet_config import EegNetConfig
# from baseline.eegnet.eegnet_trainer import EegNetTrainer
from baseline.eegpt.eegpt_adapter import EegptDataLoaderFactory
from baseline.eegpt.eegpt_config import EegptConfig
from baseline.eegpt.eegpt_trainer import EegptTrainer
from baseline.labram.labram_adapter import LabramDataLoaderFactory
from baseline.labram.labram_config import LabramConfig
from baseline.labram.labram_trainer import LabramTrainer
from baseline.bendr.bendr_config import BendrConfig
from baseline.bendr.bendr_trainer import BendrTrainer
from baseline.biot.biot_config import BiotConfig
from baseline.biot.biot_trainer import BiotTrainer
from baseline.mantis import MantisConfig, MantisDataLoaderFactory, MantisTrainer
from baseline.moment import MomentConfig, MomentDataLoaderFactory, MomentTrainer
from baseline.manas import ManasConfig, ManasDataLoaderFactory, ManasTrainer
from baseline.manas_long import ManasLongConfig, ManasLongDataLoaderFactory, ManasLongTrainer
from baseline.manas_long_v0_1 import (
    ManasLongV01Config,
    ManasLongV01DataLoaderFactory,
    ManasLongV01Trainer,
)
from baseline.manas_long_v0_2 import (
    ManasLongV02Config,
    ManasLongV02DataLoaderFactory,
    ManasLongV02Trainer,
)
from baseline.manas_long_v0_3 import (
    ManasLongV03Config,
    ManasLongV03DataLoaderFactory,
    ManasLongV03Trainer,
)
from baseline.reve.reve_adapter import ReveDataLoaderFactory
from baseline.reve.reve_config import ReveConfig
from baseline.reve.reve_trainer import ReveTrainer

ModelRegistry.register_model(
    model_type='eegpt',
    config_class=EegptConfig,
    adapter_class=EegptDataLoaderFactory,
    trainer_class=EegptTrainer
)

ModelRegistry.register_model(
    model_type='labram',
    config_class=LabramConfig,
    adapter_class=LabramDataLoaderFactory,
    trainer_class=LabramTrainer
)

ModelRegistry.register_model(
    model_type='bendr',
    config_class=BendrConfig,
    adapter_class=None,
    trainer_class=BendrTrainer
)

ModelRegistry.register_model(
    model_type='biot',
    config_class=BiotConfig,
    adapter_class=None,
    trainer_class=BiotTrainer
)

ModelRegistry.register_model(
    model_type='cbramod',
    config_class=CBraModConfig,
    adapter_class=CBraModDataLoaderFactory,
    trainer_class=CBraModTrainer
)

ModelRegistry.register_model(
    model_type='reve',
    config_class=ReveConfig,
    adapter_class=ReveDataLoaderFactory,
    trainer_class=ReveTrainer
)

ModelRegistry.register_model(
    model_type='csbrain',
    config_class=CSBrainConfig,
    adapter_class=CSBrainDataLoaderFactory,
    trainer_class=CSBrainTrainer
)

# ModelRegistry.register_model(
#     model_type='eegnet',
#     config_class=EegNetConfig,
#     adapter_class=None,
#     trainer_class=EegNetTrainer
# )

# ModelRegistry.register_model(
#     model_type='conformer',
#     config_class=ConformerConfig,
#     adapter_class=None,
#     trainer_class=ConformerTrainer
# )

ModelRegistry.register_model(
    model_type='mantis',
    config_class=MantisConfig,
    adapter_class=MantisDataLoaderFactory,
    trainer_class=MantisTrainer
)

ModelRegistry.register_model(
    model_type='moment',
    config_class=MomentConfig,
    adapter_class=MomentDataLoaderFactory,
    trainer_class=MomentTrainer
)

ModelRegistry.register_model(
    model_type='manas',
    config_class=ManasConfig,
    adapter_class=ManasDataLoaderFactory,
    trainer_class=ManasTrainer
)

ModelRegistry.register_model(
    model_type='manas-long',
    config_class=ManasLongConfig,
    adapter_class=ManasLongDataLoaderFactory,
    trainer_class=ManasLongTrainer
)

ModelRegistry.register_model(
    model_type='manas-long-v0.1',
    config_class=ManasLongV01Config,
    adapter_class=ManasLongV01DataLoaderFactory,
    trainer_class=ManasLongV01Trainer
)

ModelRegistry.register_model(
    model_type='manas-long-v0.2',
    config_class=ManasLongV02Config,
    adapter_class=ManasLongV02DataLoaderFactory,
    trainer_class=ManasLongV02Trainer
)

ModelRegistry.register_model(
    model_type='manas-long-v0.3',
    config_class=ManasLongV03Config,
    adapter_class=ManasLongV03DataLoaderFactory,
    trainer_class=ManasLongV03Trainer
)

__all__ = [
    "ModelRegistry",
    "CBraModConfig",
    "CBraModDataLoaderFactory",
    "CBraModTrainer",
    "CSBrainConfig",
    "CSBrainDataLoaderFactory",
    "CSBrainTrainer",
    "EegptConfig",
    "EegptDataLoaderFactory",
    "EegptTrainer",
    "LabramConfig",
    "LabramDataLoaderFactory",
    "LabramTrainer",
    "BendrConfig",
    "BendrTrainer",
    "BiotConfig",
    "BiotTrainer",
    "MantisConfig",
    "MantisDataLoaderFactory",
    "MantisTrainer",
    "MomentConfig",
    "MomentDataLoaderFactory",
    "MomentTrainer",
    "ManasConfig",
    "ManasDataLoaderFactory",
    "ManasTrainer",
    "ManasLongConfig",
    "ManasLongDataLoaderFactory",
    "ManasLongTrainer",
    "ManasLongV01Config",
    "ManasLongV01DataLoaderFactory",
    "ManasLongV01Trainer",
    "ManasLongV02Config",
    "ManasLongV02DataLoaderFactory",
    "ManasLongV02Trainer",
    "ManasLongV03Config",
    "ManasLongV03DataLoaderFactory",
    "ManasLongV03Trainer",
    "ReveConfig",
    "ReveDataLoaderFactory",
    "ReveTrainer",
]
