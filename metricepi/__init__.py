"""
MetricEPI: Distance-Aware Deep Learning for Enhancer-Promoter Interaction Prediction

A deep learning model that incorporates genomic distance directly into the
attention mechanism for predicting enhancer-promoter interactions.
"""

from .model import (
    DistanceAwareEPIModelV2,
    MultiScaleDistanceEncoder,
    TrueDistanceAwareAttention,
    create_distance_aware_model_v2,
)
from .trainer import (
    TrainerV2,
    FocalLoss,
    EarlyStopping,
    create_trainer_v2,
)
from .data_loader import (
    ShardedEPIDataset,
    ShardedEPICollate,
    ShardedSampler,
    get_sharded_dataloader,
    get_train_val_dataloaders,
)

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    # Model
    "DistanceAwareEPIModelV2",
    "MultiScaleDistanceEncoder",
    "TrueDistanceAwareAttention",
    "create_distance_aware_model_v2",
    # Trainer
    "TrainerV2",
    "FocalLoss",
    "EarlyStopping",
    "create_trainer_v2",
    # Data
    "ShardedEPIDataset",
    "ShardedEPICollate",
    "ShardedSampler",
    "get_sharded_dataloader",
    "get_train_val_dataloaders",
]
