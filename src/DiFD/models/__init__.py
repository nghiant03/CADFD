"""Deep learning models for fault diagnosis.

This module provides model architectures and a registry system for
managing different model implementations.

Subpackages:
    temporal/  - CNN1D, LSTM, GRU, Transformer, Autoformer, Informer, PatchTST
    spatial/   - ST-GCN
"""

from DiFD.models.base import BaseModel
from DiFD.models.registry import (
    create_model,
    get_model_class,
    is_registered,
    list_models,
    register_model,
)
from DiFD.models.spatial import STGCNClassifier
from DiFD.models.temporal import (
    AutoformerClassifier,
    CNN1DClassifier,
    GRUClassifier,
    InformerClassifier,
    LSTMClassifier,
    PatchTSTClassifier,
    TransformerClassifier,
)

__all__ = [
    "AutoformerClassifier",
    "BaseModel",
    "CNN1DClassifier",
    "GRUClassifier",
    "InformerClassifier",
    "LSTMClassifier",
    "PatchTSTClassifier",
    "STGCNClassifier",
    "TransformerClassifier",
    "create_model",
    "get_model_class",
    "is_registered",
    "list_models",
    "register_model",
]
