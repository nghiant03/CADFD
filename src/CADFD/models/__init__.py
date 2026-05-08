"""Deep learning models for fault diagnosis.

This module provides model architectures and a registry system for
managing different model implementations.

Subpackages:
    temporal/  - CNN1D, LSTM, GRU, Transformer, Autoformer, Informer, PatchTST, ModernTCN
    spatial/   - CESTA, ST-GCN
"""

from CADFD.models.base import BaseModel
from CADFD.models.registry import (
    create_model,
    get_model_class,
    is_registered,
    list_models,
    register_model,
)
from CADFD.models.spatial import CESTAClassifier, STGCNClassifier
from CADFD.models.temporal import (
    AutoformerClassifier,
    CNN1DClassifier,
    GRUClassifier,
    InformerClassifier,
    LSTMClassifier,
    ModernTCNClassifier,
    PatchTSTClassifier,
    TransformerClassifier,
)

__all__ = [
    "AutoformerClassifier",
    "BaseModel",
    "CESTAClassifier",
    "CNN1DClassifier",
    "GRUClassifier",
    "InformerClassifier",
    "LSTMClassifier",
    "ModernTCNClassifier",
    "PatchTSTClassifier",
    "STGCNClassifier",
    "TransformerClassifier",
    "create_model",
    "get_model_class",
    "is_registered",
    "list_models",
    "register_model",
]
