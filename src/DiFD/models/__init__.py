"""Deep learning models for fault diagnosis.

This module provides model architectures and a registry system for
managing different model implementations.

Subpackages:
    recurrent/    - LSTM, GRU
    transformer/  - Transformer, Autoformer, Informer, PatchTST
    graph/        - ST-GCN
"""

from DiFD.models.base import BaseModel
from DiFD.models.graph import STGCNClassifier
from DiFD.models.recurrent import GRUClassifier, LSTMClassifier
from DiFD.models.registry import (
    create_model,
    get_model_class,
    is_registered,
    list_models,
    register_model,
)
from DiFD.models.transformer import (
    AutoformerClassifier,
    InformerClassifier,
    PatchTSTClassifier,
    TransformerClassifier,
)

__all__ = [
    "AutoformerClassifier",
    "BaseModel",
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
