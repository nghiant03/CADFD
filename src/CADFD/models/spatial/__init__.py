"""Spatial models for fault diagnosis."""

from CADFD.models.spatial.cesta import CESTAClassifier
from CADFD.models.spatial.stgcn import STGCNClassifier

__all__ = [
    "CESTAClassifier",
    "STGCNClassifier",
]
