"""Hyperparameter optimization with Optuna.

Provides per-model search spaces and an :class:`Optimizer` that drives an
Optuna study by training a fresh model for each trial.
"""

from CADFD.optimization.optimizer import Optimizer, TrialResult
from CADFD.optimization.search_spaces import (
    SearchSpaceFn,
    get_search_space,
    register_search_space,
    suggest_train_hyperparams,
)

__all__ = [
    "Optimizer",
    "SearchSpaceFn",
    "TrialResult",
    "get_search_space",
    "register_search_space",
    "suggest_train_hyperparams",
]
