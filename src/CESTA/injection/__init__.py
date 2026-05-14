"""Fault injection module for generating synthetic fault datasets.

This module provides tools for injecting faults into sensor data
using a Markov chain model to generate realistic fault sequences.
"""

from CESTA.injection.base import BaseFaultInjector
from CESTA.injection.faults import (
    DriftFaultInjector,
    SpikeFaultInjector,
    StuckFaultInjector,
)
from CESTA.injection.injector import FaultInjector
from CESTA.injection.markov import MarkovStateGenerator
from CESTA.injection.registry import get_injector, register_fault

__all__ = [
    "BaseFaultInjector",
    "SpikeFaultInjector",
    "DriftFaultInjector",
    "StuckFaultInjector",
    "FaultInjector",
    "MarkovStateGenerator",
    "register_fault",
    "get_injector",
]
