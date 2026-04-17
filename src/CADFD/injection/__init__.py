"""Fault injection module for generating synthetic fault datasets.

This module provides tools for injecting faults into sensor data
using a Markov chain model to generate realistic fault sequences.
"""

from CADFD.injection.base import BaseFaultInjector
from CADFD.injection.faults import (
    DriftFaultInjector,
    SpikeFaultInjector,
    StuckFaultInjector,
)
from CADFD.injection.injector import FaultInjector
from CADFD.injection.markov import MarkovStateGenerator
from CADFD.injection.registry import get_injector, register_fault

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
