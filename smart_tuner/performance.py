# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:27:29 2025

@author: Admin
"""

# smart_tuner/performance.py
from __future__ import annotations
import numpy as np

from .models import SimulationResult


def compute_iae(result: SimulationResult) -> float:
    """Integral of absolute error."""
    return np.trapz(np.abs(result.e), result.t)


def compute_ise(result: SimulationResult) -> float:
    """Integral of squared error."""
    return np.trapz(result.e ** 2, result.t)


def compute_itae(result: SimulationResult) -> float:
    """Integral of time-weighted absolute error."""
    return np.trapz(result.t * np.abs(result.e), result.t)


def compute_overshoot(result: SimulationResult, setpoint: float) -> float:
    """Compute percentage overshoot."""
    max_y = np.max(result.y)
    if setpoint == 0:
        return 0.0
    return max(0.0, (max_y - setpoint) / setpoint * 100.0)
