# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:15:41 2025

@author: Admin
"""

# smart_tuner/__init__.py

from .models import (
    SystemModel,
    PIController,
    MPCConfig,
    SimulationConfig,
    SimulationResult,
)
from .identification import identify_fopdt_from_step
from .tuning_pi import tune_pi
from .simulation import simulate_closed_loop_pi

from .performance import (
    compute_iae,
    compute_ise,
    compute_itae,
    compute_overshoot,
)

__all__ = [
    "SystemModel",
    "PIController",
    "MPCConfig",
    "SimulationConfig",
    "SimulationResult",
    "identify_fopdt_from_step",
    "tune_pi",
    "simulate_closed_loop_pi",
    "compute_iae",
    "compute_ise",
    "compute_itae",
    "compute_overshoot",
]

from .models import (
    SystemModel,
    PIController,
    MPCConfig,
    MPCController,        # NOVO
    SimulationConfig,
    SimulationResult,
)
from .tuning_mpc import tune_mpc          # NOVO
from .simulation import simulate_closed_loop_mpc  # NOVO