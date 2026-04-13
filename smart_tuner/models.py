# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:13:09 2025

@author: Admin
"""
# smart_tuner/models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal
import numpy as np


ModelType = Literal["transfer_function", "fopdt"]

from typing import Optional, Literal

AntiWindupMethod = Literal["none", "clamping", "back_calculation"]


@dataclass
class SystemModel:
    """Representation of a SISO LTI system."""

    model_type: ModelType
    # Transfer function coefficients: numerator(s) and denominator(s)
    num: Optional[List[float]] = None
    den: Optional[List[float]] = None
    # FOPDT parameters (if used)
    K: Optional[float] = None
    tau: Optional[float] = None
    theta: Optional[float] = None

    def is_transfer_function(self) -> bool:
        return self.model_type == "transfer_function"

    def is_fopdt(self) -> bool:
        return self.model_type == "fopdt"


@dataclass
class PIController:
    """PI controller parameters."""
    Kp: float
    Ki: float  # Alternatively use Ti = Kp/Ki
    u_min: Optional[float] = None
    u_max: Optional[float] = None
    anti_windup: AntiWindupMethod = "none"
    aw_gain: float = 1.0  # koristi se za back-calculation
    
    
@dataclass
class MPCConfig:
    """Basic MPC configuration (placeholder for later expansion)."""
    Ts: float
    Np: int
    Nc: int
    Q: float
    R: float
    u_min: Optional[float] = None
    u_max: Optional[float] = None


@dataclass
class SimulationConfig:
    """Configuration for closed-loop simulations."""
    Ts: float
    T_final: float
    setpoint: float = 1.0
    disturbance_time: Optional[float] = None
    disturbance_value: float = 0.0


@dataclass
class SimulationResult:
    """Container for simulation results."""
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    e: np.ndarray


# smart_tuner/models.py  (dodatak)

@dataclass
class MPCController:
    """
    Container for MPC-related matrices and configuration.

    For now this is a placeholder; later you can add everything
    needed for the QP formulation (H, f, Phi, Gamma, constraints…).
    """
    Ts: float
    Np: int
    Nc: int
    Q: float
    R: float

    # Discrete-time state-space model
    A_d: Optional[np.ndarray] = None
    B_d: Optional[np.ndarray] = None
    C_d: Optional[np.ndarray] = None
    D_d: Optional[np.ndarray] = None

    # Prediction matrices (for y = F x + G u, placeholders for later)
    Phy: Optional[np.ndarray] = None
    Gamma: Optional[np.ndarray] = None

    # Weighting parameters
    Q: float = 1.0
    R: float = 1.0

    # Input constraints (optional)
    u_min: Optional[float] = None
    u_max: Optional[float] = None
    du_min: Optional[float] = None
    du_max: Optional[float] = None
    
#    A_d: np.ndarray
#    B_d: np.ndarray
#    C_d: np.ndarray
#    D_d: Optional[np.ndarray] = None

   

