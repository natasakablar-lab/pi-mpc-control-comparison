# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:21:09 2025

@author: Admin
"""

# smart_tuner/tuning_mpc.py
from __future__ import annotations
from typing import Tuple
import numpy as np
from dataclasses import dataclass

from .models import SystemModel, MPCConfig, MPCController



def tune_mpc(model: SystemModel, config: MPCConfig) -> MPCController:
    A_d, B_d, C_d, D_d = _discretize_model_fopdt(model, Ts=config.Ts)

    Phy, Gamma = _build_prediction_matrices(A_d, B_d, C_d, config.Np, config.Nc)

    return MPCController(
        Ts=config.Ts,
        Np=config.Np,
        Nc=config.Nc,
        Q=config.Q,
        R=config.R,
        A_d=A_d,
        B_d=B_d,
        C_d=C_d,
        D_d=D_d,
        Phy=Phy,
        Gamma=Gamma,
        u_min=config.u_min,
        u_max=config.u_max,
    )


def _discretize_model_fopdt(model: SystemModel, Ts: float):
    if not model.is_fopdt():
        raise ValueError("For MPC v1, discretization assumes FOPDT (no delay).")

    K = model.K or 1.0
    tau = model.tau or 1.0

    a = np.exp(-Ts / tau)
    b = K * (1.0 - np.exp(-Ts / tau))

    A_d = np.array([[a]])
    B_d = np.array([[b]])
    C_d = np.array([[1.0]])
    D_d = np.array([[0.0]])
    
    return A_d, B_d, C_d, D_d


def _build_prediction_matrices(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, Np: int, Nc: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict outputs:
      y_{k+i} = C A^i x_k + sum_{j=0}^{i-1} C A^{i-1-j} B u_{k+j}
    We'll build for i=1..Np (predict next Np outputs),
    so Y = F x_k + G U, where U = [u_k, ..., u_{k+Nc-1}]^T.
    Assumption v1: Nc <= Np (typical).
    """
    nx = A.shape[0]
    nu = B.shape[1]

    if nu != 1:
        raise ValueError("This MPC v1 expects SISO (nu=1).")

    Phy = np.zeros((Np, nx))
    Gamma = np.zeros((Np, Nc))

    # Precompute powers of A
    A_powers = [np.eye(nx)]
    for i in range(1, Np + 1):
        A_powers.append(A_powers[-1] @ A)

    # F row i-1 corresponds to prediction at step i (y_{k+i})
    for i in range(1, Np + 1):
        Phy[i - 1, :] = (C @ A_powers[i]).reshape(-1)

        for j in range(0, min(i, Nc)):
            # contribution of u_{k+j} to y_{k+i}
            # term: C A^{i-1-j} B
            Gamma[i - 1, j] = float(C @ A_powers[i - 1 - j] @ B)

    return Phy, Gamma
