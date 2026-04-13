# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:14:14 2025

@author: Admin
"""

# smart_tuner/tuning_pi.py
from __future__ import annotations
from typing import Literal

from .models import SystemModel, PIController

PIMethod = Literal["ziegler_nichols", "imc"]  # proširićemo kasnije


def tune_pi(
    model: SystemModel,
    method: PIMethod = "imc",
    lam: float = 1.0,
) -> PIController:
    """
    High-level entry point for PI tuning.

    Parameters
    ----------
    model : SystemModel
        Identified or given system model (ideally FOPDT for now).
    method : PIMethod
        Tuning method to use.
    lam : float
        IMC filter parameter (larger = more robust but slower).

    Returns
    -------
    PIController
        Tuned PI parameters.
    """
    if method == "imc":
        return _tune_pi_imc_fopdt(model, lam=lam)
    elif method == "ziegler_nichols":
        # TODO: implement Ziegler–Nichols
        raise NotImplementedError("Ziegler–Nichols not implemented yet.")
    else:
        raise ValueError(f"Unknown PI tuning method: {method}")


def _tune_pi_imc_fopdt(model: SystemModel, lam: float) -> PIController:
    """
    IMC-based PI tuning for a FOPDT model.

    Assumes: G(s) = K * exp(-theta s) / (tau s + 1)

    This is a simple starting point; formulas can be refined.
    """
    if not model.is_fopdt():
        raise ValueError("IMC FOPDT tuning requires a FOPDT model.")

    if model.K is None or model.tau is None:
        raise ValueError("Model K and tau must be set for FOPDT.")

    K = model.K
    tau = model.tau
    theta = model.theta or 0.0

    # Very basic IMC PI tuning rules (placeholder – možeš prilagoditi kasnije)
    # Reference: standard IMC tuning for FOPDT
    # Kp = (tau / (K * (lam + theta)))
    # Ti = min(tau, 4 * (lam + theta))

    Kp = tau / (K * (lam + theta))
    Ti = min(tau, 4.0 * (lam + theta))

    Ki = Kp / Ti

    return PIController(Kp=Kp, Ki=Ki)
