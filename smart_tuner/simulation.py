# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 10:20:36 2025

@author: Admin
"""

# smart_tuner/simulation.py
from __future__ import annotations


from typing import Optional
import numpy as np

# smart_tuner/simulation.py  (dodaj ispod PI dela)
import numpy as np

from .models import MPCController, SimulationConfig, SimulationResult

from .models import (
    SystemModel,
    PIController,
    SimulationConfig,
    SimulationResult,
    MPCController,  # ako nemaš MPCController još, obriši ovu liniju
)


def _clip(u_val: float, u_min: Optional[float], u_max: Optional[float]) -> float:
    """Clip control input to optional saturation limits."""
    if u_min is not None:
        u_val = max(u_val, u_min)
    if u_max is not None:
        u_val = min(u_val, u_max)
    return u_val


def simulate_closed_loop_pi(
    model: SystemModel,
    controller: PIController,
    config: SimulationConfig,
) -> SimulationResult:
    """
    Discrete-time closed-loop simulation with a PI controller.

    Current assumptions (v1):
    - plant model: FOPDT without delay in the simulation step (theta ignored here)
      G(s) = K / (tau s + 1)
    - discretization: simple forward Euler on the first-order dynamics
      y[k+1] = y[k] + (Ts/tau)*(-y[k] + K*u[k])

    Supports:
    - input saturation via controller.u_min, controller.u_max
    - anti-windup via controller.anti_windup: "none" | "clamping" | "back_calculation"
    """
    Ts = config.Ts
    T_final = config.T_final
    setpoint = config.setpoint

    n_steps = int(T_final / Ts) + 1
    t = np.linspace(0.0, T_final, n_steps)

    y = np.zeros_like(t)
    u = np.zeros_like(t)
    e = np.zeros_like(t)

    # PI integrator state
    integral = 0.0

    if not model.is_fopdt():
        raise ValueError("simulate_closed_loop_pi currently supports only FOPDT models.")

    K = model.K if model.K is not None else 1.0
    tau = model.tau if model.tau is not None else 1.0

    for k in range(n_steps - 1):
        # error
        e[k] = setpoint - y[k]

        # unsaturated PI output (using current integral state)
        u_unsat = controller.Kp * e[k] + controller.Ki * integral

        # apply saturation limits if any
        u_sat = _clip(u_unsat, controller.u_min, controller.u_max)

        # anti-windup handling
        aw = getattr(controller, "anti_windup", "none")

        if aw == "none":
            integral += e[k] * Ts

        elif aw == "clamping":
            # Update integrator only if:
            # - not saturated, OR
            # - saturated high but error is negative (will reduce u), OR
            # - saturated low but error is positive (will increase u)
            is_saturated = (u_sat != u_unsat)

            if not is_saturated:
                integral += e[k] * Ts
            else:
                if (controller.u_max is not None and u_sat >= controller.u_max and e[k] < 0):
                    integral += e[k] * Ts
                elif (controller.u_min is not None and u_sat <= controller.u_min and e[k] > 0):
                    integral += e[k] * Ts
                # else: freeze integrator

        elif aw == "back_calculation":
            # integral_dot = e + aw_gain*(u_sat - u_unsat)
            aw_gain = getattr(controller, "aw_gain", 1.0)
            integral += (e[k] + aw_gain * (u_sat - u_unsat)) * Ts

        else:
            raise ValueError(f"Unknown anti-windup method: {aw}")

        # final applied control
        u[k] = u_sat

        # plant update (first-order)
        y[k + 1] = y[k] + (Ts / tau) * (-y[k] + K * u[k])

    # last samples
    e[-1] = setpoint - y[-1]
    u[-1] = u[-2]

    return SimulationResult(t=t, y=y, u=u, e=e)


# -----------------------------
# OPTIONAL: MPC skeleton (keep if you already use it; otherwise remove)
# -----------------------------
def simulate_closed_loop_mpc_qp(
    controller: MPCController,
    config: SimulationConfig,
    x0: Optional[np.ndarray] = None,
) -> SimulationResult:
    """
    True MPC closed-loop simulation using a QP solved by cvxpy.

    Decision variable: U = [u_k, ..., u_{k+Nc-1}] (absolute inputs)
    Predicted outputs: Y = F x_k + G U
    Cost: sum Q*(Y - Rref)^2 + sum R*(U)^2
    Constraints: u_min <= U <= u_max (optional)
    """
    import cvxpy as cp  # local import so PI still runs if cvxpy missing

    Ts = config.Ts
    T_final = config.T_final
    setpoint = config.setpoint

    n_steps = int(T_final / Ts) + 1
    t = np.linspace(0.0, T_final, n_steps)

    y = np.zeros_like(t)
    u = np.zeros_like(t)
    e = np.zeros_like(t)

    A = controller.A_d
    B = controller.B_d
    C = controller.C_d

    nx = A.shape[0]
    x = np.zeros((nx,)) if x0 is None else x0.copy()

    Phy = controller.Phy
    Gamma = controller.Gamma
    if Phy is None or Gamma is None:
        raise ValueError("MPCController must contain prediction matrices Phy and Gamma. Call tune_mpc first.")

    Np = controller.Np
    Nc = controller.Nc
    Q = controller.Q
    Rw = controller.R

    # reference trajectory over horizon
    r_vec = np.ones((Np,)) * setpoint

    # Build cvxpy problem once (parametrized by x)
    U = cp.Variable(Nc)
    x_param = cp.Parameter(nx)

    Y = Phy @ x_param + Gamma @ U
    cost = cp.sum_squares(np.sqrt(Q) * (Y - r_vec)) + cp.sum_squares(np.sqrt(Rw) * U)

    constraints = []
    if controller.u_min is not None:
        constraints.append(U >= controller.u_min)
    if controller.u_max is not None:
        constraints.append(U <= controller.u_max)

    prob = cp.Problem(cp.Minimize(cost), constraints)

    # closed-loop simulation
    for k in range(n_steps - 1):
        y[k] = float(C @ x)
        e[k] = setpoint - y[k]

        x_param.value = x

        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if U.value is None:
            raise RuntimeError("MPC QP failed to solve. Try relaxing constraints or check cvxpy installation.")

        u[k] = float(U.value[0])

        # state update
        x = (A @ x) + (B.flatten() * u[k])

    y[-1] = float(C @ x)
    e[-1] = setpoint - y[-1]
    u[-1] = u[-2]

    return SimulationResult(t=t, y=y, u=u, e=e)

# Alias for backward compatibility
def simulate_closed_loop_mpc(*args, **kwargs):
    return simulate_closed_loop_mpc_qp(*args, **kwargs)
