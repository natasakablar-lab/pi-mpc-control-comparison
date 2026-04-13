# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 05:04:21 2025

@author: Admin
"""

# app.py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
import zipfile
from datetime import datetime


# ----------------------------
# Imports from your package
# ----------------------------
# Adjust imports here if your package layout differs.
from smart_tuner import SystemModel, SimulationConfig, tune_pi
from smart_tuner.models import PIController, MPCConfig
from smart_tuner.tuning_mpc import tune_mpc
from smart_tuner.simulation import simulate_closed_loop_pi, simulate_closed_loop_mpc
from smart_tuner.performance import compute_iae

# ----------------------------
# Scenario helpers
# ----------------------------
def ref_profile(t: float, t_switch: float, r1: float, r2: float) -> float:
    return r1 if t < t_switch else r2

def dist_profile(t: float, t_dist: float, magnitude: float) -> float:
    return magnitude if t >= t_dist else 0.0

def run_nominal(model, pi_ctrl, mpc_ctrl, sim_cfg, run_pi=True, run_mpc=True):
    res_pi = simulate_closed_loop_pi(model, pi_ctrl, sim_cfg) if run_pi else None
    res_mpc = simulate_closed_loop_mpc(mpc_ctrl, sim_cfg) if run_mpc else None
    return res_pi, res_mpc

def run_reference_change(model, pi_ctrl, mpc_ctrl, Ts, T_final, u_min, u_max,
                         t_switch, r1, r2, run_pi=True, run_mpc=True):
    t = np.linspace(0.0, T_final, int(T_final / Ts) + 1)

    # PI (measurement uses y_true; reference changes)
    def sim_pi():
        y_true = np.zeros_like(t)
        y = np.zeros_like(t)
        u = np.zeros_like(t)
        e = np.zeros_like(t)
        integral = 0.0

        K = model.K
        tau = model.tau

        for k in range(len(t) - 1):
            r = ref_profile(t[k], t_switch, r1, r2)
            y_meas = y_true[k]
            y[k] = y_meas
            e[k] = r - y_meas

            u_unsat = pi_ctrl.Kp * e[k] + pi_ctrl.Ki * integral
            u[k] = max(u_min, min(u_max, u_unsat))

            # clamping AW
            if u[k] == u_unsat:
                integral += e[k] * Ts

            # plant update
            y_true[k + 1] = y_true[k] + (Ts / tau) * (-y_true[k] + K * u[k])

        y[-1] = y_true[-1]
        e[-1] = ref_profile(t[-1], t_switch, r1, r2) - y[-1]
        u[-1] = u[-2]
        return {"t": t, "y": y, "u": u, "e": e}

    # MPC (QP each step; reference changes)
    def sim_mpc():
        import cvxpy as cp

        y = np.zeros_like(t)
        u = np.zeros_like(t)
        e = np.zeros_like(t)

        A = mpc_ctrl.A_d
        B = mpc_ctrl.B_d
        C = mpc_ctrl.C_d

        # Depending on your naming (Phi or Phy)
        Phy = getattr(mpc_ctrl, "Phy", None)
        if Phy is None:
            Phy = getattr(mpc_ctrl, "Phy")

        Gamma = mpc_ctrl.Gamma

        nx = A.shape[0]
        x = np.zeros((nx,))

        Np = mpc_ctrl.Np
        Nc = mpc_ctrl.Nc
        Q = mpc_ctrl.Q
        R = mpc_ctrl.R

        U = cp.Variable(Nc)
        x_param = cp.Parameter(nx)

        for k in range(len(t) - 1):
            r = ref_profile(t[k], t_switch, r1, r2)
            r_vec = np.ones((Np,)) * r

            y0 = (C @ x).item()
            y[k] = y0
            e[k] = r - y0

            Y = Phy @ x_param + Gamma @ U
            cost = cp.sum_squares(np.sqrt(Q) * (Y - r_vec)) + cp.sum_squares(np.sqrt(R) * U)
            constraints = [U >= u_min, U <= u_max]
            prob = cp.Problem(cp.Minimize(cost), constraints)

            x_param.value = x
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            if U.value is None:
                raise RuntimeError("MPC QP failed (U.value is None).")

            u[k] = float(U.value[0])
            x = A @ x + B.flatten() * u[k]

        y[-1] = (C @ x).item()
        e[-1] = ref_profile(t[-1], t_switch, r1, r2) - y[-1]
        u[-1] = u[-2]
        return {"t": t, "y": y, "u": u, "e": e}

    res_pi = sim_pi() if run_pi else None
    res_mpc = sim_mpc() if run_mpc else None
    return res_pi, res_mpc

def run_disturbance(model, pi_ctrl, mpc_ctrl, Ts, T_final, u_min, u_max,
                    setpoint, t_dist, dmag, run_pi=True, run_mpc=True):
    t = np.linspace(0.0, T_final, int(T_final / Ts) + 1)

    # PI disturbance as measurement offset (no accumulation)
    def sim_pi():
        y_true = np.zeros_like(t)
        y = np.zeros_like(t)
        u = np.zeros_like(t)
        e = np.zeros_like(t)
        integral = 0.0

        K = model.K
        tau = model.tau

        for k in range(len(t) - 1):
            d = dist_profile(t[k], t_dist, dmag)
            y_meas = y_true[k] + d
            y[k] = y_meas
            e[k] = setpoint - y_meas

            u_unsat = pi_ctrl.Kp * e[k] + pi_ctrl.Ki * integral
            u[k] = max(u_min, min(u_max, u_unsat))

            if u[k] == u_unsat:
                integral += e[k] * Ts

            y_true[k + 1] = y_true[k] + (Ts / tau) * (-y_true[k] + K * u[k])

        d_last = dist_profile(t[-1], t_dist, dmag)
        y[-1] = y_true[-1] + d_last
        e[-1] = setpoint - y[-1]
        u[-1] = u[-2]
        return {"t": t, "y": y, "u": u, "e": e}

    # MPC disturbance (measurement offset; offset-free not included by design)
    def sim_mpc():
        import cvxpy as cp

        y = np.zeros_like(t)
        u = np.zeros_like(t)
        e = np.zeros_like(t)

        A = mpc_ctrl.A_d
        B = mpc_ctrl.B_d
        C = mpc_ctrl.C_d

        Phy = getattr(mpc_ctrl, "Phy", None)
        if Phi is None:
            Phy = getattr(mpc_ctrl, "Phy")

        Gamma = mpc_ctrl.Gamma

        nx = A.shape[0]
        x = np.zeros((nx,))

        Np = mpc_ctrl.Np
        Nc = mpc_ctrl.Nc
        Q = mpc_ctrl.Q
        R = mpc_ctrl.R

        U = cp.Variable(Nc)
        x_param = cp.Parameter(nx)

        for k in range(len(t) - 1):
            d = dist_profile(t[k], t_dist, dmag)
            y0 = (C @ x).item()
            y_meas = y0 + d

            y[k] = y_meas
            e[k] = setpoint - y_meas

            r_vec = np.ones((Np,)) * setpoint
            Y = Phy @ x_param + Gamma @ U
            cost = cp.sum_squares(np.sqrt(Q) * (Y - r_vec)) + cp.sum_squares(np.sqrt(R) * U)
            constraints = [U >= u_min, U <= u_max]
            prob = cp.Problem(cp.Minimize(cost), constraints)

            x_param.value = x
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            if U.value is None:
                raise RuntimeError("MPC QP failed (U.value is None).")

            u[k] = float(U.value[0])
            x = A @ x + B.flatten() * u[k]

        d_last = dist_profile(t[-1], t_dist, dmag)
        y[-1] = (C @ x).item() + d_last
        e[-1] = setpoint - y[-1]
        u[-1] = u[-2]
        return {"t": t, "y": y, "u": u, "e": e}

    res_pi = sim_pi() if run_pi else None
    res_mpc = sim_mpc() if run_mpc else None
    return res_pi, res_mpc

# ----------------------------
# Plot helper
# ----------------------------
def plot_results(res_pi, res_mpc, title_y, title_u, show_ref=None, show_umax=None, show_umin=None, vline=None):
    # y(t)
    fig1 = plt.figure()
    if res_pi is not None:
        plt.plot(res_pi["t"], res_pi["y"], label="y_PI(t)")
    if res_mpc is not None:
        plt.plot(res_mpc["t"], res_mpc["y"], label="y_MPC(t)")
    if show_ref is not None:
        tvec = res_pi["t"] if res_pi else res_mpc["t"]
        plt.plot(tvec, show_ref, "--", label="reference")
    if vline is not None:
        plt.axvline(vline, linestyle=":", linewidth=1.5, label="event")
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("y(t)")
    plt.title(title_y)
    plt.legend()
    st.pyplot(fig1)

    # u(t)
    fig2 = plt.figure()
    if res_pi is not None:
        plt.plot(res_pi["t"], res_pi["u"], label="u_PI(t)")
    if res_mpc is not None:
        plt.plot(res_mpc["t"], res_mpc["u"], label="u_MPC(t)")
    if show_umax is not None:
        plt.axhline(show_umax, linestyle="--", linewidth=1.0, label="u_max")
    if show_umin is not None:
        plt.axhline(show_umin, linestyle="--", linewidth=1.0, label="u_min")
    if vline is not None:
        plt.axvline(vline, linestyle=":", linewidth=1.5, label="event")
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("u(t)")
    plt.title(title_u)
    plt.legend()
    st.pyplot(fig2)

    return fig1, fig2


# dodato za EXPORT B)
def results_to_csv_bytes(res: dict) -> bytes:
    """res: {'t','y','u','e'} numpy arrays."""
    arr = np.column_stack([res["t"], res["y"], res["u"], res["e"]])
    header = "t,y,u,e"
    buf = io.StringIO()
    np.savetxt(buf, arr, delimiter=",", header=header, comments="")
    return buf.getvalue().encode("utf-8")


def fig_to_png_bytes(fig) -> bytes:
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=200, bbox_inches="tight")
    b.seek(0)
    return b.read()


def make_export_zip(res_pi, res_mpc, fig_y, fig_u, meta_text: str) -> bytes:
    """Creates a zip with CSV + PNG + meta txt."""
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.txt", meta_text)

        if res_pi is not None:
            z.writestr("results_PI.csv", results_to_csv_bytes(res_pi))
        if res_mpc is not None:
            z.writestr("results_MPC.csv", results_to_csv_bytes(res_mpc))

        if fig_y is not None:
            z.writestr("plot_output_y.png", fig_to_png_bytes(fig_y))
        if fig_u is not None:
            z.writestr("plot_control_u.png", fig_to_png_bytes(fig_u))

    zbuf.seek(0)
    return zbuf.read()

# ----------------------------
# Controller builder (needed for export/meta/metrics)
# ----------------------------
def build_controllers(K, tau, Ts, u_min, u_max, lam, anti_windup, Np, Nc, Q, R):
    """
    Returns: model, pi_ctrl, mpc_ctrl
    """
    model = SystemModel(model_type="fopdt", K=float(K), tau=float(tau), theta=0.0)

    # PI tuned + controller
    pi_tuned = tune_pi(model, method="imc", lam=float(lam))
    pi_ctrl = PIController(
        Kp=pi_tuned.Kp,
        Ki=pi_tuned.Ki,
        u_min=float(u_min),
        u_max=float(u_max),
        anti_windup=str(anti_windup),
    )

    # MPC tuned + controller
    mpc_cfg = MPCConfig(
        Ts=float(Ts),
        Np=int(Np),
        Nc=int(Nc),
        Q=float(Q),
        R=float(R),
        u_min=float(u_min),
        u_max=float(u_max),
    )
    mpc_ctrl = tune_mpc(model, mpc_cfg)

    return model, pi_ctrl, mpc_ctrl

if "ran" not in st.session_state:
    st.session_state["ran"] = False

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="SMART PI & MPC Tuner", layout="wide")
st.title("SMART PI & MPC Tuner (MVP)")
if "ran" not in st.session_state:
    st.session_state["ran"] = False

results_slot = st.empty()

st.caption("Compare PI (anti-windup) vs MPC (QP with input constraints) under nominal, reference-change, and disturbance scenarios.")

submitted = False

with st.sidebar:
    st.header("Scenario")

    if submitted:
        st.session_state["ran"] = True


    with st.form("params_form"):
        scenario = st.selectbox("Select scenario", ["Nominal", "Reference change", "Disturbance rejection"])

        st.header("Plant (FOPDT, no delay)")
        K = st.number_input("Gain K", value=2.0, step=0.1)
        tau = st.number_input("Time constant τ", value=5.0, step=0.1)

        st.header("Simulation")
        Ts = st.number_input("Sampling time Ts [s]", value=0.1, step=0.05)
        T_final = st.number_input("Final time [s]", value=50.0, step=5.0)
        setpoint = st.number_input("Setpoint (nominal)", value=1.0, step=0.1)

        st.header("Constraints")
        u_min = st.number_input("u_min", value=0.0, step=0.1)
        u_max = st.number_input("u_max", value=0.6, step=0.1)

        st.header("Controllers")
        run_pi = st.checkbox("Run PI", value=True)
        run_mpc = st.checkbox("Run MPC", value=True)

        st.subheader("PI settings")
        lam = st.number_input("IMC λ", value=2.0, step=0.1)
        anti_windup = st.selectbox("Anti-windup", ["clamping", "none"])

        st.subheader("MPC settings")
        Np = st.slider("Np (prediction horizon)", min_value=5, max_value=60, value=20, step=1)
        Nc = st.slider("Nc (control horizon)", min_value=1, max_value=60, value=10, step=1)
        Q = st.number_input("Q (tracking weight)", value=1.0, step=0.1)
        R = st.number_input("R (effort weight)", value=0.1, step=0.05)

        if scenario == "Reference change":
            st.subheader("Reference change")
            t_switch = st.number_input("Switch time [s]", value=25.0, step=1.0)
            r1 = st.number_input("r1 (before)", value=1.0, step=0.1)
            r2 = st.number_input("r2 (after)", value=0.5, step=0.1)

        if scenario == "Disturbance rejection":
            st.subheader("Disturbance")
            t_dist = st.number_input("Disturbance time [s]", value=25.0, step=1.0)
            dmag = st.number_input("Disturbance magnitude", value=0.2, step=0.05)

        submitted = st.form_submit_button("Run simulation")

    if submitted:
        st.session_state["ran"] = True
        

if submitted:
    st.session_state["ran"] = True

go = st.button("Run simulation", type="primary")

if "ran" not in st.session_state:
    st.session_state["ran"] = False

if "last_results" not in st.session_state:
    st.session_state["last_results"] = None



if st.session_state["ran"]:
    model, pi_ctrl, mpc_ctrl = build_controllers(
        K, tau, Ts, u_min, u_max,
        lam, anti_windup,
        Np, Nc, Q, R
    )
    # simulacija, plot, export...
else:
    st.info("Izaberi parametre u sidebar-u i klikni **Run simulation**.")

if not st.session_state["ran"]:
    results_slot.empty()
    results_slot.info("Izaberi parametre u sidebar-u i klikni **Run simulation**.")
else:
    results_slot.empty()
    with results_slot.container():

        # Build model & controllers
        model, pi_ctrl, mpc_ctrl = build_controllers(
            K, tau, Ts, u_min, u_max,
            lam, anti_windup,
            Np, Nc, Q, R
        )

# Build model & controllers (on demand)

model, pi_ctrl, mpc_ctrl = build_controllers(
    K, tau, Ts, u_min, u_max,
    lam, anti_windup,
    Np, Nc, Q, R
    )

    # PI tuned + constructed
pi_tuned = tune_pi(model, method="imc", lam=float(lam))
pi_ctrl = PIController(
        Kp=pi_tuned.Kp,
        Ki=pi_tuned.Ki,
        u_min=float(u_min),
        u_max=float(u_max),
        anti_windup=anti_windup,
    )

    # MPC tuned + constructed
mpc_cfg = MPCConfig(
        Ts=float(Ts),
        Np=int(Np),
        Nc=int(Nc),
        Q=float(Q),
        R=float(R),
        u_min=float(u_min),
        u_max=float(u_max),
    )
mpc_ctrl = tune_mpc(model, mpc_cfg)
    
    

    # Run scenario
if scenario == "Nominal":
        sim_cfg = SimulationConfig(Ts=float(Ts), T_final=float(T_final), setpoint=float(setpoint))
        res_pi_obj, res_mpc_obj = run_nominal(model, pi_ctrl, mpc_ctrl, sim_cfg, run_pi=run_pi, run_mpc=run_mpc)

        # normalize to dict
        res_pi = {"t": res_pi_obj.t, "y": res_pi_obj.y, "u": res_pi_obj.u, "e": res_pi_obj.e} if res_pi_obj else None
        res_mpc = {"t": res_mpc_obj.t, "y": res_mpc_obj.y, "u": res_mpc_obj.u, "e": res_mpc_obj.e} if res_mpc_obj else None

        ref_line = np.ones_like(res_pi["t"] if res_pi else res_mpc["t"]) * float(setpoint)
        fig_y, fig_u = plot_results(res_pi, res_mpc,
                     title_y="Closed-loop output (Nominal): PI vs MPC",
                     title_u="Control input (Nominal): PI vs MPC",
                     show_ref=ref_line, show_umax=float(u_max), show_umin=float(u_min))

elif scenario == "Reference change":
        res_pi, res_mpc = run_reference_change(model, pi_ctrl, mpc_ctrl,
                                              Ts=float(Ts), T_final=float(T_final),
                                              u_min=float(u_min), u_max=float(u_max),
                                              t_switch=float(t_switch), r1=float(r1), r2=float(r2),
                                              run_pi=run_pi, run_mpc=run_mpc)

        tvec = res_pi["t"] if res_pi else res_mpc["t"]
        ref_line = np.array([ref_profile(tt, float(t_switch), float(r1), float(r2)) for tt in tvec])

        fig_y, fig_u = plot_results(res_pi, res_mpc,
                     title_y="Closed-loop output (Reference change): PI vs MPC",
                     title_u="Control input (Reference change): PI vs MPC",
                     show_ref=ref_line, show_umax=float(u_max), show_umin=float(u_min),
                     vline=float(t_switch))

else:  # Disturbance rejection
        res_pi, res_mpc = run_disturbance(model, pi_ctrl, mpc_ctrl,
                                          Ts=float(Ts), T_final=float(T_final),
                                          u_min=float(u_min), u_max=float(u_max),
                                          setpoint=float(setpoint),
                                          t_dist=float(t_dist), dmag=float(dmag),
                                          run_pi=run_pi, run_mpc=run_mpc)

        tvec = res_pi["t"] if res_pi else res_mpc["t"]
        ref_line = np.ones_like(tvec) * float(setpoint)

        fig_y, fig_u = plot_results(res_pi, res_mpc,
                     title_y="Closed-loop output (Disturbance rejection): PI vs MPC",
                     title_u="Control input (Disturbance rejection): PI vs MPC",
                     show_ref=ref_line, show_umax=float(u_max), show_umin=float(u_min),
                     vline=float(t_dist))

    # Metrics
col1, col2 = st.columns(2)
if run_pi and res_pi is not None:
        iae_pi = np.trapezoid(np.abs(res_pi["e"]), res_pi["t"])
        col1.metric("IAE (PI)", f"{iae_pi:.4f}")
if run_mpc and res_mpc is not None:
        iae_mpc = np.trapezoid(np.abs(res_mpc["e"]), res_mpc["t"])
        col2.metric("IAE (MPC)", f"{iae_mpc:.4f}")

else:
    st.info("Set parameters in the sidebar and click **Run simulation**.")
    
meta = []
meta.append(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
meta.append(f"scenario: {scenario}")
meta.append(f"plant: K={K}, tau={tau}, Ts={Ts}, T_final={T_final}")
meta.append(f"constraints: u_min={u_min}, u_max={u_max}")
meta.append(f"PI: lam={lam}, anti_windup={anti_windup}, Kp={pi_ctrl.Kp:.6g}, Ki={pi_ctrl.Ki:.6g}")
meta.append(f"MPC: Np={Np}, Nc={Nc}, Q={Q}, R={R}")
if scenario == "Reference change":
    meta.append(f"reference change: t_switch={t_switch}, r1={r1}, r2={r2}")
if scenario == "Disturbance rejection":
    meta.append(f"disturbance: t_dist={t_dist}, dmag={dmag}")

zip_bytes = make_export_zip(res_pi if run_pi else None,
                            res_mpc if run_mpc else None,
                            fig_y, fig_u,
                            meta_text="\n".join(meta))

st.download_button(
    label="⬇️ Export results (ZIP: CSV + PNG + meta)",
    data=zip_bytes,
    file_name=f"smart_pi_mpc_export_{scenario.replace(' ', '_').lower()}.zip",
    mime="application/zip",
)

st.success("Simulacija pokrenuta. Rezultati su prikazani ispod.")