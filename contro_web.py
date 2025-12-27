import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
from pyswarm import pso
import matplotlib.pyplot as plt
import io
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="OptiCTRL Studio", layout="wide", page_icon="🎛️")

# Hide code-like elements to make it look like a SaaS product
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #f4f6f9;}
    div.stButton > button:first-child {background-color: #0068c9; color: white; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("🎛️ OptiCTRL: Control Tuning Studio")
st.markdown("### Automated PID & SMC Tuning Portal")

# --- UTILS ---
def get_step_info(t, y, r_final):
    info = {}
    try:
        # Rise Time (10-90%)
        t10 = t[np.where(y >= 0.1 * r_final)[0][0]]
        t90 = t[np.where(y >= 0.9 * r_final)[0][0]]
        info["Rise Time (s)"] = t90 - t10
    except: info["Rise Time (s)"] = None
    
    # Overshoot
    ymax = np.max(y)
    info["Overshoot (%)"] = ((ymax - r_final)/r_final)*100 if ymax > r_final else 0
    
    # Settling Time (2%)
    try:
        mask = np.abs(y - r_final) > 0.02*r_final
        info["Settling Time (s)"] = t[np.where(mask)[0][-1]]
    except: info["Settling Time (s)"] = None
    
    return info

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    sim_time = st.number_input("Simulation Time (s)", 5.0, 100.0, 10.0)
    target_val = st.number_input("Target Reference (Step)", 1.0, 100.0, 1.0)
    
    st.divider()
    st.subheader("Optimizer Settings")
    algo = st.selectbox("Algorithm", ["PSO (Swarm)", "Differential Evolution"])
    iter_count = st.slider("Iterations", 10, 100, 30)

# --- MAIN LAYOUT ---
col_in, col_out = st.columns([1, 1.5])

with col_in:
    with st.expander("1. Plant System Model", expanded=True):
        sys_type = st.radio("Model Type", ["Transfer Function (Linear)", "Custom ODE (Non-Linear)"])
        
        if sys_type == "Transfer Function (Linear)":
            num_str = st.text_input("Numerator (e.g., 1)", "1")
            den_str = st.text_input("Denominator (e.g., 1, 2, 1)", "1, 2, 1")
        else:
            st.info("Write Python math: e.g., -0.5*x1 + u")
            eq1 = st.text_input("dx1/dt =", "x2")
            eq2 = st.text_input("dx2/dt =", "-0.5*x1 - 0.1*x2 + u")
            x0_str = st.text_input("Initial Conditions (0,0)", "0, 0")

    with st.expander("2. Controller Design", expanded=True):
        ctrl_mode = st.selectbox("Controller", ["PID", "PI", "PD", "P"])
        st.caption("The software will optimize these gains:")
        
        # Dynamic Bounds
        bounds = {}
        p_list = ["Kp", "Ki", "Kd"] if ctrl_mode == "PID" else ["Kp", "Ki"] if ctrl_mode == "PI" else ["Kp", "Kd"] if ctrl_mode == "PD" else ["Kp"]
        
        for p in p_list:
            c1, c2 = st.columns(2)
            mn = c1.number_input(f"Min {p}", 0.0, 10.0, 0.0)
            mx = c2.number_input(f"Max {p}", 0.1, 100.0, 20.0)
            bounds[p] = (mn, mx)

    run_btn = st.button("🚀 Optimize Controller", use_container_width=True)

# --- EXECUTION ---
if run_btn:
    with col_out:
        status = st.status("Initializing AI Optimizer...", expanded=True)
        
        # Objective Function
        def objective(params):
            try:
                t = np.linspace(0, sim_time, 200)
                if sys_type == "Transfer Function (Linear)":
                    n = [float(x) for x in num_str.split(",")]
                    d = [float(x) for x in den_str.split(",")]
                    plant = signal.TransferFunction(n, d)
                    
                    if ctrl_mode=="PID": c=signal.TransferFunction([params[2], params[0], params[1]], [1,0])
                    elif ctrl_mode=="PI": c=signal.TransferFunction([params[0], params[1]], [1,0])
                    elif ctrl_mode=="PD": c=signal.TransferFunction([params[1], params[0]], [1])
                    else: c=signal.TransferFunction([params[0]], [1])
                    
                    cl = signal.feedback(c*plant)
                    _, y, _ = signal.lsim(cl, U=np.ones_like(t)*target_val, T=t)
                
                else:
                    # Non-Linear Parser
                    x0 = [float(x) for x in x0_str.split(",")]
                    def model(t, x):
                        x1 = x[0]; x2 = x[1] if len(x)>1 else 0
                        e = target_val - x1
                        if ctrl_mode=="PID": u = params[0]*e # Simplified P for ODE demo
                        else: u = params[0]*e
                        
                        ctx = {"x1": x1, "x2": x2, "u": u, "sin": np.sin, "cos": np.cos}
                        return [eval(eq1, ctx), eval(eq2, ctx)]
                    
                    sol = solve_ivp(model, [0, sim_time], x0, t_eval=t)
                    y = sol.y[0]

                # Cost: ISE (Integral Square Error)
                ise = np.sum((target_val - y)**2)
                return ise if not np.isnan(ise) else 1e6
            except: return 1e6

        # Optimization
        status.write("Running optimization algorithms...")
        lb = [v[0] for v in bounds.values()]
        ub = [v[1] for v in bounds.values()]
        
        if algo == "PSO":
            x_opt, f_opt = pso(objective, lb, ub, swarmsize=20, maxiter=iter_count)
        else:
            res = differential_evolution(objective, bounds=list(zip(lb, ub)))
            x_opt = res.x
        
        status.update(label="Optimization Complete!", state="complete", expanded=False)
        
        # --- RESULTS ---
        st.success("Controller Tuned Successfully")
        
        # 1. Final Plot
        t_final = np.linspace(0, sim_time, 500)
        # (Re-simulating for plot - simplified for Linear logic here)
        n = [float(x) for x in num_str.split(",")]
        d = [float(x) for x in den_str.split(",")]
        plant = signal.TransferFunction(n, d)
        if ctrl_mode=="PID": c=signal.TransferFunction([x_opt[2], x_opt[0], x_opt[1]], [1,0])
        elif ctrl_mode=="PI": c=signal.TransferFunction([x_opt[0], x_opt[1]], [1,0])
        elif ctrl_mode=="PD": c=signal.TransferFunction([x_opt[1], x_opt[0]], [1])
        else: c=signal.TransferFunction([x_opt[0]], [1])
        cl = signal.feedback(c*plant)
        _, y_final, _ = signal.lsim(cl, U=np.ones_like(t_final)*target_val, T=t_final)
        
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(t_final, np.ones_like(t_final)*target_val, 'k--', label="Target")
        ax.plot(t_final, y_final, 'b-', linewidth=2, label="Optimized Response")
        ax.set_title("Step Response")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
        # 2. Metrics & Params
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Optimized Gains")
            df_p = pd.DataFrame({"Param": list(bounds.keys()), "Value": x_opt})
            st.dataframe(df_p, hide_index=True)
        with c2:
            st.subheader("Performance")
            info = get_step_info(t_final, y_final, target_val)
            for k,v in info.items():
                st.metric(k, f"{v:.3f}" if v else "N/A")