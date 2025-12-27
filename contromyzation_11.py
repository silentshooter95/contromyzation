import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import io
import traceback

# --- 1. PAGE CONFIGURATION (Must be the first command) ---
st.set_page_config(page_title="ControMyzation Studio", layout="wide", page_icon="🎛️")

# Custom CSS to make it look like professional software
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    h1 {color: #007bff;}
    div.stButton > button {width: 100%; background-color: #28a745; color: white; font-weight: bold; border: none; padding: 10px;}
    div.stButton > button:hover {background-color: #218838;}
    .reportview-container .main .block-container{padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("🎛️ ControMyzation: Web Tuning Studio")
st.markdown("---")

# --- 2. HELPER FUNCTIONS ---

# Custom Particle Swarm Optimization (Pure Python implementation)
def custom_pso(func, lb, ub, args=(), swarmsize=20, maxiter=50):
    dim = len(lb)
    lb = np.array(lb); ub = np.array(ub)
    
    positions = np.random.uniform(low=lb, high=ub, size=(swarmsize, dim))
    velocities = np.zeros((swarmsize, dim))
    pbest_pos = positions.copy()
    pbest_scores = np.full(swarmsize, np.inf)
    gbest_pos = np.zeros(dim)
    gbest_score = np.inf
    
    w, c1, c2 = 0.5, 1.5, 1.5
    
    # Progress bar in the UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(maxiter):
        status_text.caption(f"Optimizer Progress: {i+1}/{maxiter}")
        progress_bar.progress((i + 1) / maxiter)
        
        for j in range(swarmsize):
            try:
                score = func(positions[j], *args)
            except Exception:
                score = 1e9 # Penalty for crashing parameters
            
            if score < pbest_scores[j]:
                pbest_scores[j] = score
                pbest_pos[j] = positions[j]
                if score < gbest_score:
                    gbest_score = score
                    gbest_pos = positions[j]
        
        r1 = np.random.rand(swarmsize, dim)
        r2 = np.random.rand(swarmsize, dim)
        velocities = (w * velocities + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions))
        positions = positions + velocities
        positions = np.clip(positions, lb, ub)
        
    progress_bar.empty()
    status_text.empty()
    return gbest_pos, gbest_score

# Controller Logic Templates
CONTROLLER_TEMPLATES = {
    "SOSMC (Super-Twisting)": {
        "params": ["k1", "k2"], "c0": "0",
        "code": """# Super-Twisting Logic
k1, k2 = params[0], params[1]
e = r - x[0]
sigma = c[0]
v = -k1 * np.sqrt(np.abs(e)) * np.sign(e) + sigma
sigma_dot = -k2 * np.sign(e)
return v, np.array([sigma_dot])"""
    },
    "TSMC (Terminal SMC)": {
        "params": ["c_tsmc", "beta"], "c0": "",
        "code": """# Terminal SMC Logic
c, beta = params[0], params[1]
e1 = r - x[0]
e2 = -x[1]
s = e2 + c * (np.abs(e1)**beta) * np.sign(e1)
u = 5.0 * np.tanh(s * 10)
return u, np.array([])"""
    },
    "SMC (Standard)": {
        "params": ["K_smc", "lambda_smc"], "c0": "",
        "code": """# Standard SMC Logic
K, lam = params[0], params[1]
e = r - x[0]
s = -x[1] + lam * e
u = K * np.sign(s)
return u, np.array([])"""
    }
}

def calculate_metrics(t, y, r):
    metrics = {}
    r_final = r[-1] if hasattr(r, "__len__") else r
    
    if r_final == 0: return {"Status": "Ref is 0"}

    # Rise Time
    try:
        t10 = t[np.where(y >= 0.1 * r_final)[0][0]]
        t90 = t[np.where(y >= 0.9 * r_final)[0][0]]
        metrics["Rise Time"] = f"{t90 - t10:.3f} s"
    except: metrics["Rise Time"] = "N/A"
        
    # Overshoot
    y_max = np.max(y)
    overshoot = ((y_max - r_final) / r_final) * 100
    metrics["Overshoot"] = f"{overshoot:.2f} %" if overshoot > 0 else "0 %"
    
    # Settling Time (2%)
    try:
        threshold = 0.02 * r_final
        unsettled = np.where(np.abs(y - r_final) > threshold)[0]
        if len(unsettled) > 0:
            metrics["Settling Time"] = f"{t[unsettled[-1]]:.3f} s"
        else:
            metrics["Settling Time"] = f"{t[0]:.3f} s"
    except: metrics["Settling Time"] = "N/A"
    
    return metrics

# --- 3. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ System Settings")
    system_mode = st.selectbox("System Type", ["Linear (Transfer Function)", "Non-Linear (State Space)"])
    
    st.divider()
    st.header("🧠 Optimizer")
    optimizer = st.selectbox("Algorithm", ["Custom PSO", "Differential Evolution", "L-BFGS-B"])
    
    if optimizer == "Custom PSO":
        col_s1, col_s2 = st.columns(2)
        swarm_size = col_s1.number_input("Swarm Size", 10, 100, 20)
        max_iter = col_s2.number_input("Iterations", 10, 500, 30)
    
    st.divider()
    sim_time = st.number_input("Simulation Duration (s)", 1.0, 100.0, 10.0)

# --- 4. MAIN LAYOUT ---
col1, col2 = st.columns([1, 1.2])

with col1:
    with st.container():
        st.subheader("1. Plant Model")
        if system_mode.startswith("Linear"):
            num_str = st.text_input("Numerator Coefficients", "1")
            den_str = st.text_input("Denominator Coefficients", "1, 2, 1")
        else:
            st.info("Model: x_dot = f(x, u). Output y = x[0].")
            nl_x0_str = st.text_input("Initial Conditions (x0)", "0, 0")
            default_plant = "x1, x2 = x\nreturn np.array([x2, -0.5*x1 - 0.1*x2 + u])"
            nl_plant_code = st.text_area("def plant_model(t, x, u):", default_plant, height=120)

        st.subheader("2. Controller Design")
        if system_mode.startswith("Linear"):
            ctrl_type = st.selectbox("Controller Type", ["PID", "PI", "PD"])
            if ctrl_type == "PID": params = ["Kp", "Ki", "Kd"]
            elif ctrl_type == "PI": params = ["Kp", "Ki"]
            else: params = ["Kp", "Kd"]
        else:
            template_name = st.selectbox("Load Controller Template", list(CONTROLLER_TEMPLATES.keys()))
            temp = CONTROLLER_TEMPLATES[template_name]
            nl_c0_str = st.text_input("Controller Initial State (c0)", temp["c0"])
            nl_ctrl_code = st.text_area("def controller(t,x,r,c,params):", temp["code"], height=150)
            params = temp["params"]

        st.subheader("3. Tuning Bounds")
        bounds = {}
        b_cols = st.columns(2)
        for i, p in enumerate(params):
            with b_cols[i % 2]:
                min_v = st.number_input(f"Min {p}", 0.0, 1000.0, 0.1)
                max_v = st.number_input(f"Max {p}", 0.1, 1000.0, 20.0)
                bounds[p] = (min_v, max_v)

        st.write("") # Spacer
        run_btn = st.button("🚀 START OPTIMIZATION")

# --- 5. EXECUTION LOGIC ---
if run_btn:
    with col2:
        st.info("Initializing simulation engine...")
        
        try:
            # Prepare Bounds
            lb = [bounds[p][0] for p in params]
            ub = [bounds[p][1] for p in params]
            
            # --- DEFINE OBJECTIVE FUNCTION ---
            def objective(p_vals):
                t = np.linspace(0, sim_time, 200)
                try:
                    if system_mode.startswith("Linear"):
                        # Linear Simulation
                        n = [float(x) for x in num_str.split(",")]
                        d = [float(x) for x in den_str.split(",")]
                        plant = signal.TransferFunction(n, d)
                        
                        kp, ki = p_vals[0], p_vals[1]
                        kd = p_vals[2] if len(p_vals) > 2 else 0
                        
                        c = signal.TransferFunction([kd, kp, ki], [1, 0])
                        cl = signal.feedback(c * plant)
                        
                        _, y, _ = signal.lsim(cl, U=np.ones_like(t), T=t)
                        e = 1.0 - y
                        
                    else:
                        # Non-Linear Simulation
                        x0 = [float(x) for x in nl_x0_str.split(",")]
                        c0 = [float(x) for x in nl_c0_str.split(",")] if nl_c0_str.strip() else []
                        
                        # Dynamic Code Execution (Safe Scope)
                        scope = {"np": np, "params": p_vals}
                        exec(f"def plant(t, x, u):\n{nl_plant_code}", scope)
                        exec(f"def ctrl(t, x, r, c, params):\n{nl_ctrl_code}", scope)
                        
                        def model(t, y_vec):
                            x = y_vec[:len(x0)]; c = y_vec[len(x0):]
                            u, c_dot = scope['ctrl'](t, x, 1.0, c, p_vals)
                            x_dot = scope['plant'](t, x, u)
                            if not isinstance(c_dot, (np.ndarray, list)): c_dot = [c_dot]
                            return np.concatenate([x_dot, c_dot])
                            
                        y0_full = np.concatenate([x0, c0])
                        sol = solve_ivp(model, [0, sim_time], y0_full, t_eval=t)
                        y = sol.y[0]
                        e = 1.0 - y
                    
                    # Calculate Cost (ISE)
                    return np.sum(e**2)
                except Exception:
                    return 1e9 # Return high cost if simulation fails

            # --- RUN OPTIMIZER ---
            best_params = None
            best_score = 0
            
            if optimizer == "Custom PSO":
                best_params, best_score = custom_pso(objective, lb, ub, swarmsize=swarm_size, maxiter=max_iter)
            elif optimizer == "Differential Evolution":
                res = differential_evolution(objective, bounds=list(zip(lb, ub)))
                best_params = res.x; best_score = res.fun
            else:
                x0 = np.mean([lb, ub], axis=0)
                res = minimize(objective, x0, bounds=list(zip(lb, ub)), method="L-BFGS-B")
                best_params = res.x; best_score = res.fun

            st.success("Optimization Successfully Completed!")
            
            # --- VISUALIZE RESULTS ---
            t_final = np.linspace(0, sim_time, 500)
            
            # Re-run simulation for plotting (Simplified logic re-use)
            if system_mode.startswith("Linear"):
                n = [float(x) for x in num_str.split(",")]
                d = [float(x) for x in den_str.split(",")]
                plant = signal.TransferFunction(n, d)
                kp, ki = best_params[0], best_params[1]
                kd = best_params[2] if len(best_params) > 2 else 0
                c = signal.TransferFunction([kd, kp, ki], [1, 0])
                cl = signal.feedback(c * plant)
                _, y_final, _ = signal.lsim(cl, U=np.ones_like(t_final), T=t_final)
            else:
                x0 = [float(x) for x in nl_x0_str.split(",")]
                c0 = [float(x) for x in nl_c0_str.split(",")] if nl_c0_str.strip() else []
                scope = {"np": np, "params": best_params}
                exec(f"def plant(t, x, u):\n{nl_plant_code}", scope)
                exec(f"def ctrl(t, x, r, c, params):\n{nl_ctrl_code}", scope)
                def model(t, y_vec):
                    x = y_vec[:len(x0)]; c = y_vec[len(x0):]
                    u, c_dot = scope['ctrl'](t, x, 1.0, c, best_params)
                    x_dot = scope['plant'](t, x, u)
                    if not isinstance(c_dot, (np.ndarray, list)): c_dot = [c_dot]
                    return np.concatenate([x_dot, c_dot])
                sol = solve_ivp(model, [0, sim_time], np.concatenate([x0, c0]), t_eval=t_final)
                y_final = sol.y[0]

            # 1. Plot
            st.subheader("Step Response Analysis")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_final, np.ones_like(t_final), 'r--', label="Target Reference", linewidth=2)
            ax.plot(t_final, y_final, 'b-', label="Optimized Output", linewidth=2)
            ax.fill_between(t_final, y_final, alpha=0.1, color='blue')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
            # 2. Parameters Table
            st.subheader("Optimized Controller Gains")
            param_df = pd.DataFrame([best_params], columns=params)
            st.table(param_df)
            
            # 3. Performance Metrics
            perf = calculate_metrics(t_final, y_final, 1.0)
            m1, m2, m3 = st.columns(3)
            m1.metric("Rise Time", perf["Rise Time"])
            m2.metric("Overshoot", perf["Overshoot"])
            m3.metric("Settling Time", perf["Settling Time"])
            
            # 4. Download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                param_df.to_excel(writer, sheet_name='Parameters', index=False)
                pd.DataFrame({'Time': t_final, 'Output': y_final}).to_excel(writer, sheet_name='Data', index=False)
            
            st.download_button(
                label="📥 Download Report (.xlsx)",
                data=output.getvalue(),
                file_name="OptiCTRL_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Analysis Failed: {str(e)}")
            st.code(traceback.format_exc())