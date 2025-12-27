# ==========================================================
# EXE-SAFE OPTIMIZATION GUI BASE (Tkinter + SciPy + PSO)
# ==========================================================

import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from scipy.optimize import minimize, differential_evolution, Bounds
from scipy.integrate import solve_ivp

# ==========================================================
# GLOBAL CRASH PROTECTION (CRITICAL)
# ==========================================================

def global_exception_hook(exctype, value, tb):
    err = "".join(traceback.format_exception(exctype, value, tb))
    try:
        messagebox.showerror("Fatal Error", err)
    except Exception:
        print(err)

sys.excepthook = global_exception_hook

# ==========================================================
# OPTIMIZATION STRATEGIES (SAFE)
# ==========================================================

class PSOStrategy:
    def __init__(self, swarmsize=20, maxiter=50):
        self.swarmsize = swarmsize
        self.maxiter = maxiter

    def optimize(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        dim = len(lb)

        X = np.random.uniform(lb, ub, (self.swarmsize, dim))
        V = np.zeros_like(X)

        pbest = X.copy()
        pbest_val = np.array([func(x) for x in X])

        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx]

        w, c1, c2 = 0.7, 1.5, 1.5

        for _ in range(self.maxiter):
            for i in range(self.swarmsize):
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                V[i] = (
                    w * V[i]
                    + c1 * r1 * (pbest[i] - X[i])
                    + c2 * r2 * (gbest - X[i])
                )
                X[i] = np.clip(X[i] + V[i], lb, ub)

                val = func(X[i])
                if val < pbest_val[i]:
                    pbest[i] = X[i]
                    pbest_val[i] = val

            gbest_idx = np.argmin(pbest_val)
            gbest = pbest[gbest_idx]

        return gbest, func(gbest)


class DifferentialEvolutionStrategy:
    def optimize(self, func, bounds):
        result = differential_evolution(
            func,
            list(zip(bounds.lb, bounds.ub)),
            polish=True
        )
        return result.x, result.fun


class MinimizeStrategy:
    def __init__(self, method):
        self.method = method

    def optimize(self, func, bounds):
        x0 = (bounds.lb + bounds.ub) / 2
        result = minimize(func, x0, bounds=bounds, method=self.method)
        return result.x, result.fun


# ==========================================================
# MAIN APPLICATION
# ==========================================================

class OptimizationApp:

    def __init__(self, root):
        self.root = root
        self.root.title("EXE-SAFE Optimization Tool")
        self.root.geometry("500x300")

        ttk.Button(root, text="Run Optimization", command=self.run).pack(pady=20)
        self.status = ttk.Label(root, text="Idle")
        self.status.pack()

    # ----------------------------
    # SAFE THREAD ENTRY
    # ----------------------------
    def run(self):
        self.status.config(text="Running...")
        data = {
            "optimizer": "PSO"
        }
        threading.Thread(
            target=self._thread_wrapper,
            args=(data,),
            daemon=True
        ).start()

    def _thread_wrapper(self, data):
        try:
            result = self._safe_optimize(data)
            self.root.after(0, self._on_complete, result, None)
        except Exception:
            self.root.after(
                0,
                self._on_complete,
                None,
                traceback.format_exc()
            )

    # ----------------------------
    # SAFE OPTIMIZATION CORE
    # ----------------------------
    def _safe_optimize(self, data):

        bounds = Bounds([-5, -5], [5, 5])

        def objective(x):
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                return 1e9
            return np.sum(x**2)

        strategy = PSOStrategy()
        xopt, fopt = strategy.optimize(objective, bounds)

        if np.any(np.isnan(xopt)):
            raise RuntimeError("Invalid optimizer output")

        return xopt, fopt

    # ----------------------------
    # GUI CALLBACK
    # ----------------------------
    def _on_complete(self, result, error):
        if error:
            messagebox.showerror("Optimization Error", error)
            self.status.config(text="Failed")
        else:
            xopt, fopt = result
            messagebox.showinfo(
                "Success",
                f"Optimal x: {xopt}\nCost: {fopt:.4f}"
            )
            self.status.config(text="Done")


# ==========================================================
# EXE-SAFE ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = OptimizationApp(root)
        root.mainloop()
    except Exception:
        messagebox.showerror("Startup Error", traceback.format_exc())
import tkinter as tk
from tkinter import messagebox, font, filedialog
import numpy as np
from scipy import signal
from scipy.optimize import minimize, differential_evolution, Bounds
from scipy.integrate import solve_ivp 
from pyswarm import pso
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import re
import xlwt 
import traceback 
import threading 

# ------------------- CONTROLLER TEMPLATES -------------------

CONTROLLER_TEMPLATES = {
    "SOSMC (Super-Twisting)": {
        "params": ["k1", "k2"],
        "c0": "0", 
        "code": """    # Super-Twisting (SOSMC)
    # params[0]=k1, params[1]=k2
    # Plant output y = x[0]

    k1, k2 = params[0], params[1]
    
    e = r - x[0]  # Error
    sigma = c[0]  # Controller state
    
    # Super-twisting logic
    v1 = -k1 * np.sqrt(np.abs(e)) * np.sign(e)
    v = v1 + sigma
    sigma_dot = -k2 * np.sign(e)
    
    u = v 

    # Return control u, and controller state derivatives
    return u, np.array([sigma_dot])
"""
    },
    
    "TSMC (Terminal SMC)": {
        "params": ["c_tsmc", "beta"],
        "c0": "", 
        "code": """    # Terminal Sliding Mode (TSMC)
    # params[0]=c, params[1]=beta (0 < beta < 1)
    
    c, beta = params[0], params[1]

    e1 = r - x[0]
    e2 = 0 - x[1] # 0 is r_dot
    
    # Non-linear sliding surface
    s = e2 + c * (e1 ** beta)
    
    # Smooth reaching law
    k_reach = 5.0 
    u = k_reach * np.tanh(s * 10) 
    
    return u, np.array([])
"""
    },
    
    "SMC (Standard Chattering)": {
        "params": ["K_smc", "lambda_smc"],
        "c0": "", 
        "code": """    # Standard SMC
    # params[0]=K, params[1]=lambda
    
    K, lam = params[0], params[1]

    e = r - x[0]
    e_dot = 0 - x[1]
    
    s = e_dot + lam * e
    u = K * np.sign(s)
    
    return u, np.array([])
"""
    }
}


# ------------------- OPTIMIZATION STRATEGIES -------------------
class PSOStrategy:
    def __init__(self, swarmsize, maxiter):
        self.swarmsize = swarmsize
        self.maxiter = maxiter
    def optimize(self, func, bounds):
        lb = bounds.lb; ub = bounds.ub
        return pso(func, lb, ub, swarmsize=self.swarmsize, maxiter=self.maxiter)

class DifferentialEvolutionStrategy:
    def optimize(self, func, bounds):
        result = differential_evolution(func, bounds, maxiter=100, popsize=15, tol=0.01)
        return result.x, result.fun

class MinimizeStrategy:
    def __init__(self, method): self.method = method
    def optimize(self, func, bounds):
        x0 = np.random.uniform(bounds.lb, bounds.ub)
        result = minimize(func, x0, method=self.method, bounds=bounds)
        return result.x, result.fun

# ------------------- MAIN APPLICATION -------------------

class PIDTunerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OptiCTRL - Controller Tuning Studio")
        self.root.geometry("1150x850") 

        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Arial", size=10)
        self.root.option_add("*Font", default_font)

        self.optimized_params = None
        self.plant_tf = None 
        self.last_plot_data = None 
        
        self.nl_plant_model = None
        self.nl_controller_model = None
        self.nl_param_count = 0
        self.nl_param_names = ["P1", "P2", "P3", "P4"] 

        # --- FIX: Define optimizers dictionary BEFORE building GUI ---
        # The GUI needs the keys of this dictionary to populate the dropdown
        self.optimizers = {
            "PSO": None, 
            "Differential Evolution": None,
            "L-BFGS-B (Local)": None,
            "SLSQP (Local)": None,
        }

        self.plant_model_var = tk.StringVar(value="Transfer Function")
        self.system_type_var = tk.StringVar(value="Linear") 
        
        self.rise_time_var = tk.StringVar(value="N/A")
        self.settling_time_var = tk.StringVar(value="N/A")
        self.overshoot_var = tk.StringVar(value="N/A")
        self.peak_time_var = tk.StringVar(value="N/A")

        self._build_gui()
        self._on_system_type_change() 
        self._update_results_labels() 
        self._update_plant_input_frame() 

    # ---------------- GUI Layout ----------------

    def _build_gui(self):
        left_container = tk.Frame(self.root, width=410) 
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_container.pack_propagate(False) 

        canvas = tk.Canvas(left_container, bg="#f0f0f0")
        v_scrollbar = tk.Scrollbar(left_container, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = tk.Scrollbar(left_container, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(canvas, padx=10, pady=10, bg="#f0f0f0")
        canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.control_frame.bind("<Configure>", on_configure)

        def on_mouse_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mouse_wheel)

        self.plot_frame = tk.Frame(self.root, padx=10, pady=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._create_control_widgets()

        self.fig = plt.Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.ax.set_title("Step Response")
        self.ax.grid(True)
        self.canvas.draw()

    def _create_control_widgets(self):
        row_idx = 0

        # 1. System Type
        sys_type_frame = tk.LabelFrame(self.control_frame, text="1. System Type", padx=10, pady=10, bg="#f0f0f0")
        sys_type_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        tk.Label(sys_type_frame, text="Select:", bg="#f0f0f0").grid(row=0, column=0)
        sys_type_menu = tk.OptionMenu(sys_type_frame, self.system_type_var, "Linear", "Non-Linear", command=self._on_system_type_change)
        sys_type_menu.config(bg="white", width=20)
        sys_type_menu.grid(row=0, column=1)

        # 2a. Linear Plant
        self.l_plant_frame = tk.LabelFrame(self.control_frame, text="2. Linear Plant Model", padx=10, pady=10, bg="#f0f0f0")
        self.l_plant_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        tk.Label(self.l_plant_frame, text="Model Type:", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
        model_options = ["Transfer Function", "State Space", "ODE"]
        model_menu = tk.OptionMenu(self.l_plant_frame, self.plant_model_var, *model_options, command=self._on_plant_model_change)
        model_menu.config(bg="white", width=20)
        model_menu.grid(row=0, column=1, sticky="ew")
        self.plant_input_frame = tk.Frame(self.l_plant_frame, bg="#f0f0f0")
        self.plant_input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        # 2b. Non-Linear Plant
        self.nl_plant_frame = tk.LabelFrame(self.control_frame, text="2. Non-Linear Plant Model", padx=10, pady=10, bg="#f0f0f0")
        self.nl_plant_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        tk.Label(self.nl_plant_frame, text="Initial States (x0):", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
        self.nl_x0_entry = tk.Entry(self.nl_plant_frame, width=20); self.nl_x0_entry.insert(0, "0, 0")
        self.nl_x0_entry.grid(row=0, column=1, sticky="w")
        tk.Label(self.nl_plant_frame, text="def plant_model(t, x, u): -> np.array", bg="#f0f0f0", font=("Arial", 9, "italic")).grid(row=1, column=0, columnspan=2, sticky="w")
        self.nl_plant_text = tk.Text(self.nl_plant_frame, height=8, width=40, font=("Courier New", 9))
        self.nl_plant_text.insert("1.0", "    # Example: 2nd-order system (y=x[0])\n    x1, x2 = x\n    x1_dot = x2\n    x2_dot = -0.5 * x1 - 0.1 * x2**3 + u\n    return np.array([x1_dot, x2_dot])\n")
        self.nl_plant_text.grid(row=2, column=0, columnspan=2)
        row_idx += 1

        # 3a. Linear Controller
        self.l_ctrl_frame = tk.LabelFrame(self.control_frame, text="3. Linear Controller Type", padx=10, pady=10, bg="#f0f0f0")
        self.l_ctrl_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        tk.Label(self.l_ctrl_frame, text="Select Type:", bg="#f0f0f0").grid(row=0, column=0)
        self.controller_var = tk.StringVar(value="PID")
        options = ["P", "PI", "PD", "PID", "PID (Filtered-D)", "SMC (PD-Type)", "SMC (PI-Type)", "SMC (PID-Type)"]
        ctrl_menu = tk.OptionMenu(self.l_ctrl_frame, self.controller_var, *options, command=self._on_controller_change)
        ctrl_menu.config(bg="white", width=20) 
        ctrl_menu.grid(row=0, column=1)

        # 3b. Non-Linear Controller
        self.nl_ctrl_frame = tk.LabelFrame(self.control_frame, text="3. Non-Linear Controller", padx=10, pady=10, bg="#f0f0f0")
        self.nl_ctrl_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        
        template_frame = tk.Frame(self.nl_ctrl_frame, bg="#f0f0f0")
        template_frame.grid(row=0, column=0, columnspan=2)
        tk.Label(template_frame, text="Load Template:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        tk.Button(template_frame, text="SOSMC", command=lambda: self._load_nl_controller_template("SOSMC (Super-Twisting)")).pack(side=tk.LEFT, padx=2)
        tk.Button(template_frame, text="TSMC", command=lambda: self._load_nl_controller_template("TSMC (Terminal SMC)")).pack(side=tk.LEFT, padx=2)
        tk.Button(template_frame, text="SMC", command=lambda: self._load_nl_controller_template("SMC (Standard Chattering)")).pack(side=tk.LEFT, padx=2)
        
        tk.Label(self.nl_ctrl_frame, text="Initial States (c0):", bg="#f0f0f0").grid(row=1, column=0, sticky="w", pady=(5,0))
        self.nl_c0_entry = tk.Entry(self.nl_ctrl_frame, width=20); self.nl_c0_entry.insert(0, "0")
        self.nl_c0_entry.grid(row=1, column=1, sticky="w", pady=(5,0))
        
        tk.Label(self.nl_ctrl_frame, text="def controller_model(t, x, r, c, params):", bg="#f0f0f0", font=("Arial", 9, "italic")).grid(row=2, column=0, columnspan=2, sticky="w")
        self.nl_ctrl_text = tk.Text(self.nl_ctrl_frame, height=15, width=40, font=("Courier New", 9))
        self.nl_ctrl_text.grid(row=3, column=0, columnspan=2)
        row_idx += 1
        
        # 4. Input Signal
        input_frame = tk.LabelFrame(self.control_frame, text="4. Input Signal", padx=10, pady=10, bg="#f0f0f0")
        input_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        self.input_type_var = tk.StringVar(value="Step")
        tk.Label(input_frame, text="Type:", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
        tk.OptionMenu(input_frame, self.input_type_var, "Step", "Ramp", "Parabolic").grid(row=0, column=1, sticky="w")
        tk.Label(input_frame, text="Sim. Time (s):", bg="#f0f0f0").grid(row=1, column=0, sticky="w")
        self.sim_time_entry = tk.Entry(input_frame, width=10); self.sim_time_entry.insert(0, "20")
        self.sim_time_entry.grid(row=1, column=1, sticky="w")

        # 5. Objective Function
        obj_frame = tk.LabelFrame(self.control_frame, text="5. Objective Function", padx=10, pady=10, bg="#f0f0f0")
        obj_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        tk.Label(obj_frame, text="Select:", bg="#f0f0f0").grid(row=0, column=0)
        self.obj_type_var = tk.StringVar(value="ISE")
        obj_options = ["ISE", "IAE", "ITAE", "ITSE", "IATE", "ISE+IAE", "ISE+ITAE", "IAE+ITAE", "ISE+IAE+ITAE", "ALL"]
        tk.OptionMenu(obj_frame, self.obj_type_var, *obj_options).grid(row=0, column=1)

        # 6. Bounds
        self.bounds_frame_container = tk.Frame(self.control_frame, bg="#f0f0f0")
        self.bounds_frame_container.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        
        self.l_bounds_frame = tk.LabelFrame(self.bounds_frame_container, text="6. Controller Parameter Bounds", padx=10, pady=10, bg="#f0f0f0")
        self.l_bounds_frame.pack(fill="x")
        self._update_bounds_frame() 
        
        self.nl_bounds_frame = tk.LabelFrame(self.bounds_frame_container, text="6. Controller Parameter Bounds", padx=10, pady=10, bg="#f0f0f0")
        self.nl_bounds_frame.pack(fill="x")
        self.nl_bounds_entries = {}
        tk.Label(self.nl_bounds_frame, text="Param", bg="#f0f0f0").grid(row=0, column=0)
        tk.Label(self.nl_bounds_frame, text="Min", bg="#f0f0f0").grid(row=0, column=1)
        tk.Label(self.nl_bounds_frame, text="Max", bg="#f0f0f0").grid(row=0, column=2)
        self.nl_bounds_labels = [] 
        for i, p in enumerate(["P1", "P2", "P3", "P4"]):
            label = tk.Label(self.nl_bounds_frame, text=f"{p}:", bg="#f0f0f0")
            label.grid(row=i+1, column=0)
            self.nl_bounds_labels.append(label)
            e1 = tk.Entry(self.nl_bounds_frame, width=8); e1.insert(0, "0.1")
            e2 = tk.Entry(self.nl_bounds_frame, width=8); e2.insert(0, "20.0")
            e1.grid(row=i+1, column=1); e2.grid(row=i+1, column=2)
            self.nl_bounds_entries[f"P{i+1}"] = (e1, e2) 

        # 7. Optimizer
        opt_frame = tk.LabelFrame(self.control_frame, text="7. Optimization Method", padx=10, pady=10, bg="#f0f0f0")
        opt_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        self.optimizer_var = tk.StringVar(value="PSO")
        tk.Label(opt_frame, text="Algorithm:", bg="#f0f0f0").grid(row=0, column=0)
        
        # This checks self.optimizers to create the dropdown
        tk.OptionMenu(opt_frame, self.optimizer_var, *self.optimizers.keys(), command=self._on_optimizer_change).grid(row=0, column=1)

        # 8. PSO Params
        self.pso_frame = tk.LabelFrame(self.control_frame, text="8. PSO Parameters", padx=10, pady=10, bg="#f0f0f0")
        self.pso_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        tk.Label(self.pso_frame, text="Swarm Size:", bg="#f0f0f0").grid(row=0, column=0)
        self.swarm_entry = tk.Entry(self.pso_frame, width=8); self.swarm_entry.insert(0, "20")
        self.swarm_entry.grid(row=0, column=1)
        tk.Label(self.pso_frame, text="Max Iterations:", bg="#f0f0f0").grid(row=1, column=0)
        self.iter_entry = tk.Entry(self.pso_frame, width=8); self.iter_entry.insert(0, "50")
        self.iter_entry.grid(row=1, column=1)

        # 9. Run Button
        self.run_button = tk.Button(self.control_frame, text="9. Run Optimization", command=self.run_optimization,
                                    bg="#007bff", fg="white", font=("Arial", 10, "bold"), height=2)
        self.run_button.grid(row=row_idx, column=0, sticky="ew", pady=10)
        row_idx += 1

        # 10. Results
        results_frame = tk.LabelFrame(self.control_frame, text="10. Optimization Results", padx=10, pady=10, bg="#f0f0f0")
        results_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        self.kp_val_var, self.ki_val_var, self.kd_val_var, self.p4_val_var, self.metric_var = (tk.StringVar(value="N/A") for _ in range(5))
        
        labels = [("P1/Kp:", self.kp_val_var), ("P2/Ki:", self.ki_val_var), ("P3/Kd:", self.kd_val_var), ("P4/Tf:", self.p4_val_var)]
        self.res_labels = []
        self.res_vals = []
        for i, (txt, var) in enumerate(labels):
            lbl = tk.Label(results_frame, text=txt, bg="#f0f0f0", font=("Arial", 10, "bold"))
            lbl.grid(row=i, column=0, sticky="w")
            val = tk.Label(results_frame, textvariable=var, bg="#f0f0f0")
            val.grid(row=i, column=1, sticky="w")
            self.res_labels.append(lbl)
            self.res_vals.append(val)
        
        self.kp_label, self.ki_label, self.kd_label, self.p4_label = self.res_labels
        self.ki_val_widget = self.res_vals[1]
        self.kd_val_widget = self.res_vals[2]
        self.p4_val_widget = self.res_vals[3]

        tk.Label(results_frame, text="Metric:", bg="#f0f0f0", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w")
        tk.Label(results_frame, textvariable=self.metric_var, bg="#f0f0f0").grid(row=4, column=1, sticky="w")

        # 11. Metrics
        metrics_frame = tk.LabelFrame(self.control_frame, text="11. Response Metrics", padx=10, pady=10, bg="#f0f0f0")
        metrics_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        m_labels = [("Rise Time:", self.rise_time_var), ("Settling Time:", self.settling_time_var), 
                    ("Overshoot:", self.overshoot_var), ("Peak Time:", self.peak_time_var)]
        for i, (txt, var) in enumerate(m_labels):
            tk.Label(metrics_frame, text=txt, bg="#f0f0f0", font=("Arial", 10, "bold")).grid(row=i, column=0, sticky="w")
            tk.Label(metrics_frame, textvariable=var, bg="#f0f0f0").grid(row=i, column=1, sticky="w")

        # 12. Final Params
        final_params_frame = tk.LabelFrame(self.control_frame, text="12. Final Optimized Parameters", padx=10, pady=10, bg="#f0f0f0")
        final_params_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        self.final_params_text = tk.Text(final_params_frame, height=5, width=35, state=tk.DISABLED, bg="#f9f9f9", font=("Arial", 9))
        self.final_params_text.pack(fill=tk.X, expand=True)
        
        # 13. Export
        export_frame = tk.LabelFrame(self.control_frame, text="13. Export Data", padx=10, pady=10, bg="#f0f0f0")
        export_frame.grid(row=row_idx, column=0, sticky="ew", pady=5)
        row_idx += 1
        self.export_button = tk.Button(export_frame, text="Save to Excel (.xls)", command=self.export_to_excel,
                                       bg="#1e9c5d", fg="white", font=("Arial", 10, "bold"))
        self.export_button.pack(fill=tk.X, expand=True)

        self._load_nl_controller_template("SOSMC (Super-Twisting)")

    # ---------------- Dynamic UI ----------------
    
    def _on_system_type_change(self, *args):
        mode = self.system_type_var.get()
        if mode == "Linear":
            self.nl_plant_frame.grid_remove()
            self.nl_ctrl_frame.grid_remove()
            self.nl_bounds_frame.pack_forget()
            self.l_plant_frame.grid()
            self.l_ctrl_frame.grid()
            self.l_bounds_frame.pack(fill="x")
        else: 
            self.l_plant_frame.grid_remove()
            self.l_ctrl_frame.grid_remove()
            self.l_bounds_frame.pack_forget()
            self.nl_plant_frame.grid()
            self.nl_ctrl_frame.grid()
            self.nl_bounds_frame.pack(fill="x")
            if not self.nl_param_names: self.nl_param_names = ["P1", "P2", "P3", "P4"]
        
        self._update_results_labels()
        self._clear_metrics() 
        self.last_plot_data = None

    def _on_plant_model_change(self, *args): self._update_plant_input_frame()

    def _update_plant_input_frame(self):
        for widget in self.plant_input_frame.winfo_children(): widget.destroy()
        model_type = self.plant_model_var.get()
        if model_type == "Transfer Function":
            tk.Label(self.plant_input_frame, text="Numerator:", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
            self.tf_num_entry = tk.Entry(self.plant_input_frame, width=25); self.tf_num_entry.insert(0, "1")
            self.tf_num_entry.grid(row=0, column=1, sticky="w")
            tk.Label(self.plant_input_frame, text="Denominator:", bg="#f0f0f0").grid(row=1, column=0, sticky="w")
            self.tf_den_entry = tk.Entry(self.plant_input_frame, width=25); self.tf_den_entry.insert(0, "1, 2, 1")
            self.tf_den_entry.grid(row=1, column=1, sticky="w")
        elif model_type == "ODE":
            tk.Label(self.plant_input_frame, text="Input (u) Coeffs (b_m...b_0):", bg="#f0f0f0").grid(row=0, column=0, sticky="w")
            self.ode_num_entry = tk.Entry(self.plant_input_frame, width=25); self.ode_num_entry.insert(0, "1")
            self.ode_num_entry.grid(row=0, column=1, sticky="w")
            tk.Label(self.plant_input_frame, text="Output (y) Coeffs (a_n...a_0):", bg="#f0f0f0").grid(row=1, column=0, sticky="w")
            self.ode_den_entry = tk.Entry(self.plant_input_frame, width=25); self.ode_den_entry.insert(0, "1, 2, 1")
            self.ode_den_entry.grid(row=1, column=1, sticky="w")
        elif model_type == "State Space":
            tk.Label(self.plant_input_frame, text="Use ';' for rows, ',' for cols", bg="#f0f0f0", font=("Arial", 8, "italic")).grid(row=0, column=0, columnspan=2, sticky="w")
            tk.Label(self.plant_input_frame, text="A:", bg="#f0f0f0").grid(row=1, column=0, sticky="w")
            self.ss_A_entry = tk.Entry(self.plant_input_frame, width=25); self.ss_A_entry.insert(0, "0, 1; -1, -2") 
            self.ss_A_entry.grid(row=1, column=1, sticky="w")
            tk.Label(self.plant_input_frame, text="B:", bg="#f0f0f0").grid(row=2, column=0, sticky="w")
            self.ss_B_entry = tk.Entry(self.plant_input_frame, width=25); self.ss_B_entry.insert(0, "0; 1")
            self.ss_B_entry.grid(row=2, column=1, sticky="w")
            tk.Label(self.plant_input_frame, text="C:", bg="#f0f0f0").grid(row=3, column=0, sticky="w")
            self.ss_C_entry = tk.Entry(self.plant_input_frame, width=25); self.ss_C_entry.insert(0, "1, 0")
            self.ss_C_entry.grid(row=3, column=1, sticky="w")
            tk.Label(self.plant_input_frame, text="D:", bg="#f0f0f0").grid(row=4, column=0, sticky="w")
            self.ss_D_entry = tk.Entry(self.plant_input_frame, width=25); self.ss_D_entry.insert(0, "0")
            self.ss_D_entry.grid(row=4, column=1, sticky="w")
    
    def _on_controller_change(self, *args):
        self._update_bounds_frame()
        self._update_results_labels() 
        self._clear_metrics()

    def _update_bounds_frame(self):
        for widget in self.l_bounds_frame.winfo_children(): widget.destroy()
        ctrl_type = self.controller_var.get()
        
        if ctrl_type == "PID": params = ["Kp", "Ki", "Kd"]
        elif ctrl_type == "PI": params = ["Kp", "Ki"]
        elif ctrl_type == "PD": params = ["Kp", "Kd"]
        elif ctrl_type == "P": params = ["Kp"]
        elif ctrl_type == "PID (Filtered-D)": params = ["Kp", "Ki", "Kd", "Tf"]
        elif ctrl_type == "SMC (PD-Type)": params = ["Kp", "Lambda"]
        elif ctrl_type == "SMC (PI-Type)": params = ["Kp", "Lambda"]
        elif ctrl_type == "SMC (PID-Type)": params = ["Kp", "Lambda", "Gamma"]
        else: params = ["Kp"]

        tk.Label(self.l_bounds_frame, text="Parameter").grid(row=0, column=0)
        tk.Label(self.l_bounds_frame, text="Min").grid(row=0, column=1)
        tk.Label(self.l_bounds_frame, text="Max").grid(row=0, column=2)
        self.bounds_entries = {}
        for i, p in enumerate(params):
            tk.Label(self.l_bounds_frame, text=p).grid(row=i+1, column=0)
            e1 = tk.Entry(self.l_bounds_frame, width=8); e2 = tk.Entry(self.l_bounds_frame, width=8)
            if p == "Tf": e1.insert(0, "0.01"); e2.insert(0, "1")
            else: e1.insert(0, "0.1"); e2.insert(0, "20")
            e1.grid(row=i+1, column=1); e2.grid(row=i+1, column=2)
            self.bounds_entries[p] = (e1, e2)

    def _update_results_labels(self):
        self.last_plot_data = None 
        mode = self.system_type_var.get()
        
        self.ki_label.grid_remove(); self.ki_val_widget.grid_remove()
        self.kd_label.grid_remove(); self.kd_val_widget.grid_remove()
        self.p4_label.grid_remove(); self.p4_val_widget.grid_remove()
        
        self.kp_val_var.set("N/A"); self.ki_val_var.set("N/A")
        self.kd_val_var.set("N/A"); self.p4_val_var.set("N/A"); self.metric_var.set("N/A")
        
        if hasattr(self, 'final_params_text'):
            self.final_params_text.config(state=tk.NORMAL)
            self.final_params_text.delete("1.0", tk.END)
            self.final_params_text.insert(tk.END, "Run optimization to see values.")
            self.final_params_text.config(state=tk.DISABLED)

        if mode == "Linear":
            ctrl_type = self.controller_var.get()
            if ctrl_type == "P":
                self.kp_label.config(text="Kp:")
            elif ctrl_type == "PI":
                self.kp_label.config(text="Kp:"); self.ki_label.config(text="Ki:")
                self.ki_label.grid(); self.ki_val_widget.grid()
            elif ctrl_type == "PD":
                self.kp_label.config(text="Kp:"); self.ki_label.config(text="Kd:") 
                self.ki_label.grid(); self.ki_val_widget.grid()
            elif ctrl_type == "PID":
                self.kp_label.config(text="Kp:"); self.ki_label.config(text="Ki:"); self.kd_label.config(text="Kd:")
                self.ki_label.grid(); self.ki_val_widget.grid(); self.kd_label.grid(); self.kd_val_widget.grid()
            elif ctrl_type == "PID (Filtered-D)":
                self.kp_label.config(text="Kp:"); self.ki_label.config(text="Ki:"); self.kd_label.config(text="Kd:"); self.p4_label.config(text="Tf:")
                self.ki_label.grid(); self.ki_val_widget.grid(); self.kd_label.grid(); self.kd_val_widget.grid(); self.p4_label.grid(); self.p4_val_widget.grid()
            elif ctrl_type == "SMC (PD-Type)":
                self.kp_label.config(text="Kp:"); self.ki_label.config(text="Lambda:")
                self.ki_label.grid(); self.ki_val_widget.grid()
            elif ctrl_type == "SMC (PI-Type)":
                self.kp_label.config(text="Kp:"); self.ki_label.config(text="Lambda:")
                self.ki_label.grid(); self.ki_val_widget.grid()
            elif ctrl_type == "SMC (PID-Type)":
                self.kp_label.config(text="Kp:"); self.ki_label.config(text="Lambda:"); self.kd_label.config(text="Gamma:")
                self.ki_label.grid(); self.ki_val_widget.grid(); self.kd_label.grid(); self.kd_val_widget.grid()
        
        else: 
            p_names = self.nl_param_names + ["P3", "P4"] 
            self.kp_label.config(text=f"{p_names[0]}:")
            self.ki_label.config(text=f"{p_names[1]}:")
            self.kd_label.config(text=f"{p_names[2]}:")
            self.p4_label.config(text=f"{p_names[3]}:")
            
            if len(self.nl_param_names) >= 1: self.ki_label.grid(); self.ki_val_widget.grid()
            if len(self.nl_param_names) >= 2: self.kd_label.grid(); self.kd_val_widget.grid()
            if len(self.nl_param_names) >= 3: self.p4_label.grid(); self.p4_val_widget.grid()


    def _on_optimizer_change(self, *args):
        if self.optimizer_var.get() == "PSO":
            self.pso_frame.grid()
        else:
            self.pso_frame.grid_remove()

    def _load_nl_controller_template(self, template_name):
        if template_name not in CONTROLLER_TEMPLATES:
            messagebox.showerror("Error", f"Template '{template_name}' not found.")
            return
            
        template = CONTROLLER_TEMPLATES[template_name]
        
        self.nl_ctrl_text.delete("1.0", tk.END)
        self.nl_ctrl_text.insert("1.0", template["code"])
        
        self.nl_c0_entry.delete(0, tk.END)
        self.nl_c0_entry.insert(0, template["c0"])
        
        self.nl_param_names = template["params"]
        
        all_labels = ["P1", "P2", "P3", "P4"]
        for i, label_widget in enumerate(self.nl_bounds_labels):
            if i < len(self.nl_param_names):
                label_widget.config(text=f"{self.nl_param_names[i]}:")
            else:
                label_widget.config(text=f"{all_labels[i]}:") 
        
        self._update_results_labels()
        self._clear_metrics()

    # ---------------- Core Logic ----------------

    def _parse_vector_entry(self, entry_widget, dtype=float):
        try:
            text = entry_widget.get()
            if not text.strip(): return [] 
            return [dtype(x.strip()) for x in text.split(",")]
        except Exception as e:
            messagebox.showerror("Parsing Error", f"Invalid vector format: {text}\nUse comma-separated numbers.\n{e}")
            return None

    def _parse_matrix_entry(self, entry_widget):
        try:
            text = entry_widget.get().strip(); text = re.sub(r'[\[\]]', '', text)
            rows = text.split(';')
            matrix = [[float(c.strip()) for c in r.strip().split(',')] for r in rows]
            if len(matrix) > 1:
                it = iter(matrix); the_len = len(next(it))
                if not all(len(l) == the_len for l in it): raise ValueError("Matrix rows have inconsistent lengths.")
            return np.array(matrix)
        except Exception as e:
            messagebox.showerror("Parsing Error", f"Invalid matrix format: {text}\nUse ';' for rows, ',' for columns.\n{e}")
            return None

    def _get_plant_tf(self):
        model_type = self.plant_model_var.get()
        try:
            if model_type == "Transfer Function":
                num = self._parse_vector_entry(self.tf_num_entry); den = self._parse_vector_entry(self.tf_den_entry)
                if num is None or den is None: return None
                return signal.TransferFunction(num, den)
            elif model_type == "ODE":
                num = self._parse_vector_entry(self.ode_num_entry); den = self._parse_vector_entry(self.ode_den_entry)
                if num is None or den is None: return None
                return signal.TransferFunction(num, den)
            elif model_type == "State Space":
                A = self._parse_matrix_entry(self.ss_A_entry); B = self._parse_matrix_entry(self.ss_B_entry)
                C = self._parse_matrix_entry(self.ss_C_entry); D = self._parse_matrix_entry(self.ss_D_entry)
                if A is None or B is None or C is None or D is None: return None
                if A.shape[0] != A.shape[1]: raise ValueError("Matrix A must be square.")
                if A.shape[0] != B.shape[0]: raise ValueError("A and B must have same number of rows.")
                if A.shape[1] != C.shape[1]: raise ValueError("A and C must have same number of columns.")
                if B.shape[1] != D.shape[1]: raise ValueError("B and D must have same number of columns.")
                if C.shape[0] != D.shape[0]: raise ValueError("C and D must have same number of rows.")
                num, den = signal.ss2tf(A, B, C, D)
                if num.ndim == 2 and num.shape[0] == 1: num = num.flatten()
                return signal.TransferFunction(num, den)
        except Exception as e:
            messagebox.showerror("Model Error", f"Error creating plant model: {e}"); return None
        return None 

    def _get_sim_time(self):
        try:
            time = float(self.sim_time_entry.get())
            if time <= 0: messagebox.showerror("Input Error", "Simulation Time must be greater than 0."); return None
            return time
        except ValueError: messagebox.showerror("Input Error", "Invalid Simulation Time. Please enter a number."); return None

    def _generate_input(self, t):
        sig = self.input_type_var.get()
        if isinstance(t, np.ndarray):
            if sig == "Step": return np.ones_like(t)
            elif sig == "Ramp": return t
            else: return 0.5 * t ** 2
        else: 
            if sig == "Step": return 1.0
            elif sig == "Ramp": return t
            else: return 0.5 * t ** 2
            
    def _get_controller_tf(self, params, ctrl_type):
        if ctrl_type == "P": return signal.TransferFunction([params[0]], [1])
        elif ctrl_type == "PI": return signal.TransferFunction([params[0], params[1]], [1, 0])
        elif ctrl_type == "PD": return signal.TransferFunction([params[1], params[0]], [1])
        elif ctrl_type == "PID": return signal.TransferFunction([params[2], params[0], params[1]], [1, 0])
        elif ctrl_type == "PID (Filtered-D)":
            Kp, Ki, Kd, Tf = params
            ctrl_num = [(Kp * Tf + Kd), (Kp + Ki * Tf), Ki]; ctrl_den = [Tf, 1, 0]
            return signal.TransferFunction(ctrl_num, ctrl_den)
        elif ctrl_type == "SMC (PD-Type)": return signal.TransferFunction([params[1], params[0]], [1])
        elif ctrl_type == "SMC (PI-Type)": return signal.TransferFunction([params[0], params[1]], [1, 0])
        elif ctrl_type == "SMC (PID-Type)": return signal.TransferFunction([params[2], params[0], params[1]], [1, 0])
        else: return signal.TransferFunction([params[0]], [1])

    # --- Worker Thread ---
    def _run_optimization_thread(self, data):
        try:
            # Reconstruct arguments in the thread
            mode = data["mode"]
            sim_time = data["sim_time"]
            optimizer_name = data["optimizer_name"]
            
            # Create strategy based on passed params (swarmsize, maxiter are in data)
            if optimizer_name == "PSO":
                strategy = PSOStrategy(swarmsize=data["pso_swarmsize"], maxiter=data["pso_maxiter"])
            elif optimizer_name == "Differential Evolution":
                strategy = DifferentialEvolutionStrategy(self)
            else:
                strategy = MinimizeStrategy(self, optimizer_name.split(" ")[0])

            objective_func = None
            bounds = None

            if mode == "Linear":
                # Linear Logic (Plant TF is passed in data or constructed here?)
                # Constructing TF inside thread is safer if data is raw numbers
                # But for simplicity, we assume self.plant_tf is accessible or passed. 
                # Tkinter objects are NOT accessible. self.plant_tf is just a scipy object, so it's OK.
                
                # Check for plant
                if self.plant_tf is None:
                    raise ValueError("Plant model is not defined.")

                params = data["l_params"]
                lb = data["l_lb"]
                ub = data["l_ub"]
                
                # Bounds check
                for i, p in enumerate(params):
                    if p == "Tf" and lb[i] <= 0: lb[i] = 1e-4
                    elif lb[i] == 0: lb[i] = 1e-6
                
                bounds = Bounds(lb, ub)
                
                # Define Objective Wrapper
                def objective(p):
                    return self._objective_linear_calc(p, data["obj_type"], data["l_ctrl_type"], 
                                                     self.plant_tf, sim_time, data["input_type"])
                objective_func = objective

            else: # Non-Linear
                if not self.nl_plant_model or not self.nl_controller_model:
                     raise ValueError("Non-linear models not compiled.")
                
                lb = data["nl_lb"]
                ub = data["nl_ub"]
                bounds = Bounds(lb, ub)
                
                # Define Objective Wrapper
                def objective(p):
                    return self._objective_nonlinear_calc(p, data["obj_type"], 
                                                        data["nl_x0"], data["nl_c0"], 
                                                        sim_time, data["input_type"])
                objective_func = objective

            # Run Optimization
            xopt, fopt = strategy.optimize(objective_func, bounds)
            
            # Post back to main thread
            self.root.after(0, self._on_optimization_complete, xopt, fopt, None)

        except Exception as e:
            self.root.after(0, self._on_optimization_complete, None, None, str(e))

    # --- Calculation Logic (Separated from GUI) ---
    def _objective_linear_calc(self, params, obj_type, ctrl_type, plant_tf, sim_time, input_type):
        try:
            num, den = plant_tf.num, plant_tf.den
            
            # Reconstruct controller TF (Manual copy of logic to avoid self references)
            if ctrl_type == "P": pid_tf = signal.TransferFunction([params[0]], [1])
            elif ctrl_type == "PI": pid_tf = signal.TransferFunction([params[0], params[1]], [1, 0])
            elif ctrl_type == "PD": pid_tf = signal.TransferFunction([params[1], params[0]], [1])
            elif ctrl_type == "PID": pid_tf = signal.TransferFunction([params[2], params[0], params[1]], [1, 0])
            elif ctrl_type == "PID (Filtered-D)":
                Kp, Ki, Kd, Tf = params
                ctrl_num = [(Kp * Tf + Kd), (Kp + Ki * Tf), Ki]; ctrl_den = [Tf, 1, 0]
                pid_tf = signal.TransferFunction(ctrl_num, ctrl_den)
            elif ctrl_type == "SMC (PD-Type)": pid_tf = signal.TransferFunction([params[1], params[0]], [1])
            elif ctrl_type == "SMC (PI-Type)": pid_tf = signal.TransferFunction([params[0], params[1]], [1, 0])
            elif ctrl_type == "SMC (PID-Type)": pid_tf = signal.TransferFunction([params[2], params[0], params[1]], [1, 0])
            else: pid_tf = signal.TransferFunction([params[0]], [1])

            loop_num = signal.convolve(pid_tf.num, num); loop_den = signal.convolve(pid_tf.den, den)
            if np.all(np.polyadd(loop_den, loop_num) == 0): return 1e6
            cl_tf = signal.TransferFunction(loop_num, np.polyadd(loop_den, loop_num))

            t = np.linspace(0, sim_time, 500)
            
            # Generate Input
            if input_type == "Step": r = np.ones_like(t)
            elif input_type == "Ramp": r = t
            else: r = 0.5 * t ** 2
            
            t_out, y, _ = signal.lsim(cl_tf, U=r, T=t)
            y = np.interp(t, t_out, y); e = r - y
            
            return self._calculate_cost(t, e, obj_type, y)
        except: return 1e6

    def _objective_nonlinear_calc(self, params, obj_type, x0, c0, sim_time, input_type):
        try:
            y0 = np.concatenate([x0, c0])
            
            def combined_model(t, y_vec):
                x = y_vec[:len(x0)]; c = y_vec[len(x0):]
                # Input Gen
                if input_type == "Step": r = 1.0
                elif input_type == "Ramp": r = t
                else: r = 0.5 * t ** 2
                
                u, c_dot = self.nl_controller_model(t, x, r, c, params)
                x_dot = self.nl_plant_model(t, x, u)
                if not isinstance(c_dot, (np.ndarray, list, tuple)): c_dot = np.array([c_dot])
                return np.concatenate([x_dot, c_dot])

            sol = solve_ivp(combined_model, [0, sim_time], y0, 
                            t_eval=np.linspace(0, sim_time, 500), method='RK45')
            t = sol.t
            # Input Gen Array
            if input_type == "Step": r = np.ones_like(t)
            elif input_type == "Ramp": r = t
            else: r = 0.5 * t ** 2
            
            y = sol.y[0, :]; e = r - y
            return self._calculate_cost(t, e, obj_type, y)
        except: return 1e6

    def _calculate_cost(self, t, e, obj_type, y):
        dt = t[1] - t[0]
        ise = np.sum(e**2) * dt
        iae = np.sum(np.abs(e)) * dt
        itae = np.sum(t * np.abs(e)) * dt
        itse = np.sum(t**2 * e**2) * dt
        iate = np.sum(t * np.abs(e)) * dt
        
        obj_type = obj_type.upper()
        if obj_type == "ISE": J = ise
        elif obj_type == "IAE": J = iae
        elif obj_type == "ITAE": J = itae
        elif obj_type == "ITSE": J = itse
        elif obj_type == "IATE": J = iate
        elif obj_type == "ISE+IAE": J = ise + iae
        elif obj_type == "ISE+ITAE": J = ise + itae
        elif obj_type == "IAE+ITAE": J = iae + itae
        elif obj_type == "ISE+IAE+ITAE": J = ise + iae + itae
        elif obj_type == "ALL": J = ise + iae + itae + itse + iate
        else: J = ise
        
        if np.isnan(y).any() or np.max(np.abs(y)) > 1e6: return 1e6
        return J

    # ---------------- Run Optimization (Threaded) ----------------
    def run_optimization(self):
        self._clear_metrics() 
        self.last_plot_data = None 
        
        # 1. Collect ALL Data needed for optimization
        try:
            data = {}
            data["mode"] = self.system_type_var.get()
            data["sim_time"] = self._get_sim_time()
            if data["sim_time"] is None: return
            
            data["optimizer_name"] = self.optimizer_var.get()
            try:
                data["pso_swarmsize"] = int(self.swarm_entry.get())
                data["pso_maxiter"] = int(self.iter_entry.get())
            except: 
                data["pso_swarmsize"] = 20; data["pso_maxiter"] = 50

            data["obj_type"] = self.obj_type_var.get()
            data["input_type"] = self.input_type_var.get()

            if data["mode"] == "Linear":
                self.plant_tf = self._get_plant_tf() # Create object here
                if self.plant_tf is None: 
                    messagebox.showerror("Error", "Check plant model."); return
                
                data["l_ctrl_type"] = self.controller_var.get()
                data["l_params"] = list(self.bounds_entries.keys())
                try:
                    data["l_lb"] = [float(self.bounds_entries[p][0].get()) for p in data["l_params"]]
                    data["l_ub"] = [float(self.bounds_entries[p][1].get()) for p in data["l_params"]]
                except ValueError: messagebox.showerror("Error", "Invalid Bounds"); return

            else: # Non-linear
                if not self._compile_nl_models(): return
                if self.nl_param_count == 0: messagebox.showerror("Error", "No params."); return
                
                try:
                    data["nl_lb"] = [float(self.nl_bounds_entries[f"P{i+1}"][0].get()) for i in range(self.nl_param_count)]
                    data["nl_ub"] = [float(self.nl_bounds_entries[f"P{i+1}"][1].get()) for i in range(self.nl_param_count)]
                    data["nl_x0"] = np.array(self._parse_vector_entry(self.nl_x0_entry))
                    c0_list = self._parse_vector_entry(self.nl_c0_entry)
                    data["nl_c0"] = np.array(c0_list) if c0_list else np.array([])
                except Exception as e: messagebox.showerror("Error", f"Invalid NL Config: {e}"); return

            # 2. Disable UI
            self.run_button.config(text="Optimizing... (Please Wait)", state=tk.DISABLED, bg="#555")
            
            # 3. Start Thread
            threading.Thread(target=self._run_optimization_thread, args=(data,), daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Could not start optimization:\n{e}")

    def _on_optimization_complete(self, xopt, fopt, error_msg):
        # Re-enable UI
        self.run_button.config(text="Run Optimization", state=tk.NORMAL, bg="#007bff")
        
        if error_msg:
            messagebox.showerror("Optimization Error", f"An error occurred in the solver thread:\n\n{error_msg}")
            return

        if xopt is not None:
            self.optimized_params = xopt
            
            self.kp_val_var.set("N/A"); self.ki_val_var.set("N/A")
            self.kd_val_var.set("N/A"); self.p4_val_var.set("N/A")

            if len(xopt) >= 1: self.kp_val_var.set(f"{xopt[0]:.4f}")
            if len(xopt) >= 2: self.ki_val_var.set(f"{xopt[1]:.4f}")
            if len(xopt) >= 3: self.kd_val_var.set(f"{xopt[2]:.4f}")
            if len(xopt) >= 4: self.p4_val_var.set(f"{xopt[3]:.4f}")
            self.metric_var.set(f"{fopt:.4f} ({self.obj_type_var.get()})")
            
            self.final_params_text.config(state=tk.NORMAL)
            self.final_params_text.delete("1.0", tk.END)
            
            mode = self.system_type_var.get()
            param_names = list(self.bounds_entries.keys()) if mode == "Linear" else self.nl_param_names[:len(xopt)]
            
            for name, val in zip(param_names, xopt):
                self.final_params_text.insert(tk.END, f"{name}: {val:.6f}\n")
            self.final_params_text.config(state=tk.DISABLED)

            self.plot_response()
        else:
            messagebox.showerror("Optimization Failed", "Did not find a valid solution.")

    # ---------------- Plot ----------------
    def plot_response(self):
        try:
            if self.optimized_params is None: return
            sim_time = self._get_sim_time();
            if sim_time is None: return
            
            mode = self.system_type_var.get()
            t = np.linspace(0, sim_time, 500)
            r = self._generate_input(t)
            
            if mode == "Linear":
                if self.plant_tf is None: return
                num, den = self.plant_tf.num, self.plant_tf.den
                p = self.optimized_params
                if num.ndim == 2 and num.shape[0] == 1: num = num.flatten()
                
                ctrl_type = self.controller_var.get()
                pid_tf = self._get_controller_tf(p, ctrl_type)
                
                loop_num = signal.convolve(pid_tf.num, num); loop_den = signal.convolve(pid_tf.den, den)
                if np.all(np.polyadd(loop_den, loop_num) == 0): messagebox.showwarning("Plot Error", "Unstable"); return
                cl_tf = signal.TransferFunction(loop_num, np.polyadd(loop_den, loop_num))
                
                t_out, y, _ = signal.lsim(cl_tf, U=r, T=t); y = np.interp(t, t_out, y)
                
            else: 
                x0 = np.array(self._parse_vector_entry(self.nl_x0_entry))
                c0_list = self._parse_vector_entry(self.nl_c0_entry)
                c0 = np.array(c0_list) if c0_list else np.array([])
                y0 = np.concatenate([x0, c0])
                params = self.optimized_params
                
                def combined_model(t, y_vec):
                    x = y_vec[:len(x0)]; c = y_vec[len(x0):]
                    r_t = self._generate_input(t)
                    u, c_dot = self.nl_controller_model(t, x, r_t, c, params)
                    x_dot = self.nl_plant_model(t, x, u)
                    if not isinstance(c_dot, (np.ndarray, list, tuple)): c_dot = np.array([c_dot])
                    return np.concatenate([x_dot, c_dot])

                sol = solve_ivp(combined_model, [0, sim_time], y0, t_eval=t, method='RK45')
                y = sol.y[0, :]

            self.ax.clear()
            self.ax.plot(t, r, "r--", label="Reference")
            self.ax.plot(t, y, "b", label=f"Response")
            self.ax.grid(True); self.ax.legend(); self.ax.set_xlabel("Time (s)"); self.ax.set_ylabel("Amplitude")
            self.ax.set_title(f"Closed-Loop Response ({self.obj_type_var.get()})"); self.ax.set_xlim(0, sim_time)
            self.last_plot_data = {"t": t, "r": r, "y": y} 
            self.canvas.draw()
            
            self._calculate_response_metrics(t, y, r)
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot: {e}")
        
    def export_to_excel(self):
        if self.optimized_params is None or self.last_plot_data is None:
            messagebox.showerror("Error", "No data to export.")
            return

        params = self.optimized_params
        mode = self.system_type_var.get()
        param_names = list(self.bounds_entries.keys()) if mode == "Linear" else self.nl_param_names[:len(params)]
        try: metric_val = float(self.metric_var.get().split(" ")[0])
        except ValueError: metric_val = "N/A"
            
        t = self.last_plot_data["t"]; r = self.last_plot_data["r"]; y = self.last_plot_data["y"]

        file_path = filedialog.asksaveasfilename(defaultextension=".xls", filetypes=[("Excel", "*.xls")], title="Save")
        if not file_path: return

        try:
            wb = xlwt.Workbook()
            ws_params = wb.add_sheet("Params")
            header_style = xlwt.easyxf('font: bold on')
            ws_params.write(0, 0, "Parameter", header_style); ws_params.write(0, 1, "Value", header_style)
            for i, (name, val) in enumerate(zip(param_names, params)):
                ws_params.write(i + 1, 0, name); ws_params.write(i + 1, 1, val)
            
            row = len(params) + 2
            ws_params.write(row, 0, "Metric", header_style); ws_params.write(row, 1, metric_val); row+=2
            ws_params.write(row, 0, "Metrics", header_style); row+=1
            ws_params.write(row, 0, "Rise Time"); ws_params.write(row, 1, self.rise_time_var.get()); row+=1
            ws_params.write(row, 0, "Settling Time"); ws_params.write(row, 1, self.settling_time_var.get()); row+=1
            ws_params.write(row, 0, "Overshoot"); ws_params.write(row, 1, self.overshoot_var.get()); row+=1
            ws_params.write(row, 0, "Peak Time"); ws_params.write(row, 1, self.peak_time_var.get())

            ws_data = wb.add_sheet("Data")
            ws_data.write(0, 0, "Time", header_style); ws_data.write(0, 1, "Ref", header_style); ws_data.write(0, 2, "Resp", header_style)
            for i in range(len(t)):
                ws_data.write(i + 1, 0, t[i]); ws_data.write(i + 1, 1, r[i]); ws_data.write(i + 1, 2, y[i])
            wb.save(file_path)
            messagebox.showinfo("Success", "Exported successfully.")
        except Exception as e:
            messagebox.showerror("Export Error", f"{e}")
            
    def _clear_metrics(self):
        self.rise_time_var.set("N/A"); self.settling_time_var.set("N/A")
        self.overshoot_var.set("N/A"); self.peak_time_var.set("N/A")
            
    def _calculate_response_metrics(self, t, y, r):
        try:
            if not np.all(r == r[0]): self._clear_metrics(); return
            y_final = y[-1]
            
            if not np.isclose(y_final, 0): 
                y_max = np.max(y)
                peak_time = t[np.argmax(y)]
                overshoot = ((y_max - y_final) / (y_final)) * 100.0
                
                settling_thresh_high = y_final * 1.02
                settling_thresh_low = y_final * 0.98
                settled_indices = np.where((y > settling_thresh_high) | (y < settling_thresh_low))[0]
                
                if settled_indices.size > 0:
                    last_unsettled = settled_indices[-1]
                    settling_time = t[last_unsettled + 1] if last_unsettled + 1 < len(t) else "N/A"
                else: settling_time = t[0]
                    
                try:
                    rise_10 = t[np.where(y >= y_final * 0.1)[0][0]]
                    rise_90 = t[np.where(y >= y_final * 0.9)[0][0]]
                    rise_time = rise_90 - rise_10
                except IndexError: rise_time = "N/A" 
                
                self.rise_time_var.set(f"{rise_time:.4f} s" if isinstance(rise_time, float) else rise_time)
                self.settling_time_var.set(f"{settling_time:.4f} s" if isinstance(settling_time, float) else settling_time)
                self.overshoot_var.set(f"{overshoot:.2f} %")
                self.peak_time_var.set(f"{peak_time:.4f} s")
            else: self._clear_metrics()
        except Exception: self._clear_metrics()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() 

    splash = tk.Toplevel(root)
    splash.overrideredirect(True) 
    splash.config(bg="#222222") 

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    w, h = 450, 250
    x = (sw / 2) - (w / 2)
    y = (sh / 2) - (h / 2)
    splash.geometry(f'{w}x{h}+{int(x)}+{int(y)}')

    title_font = font.Font(family="Arial", size=36, weight="bold")
    creator_font = font.Font(family="Arial", size=12)

    tk.Label(splash, text="ControMyzation", font=title_font, bg="#222222", fg="#007bff").pack(pady=(60, 10))
    tk.Label(splash, text="Precision in Control with the Power of Optimization", font=creator_font, bg="#222222", fg="white").pack(pady=(0, 60))
    
    loading_label = tk.Label(splash, text="Created by Dayarnab & Mitradip", font=creator_font, bg="#222222", fg="#14E7B2")
    loading_label.pack(side=tk.BOTTOM, pady=5)

    splash.update() 

    def load_and_show():
        app = PIDTunerApp(root) 
        splash.destroy()
        root.deiconify() 

    root.after(5000, load_and_show) 
    root.mainloop()