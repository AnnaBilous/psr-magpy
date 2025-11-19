#!/usr/bin/env python3
import numpy as np
import CyRK
from CyRK import pysolve_ivp

print("CyRK:", getattr(CyRK, "__version__", "?"))

# --- ODE: y' = 1 ---
def rhs(t, y):
    return np.ones_like(y)

# --- Terminal event at t = 1.0 ---
def ev_t1(t, y):
    return t - 1.0
ev_t1.terminal = True
ev_t1.direction = 1

y0 = np.array([0.0], dtype=float)
time_span = (0.0, 2.0)

sol = pysolve_ivp(
    py_diffeq=rhs,
    time_span=time_span,
    y0=y0,
    method="RK45",
    rtol=1e-9,
    atol=1e-12,
    max_step=0.1,
    events=(ev_t1,),
    # t_eval=None  # omit; adaptive steps
)

# --- Access fields on the object ---
# These names mirror SciPy closely in 0.16+
print("success:", getattr(sol, "success", True))
print("message:", getattr(sol, "message", ""))

t = sol.t            # 1D array of times
y = sol.y            # shape (n, m)
print("t_final:", float(t[-1]))
print("y_final:", float(y[-1] if y.ndim == 1 else y[:, -1]))

# Event diagnostics (present when events were provided)
if hasattr(sol, "t_events"):
    print("t_events:", [arr.tolist() for arr in sol.t_events])
if hasattr(sol, "y_events"):
    print("y_events shapes:", [arr.shape for arr in sol.y_events])
if hasattr(sol, "event_terminated"):
    print("event_terminated:", sol.event_terminated)
if hasattr(sol, "event_terminate_index"):
    print("event_terminate_index:", sol.event_terminate_index)
