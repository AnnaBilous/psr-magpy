"""
Three-way benchmark:
A1) SciPy solve_ivp (RK45) with Python field evaluation
A2) SciPy solve_ivp (RK45) with Numba-compiled field evaluation
B)  CyRK pysolve_ivp (RK45) with Numba-compiled field evaluation
All use the same resampler → apples-to-apples. Dipole field integration.
"""

import time
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from CyRK import pysolve_ivp
from numba import njit

# ----------------------------
# Shared problem setup
# ----------------------------

# Original Python versions
def B_dipole_py(r):
    """Python dipole field (original, not compiled)."""
    x, y, z = r
    rr = x*x + y*y + z*z
    r1 = np.sqrt(rr)
    if r1 == 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    r3 = rr * r1
    r5 = rr * r3
    mr = z
    return 3.0*mr*np.array([x, y, z])/r5 - np.array([0.0, 0.0, 1.0])/r3

def Bhat_py(r, K):
    """Python unit field (original, not compiled)."""
    B = B_dipole_py(r)
    n = norm(B)
    if n == 0.0:
        return np.zeros(3)
    u = B / n
    return u if K >= 0.0 else -u

# Numba-compiled versions
@njit(fastmath=True, cache=True)
def B_dipole_numba(r):
    """Numba-compiled dipole field."""
    x, y, z = r[0], r[1], r[2]
    rr = x*x + y*y + z*z
    r1 = np.sqrt(rr)
    if r1 == 0.0:
        return np.array((0.0, 0.0, 0.0))
    r3 = rr * r1
    r5 = rr * r3
    mr = z
    Bx = 3.0*mr*x/r5
    By = 3.0*mr*y/r5
    Bz = 3.0*mr*z/r5 - 1.0/r3
    return np.array((Bx, By, Bz))

@njit(fastmath=True, cache=True)
def Bhat_numba(r, K):
    """Numba-compiled unit field."""
    B = B_dipole_numba(r)
    n = np.sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2])
    if n == 0.0:
        return np.array((0.0, 0.0, 0.0))
    inv = 1.0/n
    u0 = B[0]*inv
    u1 = B[1]*inv
    u2 = B[2]*inv
    if K >= 0.0:
        return np.array((u0, u1, u2))
    else:
        return np.array((-u0, -u1, -u2))

# Domain
R_SURF = 1.0
R_MAX  = 200.0
Rs2, Rm2 = R_SURF**2, R_MAX**2

# Integration control (tighten if you want smaller dmax)
t0, t1_guess = 0.0, 4000.0   # arc-length upper bound; events stop earlier
rtol, atol   = 5e-8, 1e-10  # tighter → closer match
max_step     = 0.05
ds           = 0.02         # uniform arc-length sampling

# Multiple seeds to test
seeds = [
    np.array([2.5, 1.0, 1.5], dtype=np.float64),
    np.array([1.5, 2.0, 1.0], dtype=np.float64),
    np.array([2.0, 0.5, 2.0], dtype=np.float64),
    np.array([3.0, 0.0, 0.0], dtype=np.float64),  # Equatorial
    np.array([1.5, 1.5, 1.5], dtype=np.float64),  # Near surface
]

# ----------------------------
# Common linear resampler
# ----------------------------

def resample_linear(T, Y, t_eval):
    """
    Linear interpolation from native solver times T (monotonic) and states Y(3,N)
    to requested times t_eval (subset/within [T0, T_end]).
    """
    T = np.asarray(T)
    Y = np.asarray(Y)
    M = t_eval.size
    out = np.empty((3, M), dtype=np.float64)
    j = 0
    for i in range(M):
        ti = t_eval[i]
        # advance j until T[j] <= ti <= T[j+1]
        while j+1 < T.size and T[j+1] < ti:
            j += 1
        if j+1 == T.size:
            out[:, i] = Y[:, -1]
        else:
            t0_, t1_ = T[j], T[j+1]
            if t1_ == t0_:
                out[:, i] = Y[:, j]
            else:
                a = (ti - t0_) / (t1_ - t0_)
                out[:, i] = (1.0 - a)*Y[:, j] + a*Y[:, j+1]
    return out

# ----------------------------
# A1) SciPy solve_ivp with Python field (RK45, no dense_output)
# ----------------------------

def integrate_scipy_py_linear(r0, K):
    t0_setup = time.perf_counter()
    def rhs(_, r): return Bhat_py(r, K)
    def hit_surface(t, r): return r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - Rs2
    hit_surface.terminal = True; hit_surface.direction = -1
    def hit_Rmax(t, r):   return r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - Rm2
    hit_Rmax.terminal = True; hit_Rmax.direction = +1
    t_span = (t0, t1_guess)
    t_setup = time.perf_counter() - t0_setup

    t0_solve = time.perf_counter()
    res = solve_ivp(
        fun=rhs, t_span=t_span, y0=r0,
        method="RK45", rtol=rtol, atol=atol, max_step=max_step,
        dense_output=False, events=(hit_surface, hit_Rmax)
    )
    t_solve = time.perf_counter() - t0_solve
    
    t0_post = time.perf_counter()
    t_stop = res.t[-1]
    if t_stop >= t0:
        t_eval = np.append(np.arange(t0, t_stop, ds), t_stop)
    else:
        t_eval = np.append(np.arange(t0, t_stop, -ds), t_stop)
    t_post1 = time.perf_counter() - t0_post
    
    t0_resample = time.perf_counter()
    Y = resample_linear(res.t, res.y, t_eval)
    t_resample = time.perf_counter() - t0_resample
    
    info = {
        "nfev": res.nfev, 
        "time_s": t_setup + t_solve + t_post1 + t_resample,
        "time_setup_ms": t_setup * 1e3,
        "time_solve_ms": t_solve * 1e3,
        "time_post_ms": t_post1 * 1e3,
        "time_resample_ms": t_resample * 1e3,
        "t_stop": t_stop, 
        "samples": t_eval.size
    }
    return t_eval, Y, info

# ----------------------------
# A2) SciPy solve_ivp with Numba field (RK45, no dense_output)
# ----------------------------

def integrate_scipy_numba_linear(r0, K):
    t0_setup = time.perf_counter()
    def rhs(_, r): return Bhat_numba(r, K)
    def hit_surface(t, r): return r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - Rs2
    hit_surface.terminal = True; hit_surface.direction = -1
    def hit_Rmax(t, r):   return r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - Rm2
    hit_Rmax.terminal = True; hit_Rmax.direction = +1
    t_span = (t0, t1_guess)
    t_setup = time.perf_counter() - t0_setup

    t0_solve = time.perf_counter()
    res = solve_ivp(
        fun=rhs, t_span=t_span, y0=r0,
        method="RK45", rtol=rtol, atol=atol, max_step=max_step,
        dense_output=False, events=(hit_surface, hit_Rmax)
    )
    t_solve = time.perf_counter() - t0_solve
    
    t0_post = time.perf_counter()
    t_stop = res.t[-1]
    if t_stop >= t0:
        t_eval = np.append(np.arange(t0, t_stop, ds), t_stop)
    else:
        t_eval = np.append(np.arange(t0, t_stop, -ds), t_stop)
    t_post1 = time.perf_counter() - t0_post
    
    t0_resample = time.perf_counter()
    Y = resample_linear(res.t, res.y, t_eval)
    t_resample = time.perf_counter() - t0_resample
    
    info = {
        "nfev": res.nfev, 
        "time_s": t_setup + t_solve + t_post1 + t_resample,
        "time_setup_ms": t_setup * 1e3,
        "time_solve_ms": t_solve * 1e3,
        "time_post_ms": t_post1 * 1e3,
        "time_resample_ms": t_resample * 1e3,
        "t_stop": t_stop, 
        "samples": t_eval.size
    }
    return t_eval, Y, info

# ----------------------------
# B) CyRK pysolve_ivp (RK45, native events)
# ----------------------------

def integrate_cyrk_linear(r0, K):
    """Integrate using CyRK with Numba field and native events, then resample."""
    t0_setup = time.perf_counter()
    def rhs(t, r): return Bhat_numba(r, K)
    
    def hit_surface(t, r): 
        return r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - Rs2
    hit_surface.terminal = True
    hit_surface.direction = -1
    
    def hit_Rmax(t, r): 
        return r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - Rm2
    hit_Rmax.terminal = True
    hit_Rmax.direction = +1
    
    t_span = (t0, t1_guess)
    t_setup = time.perf_counter() - t0_setup
    
    t0_solve = time.perf_counter()
    sol = pysolve_ivp(
        py_diffeq=rhs,
        time_span=t_span,
        y0=r0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=(hit_surface, hit_Rmax),
    )
    t_solve = time.perf_counter() - t0_solve
    
    t0_extract = time.perf_counter()
    T_nat = sol.t  # 1D array of times
    Y_nat = sol.y  # shape (n, m) where n is state dim, m is num steps
    
    # Handle shape: CyRK might return (n,) or (n, m)
    if Y_nat.ndim == 1:
        Y_nat = Y_nat.reshape((Y_nat.size, 1))
    if Y_nat.shape[0] != 3:
        # Transpose if needed: should be (3, m)
        Y_nat = Y_nat.T if Y_nat.shape[1] == 3 else Y_nat
    t_extract = time.perf_counter() - t0_extract
    
    t0_post = time.perf_counter()
    t_stop = float(T_nat[-1])
    if t_stop >= t0:
        t_eval = np.append(np.arange(t0, t_stop, ds), t_stop)
    else:
        t_eval = np.append(np.arange(t0, t_stop, -ds), t_stop)
    t_post1 = time.perf_counter() - t0_post
    
    t0_resample = time.perf_counter()
    Y = resample_linear(T_nat, Y_nat, t_eval)
    t_resample = time.perf_counter() - t0_resample
    
    nfev = getattr(sol, 'nfev', None)
    
    info = {
        "nfev": nfev, 
        "time_s": t_setup + t_solve + t_extract + t_post1 + t_resample,
        "time_setup_ms": t_setup * 1e3,
        "time_solve_ms": t_solve * 1e3,
        "time_extract_ms": t_extract * 1e3,
        "time_post_ms": t_post1 * 1e3,
        "time_resample_ms": t_resample * 1e3,
        "t_stop": t_stop, 
        "samples": t_eval.size,
        "native_steps": T_nat.size
    }
    return t_eval, Y, info

# ----------------------------
# Run & compare
# ----------------------------

if __name__ == "__main__":
    # Warmup: compile Numba functions
    print("Compiling Numba functions...")
    _r_test = np.array([2.5, 1.0, 1.5], dtype=np.float64)
    _ = B_dipole_numba(_r_test)
    _ = Bhat_numba(_r_test, 1.0)
    print("Compilation complete.\n")
    
    results = []
    
    for seed_idx, r0 in enumerate(seeds):
        t0_init = time.perf_counter()
        K = 1.0 if np.dot(B_dipole_py(r0), r0) >= 0.0 else -1.0
        t_init = time.perf_counter() - t0_init
        
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx+1}/{len(seeds)}: r0={r0}, K={K:.1f}")
        print(f"  Init: {t_init*1e3:.3f} ms")
        
        # --- SciPy with Python field ---
        T_sci_py, Y_sci_py, info_sci_py = integrate_scipy_py_linear(r0, K)
        
        # --- SciPy with Numba field ---
        T_sci_nb, Y_sci_nb, info_sci_nb = integrate_scipy_numba_linear(r0, K)
        
        # --- CyRK with Numba field ---
        T_cyr, Y_cyr, info_cyr = integrate_cyrk_linear(r0, K)
        
        # Comparison: resample all to the SAME uniform ds grid
        t0_compare = time.perf_counter()
        t_stop = min(T_sci_py[-1], T_sci_nb[-1], T_cyr[-1])
        t_eval = np.append(np.arange(t0, t_stop, ds), t_stop)
        t_compare1 = time.perf_counter() - t0_compare
        
        t0_resample = time.perf_counter()
        Y_sci_py_uni = resample_linear(T_sci_py, Y_sci_py, t_eval)
        Y_sci_nb_uni = resample_linear(T_sci_nb, Y_sci_nb, t_eval)
        Y_cyr_uni = resample_linear(T_cyr, Y_cyr, t_eval)
        t_resample = time.perf_counter() - t0_resample
        
        t0_dmax = time.perf_counter()
        dmax_sci_py_vs_nb = np.max(np.abs(Y_sci_py_uni - Y_sci_nb_uni))
        dmax_sci_nb_vs_cyr = np.max(np.abs(Y_sci_nb_uni - Y_cyr_uni))
        dmax_sci_py_vs_cyr = np.max(np.abs(Y_sci_py_uni - Y_cyr_uni))
        t_dmax = time.perf_counter() - t0_dmax
        
        results.append({
            "seed": r0,
            "K": K,
            "scipy_py_time_ms": info_sci_py['time_s']*1e3,
            "scipy_nb_time_ms": info_sci_nb['time_s']*1e3,
            "cyrk_time_ms": info_cyr['time_s']*1e3,
            "scipy_py_nfev": info_sci_py['nfev'],
            "scipy_nb_nfev": info_sci_nb['nfev'],
            "cyrk_native_steps": info_cyr.get('native_steps', None),
            "t_stop": t_stop,
            "samples": t_eval.size,
            "dmax_sci_py_vs_nb": dmax_sci_py_vs_nb,
            "dmax_sci_nb_vs_cyr": dmax_sci_nb_vs_cyr,
            "dmax_sci_py_vs_cyr": dmax_sci_py_vs_cyr,
        })
        
        # Detailed timing breakdown
        print(f"\n  SciPy-Py breakdown:")
        print(f"    setup:     {info_sci_py['time_setup_ms']:>8.3f} ms")
        print(f"    solve:     {info_sci_py['time_solve_ms']:>8.3f} ms")
        print(f"    post:      {info_sci_py['time_post_ms']:>8.3f} ms")
        print(f"    resample:  {info_sci_py['time_resample_ms']:>8.3f} ms")
        print(f"    total:     {info_sci_py['time_s']*1e3:>8.3f} ms")
        
        print(f"\n  SciPy-Numba breakdown:")
        print(f"    setup:     {info_sci_nb['time_setup_ms']:>8.3f} ms")
        print(f"    solve:     {info_sci_nb['time_solve_ms']:>8.3f} ms")
        print(f"    post:      {info_sci_nb['time_post_ms']:>8.3f} ms")
        print(f"    resample:  {info_sci_nb['time_resample_ms']:>8.3f} ms")
        print(f"    total:     {info_sci_nb['time_s']*1e3:>8.3f} ms")
        
        print(f"\n  CyRK-Numba breakdown:")
        print(f"    setup:     {info_cyr['time_setup_ms']:>8.3f} ms")
        print(f"    solve:     {info_cyr['time_solve_ms']:>8.3f} ms")
        print(f"    extract:   {info_cyr['time_extract_ms']:>8.3f} ms")
        print(f"    post:      {info_cyr['time_post_ms']:>8.3f} ms")
        print(f"    resample:  {info_cyr['time_resample_ms']:>8.3f} ms")
        print(f"    total:     {info_cyr['time_s']*1e3:>8.3f} ms")
        
        print(f"\n  Comparison:")
        print(f"    common_grid: {t_compare1*1e3:>8.3f} ms")
        print(f"    resample_all: {t_resample*1e3:>8.3f} ms")
        print(f"    dmax_calc:    {t_dmax*1e3:>8.3f} ms")
        
        print(f"\n  Accuracy (Max |Δr|):")
        print(f"    SciPy-Py vs SciPy-Numba: {dmax_sci_py_vs_nb:.3e}")
        print(f"    SciPy-Numba vs CyRK:     {dmax_sci_nb_vs_cyr:.3e}")
        print(f"    SciPy-Py vs CyRK:        {dmax_sci_py_vs_cyr:.3e}")
        
        speedup_nb_vs_py = info_sci_py['time_s'] / info_sci_nb['time_s'] if info_sci_nb['time_s'] > 0 else 0
        speedup_cyr_vs_py = info_sci_py['time_s'] / info_cyr['time_s'] if info_cyr['time_s'] > 0 else 0
        speedup_cyr_vs_nb = info_sci_nb['time_s'] / info_cyr['time_s'] if info_cyr['time_s'] > 0 else 0
        print(f"\n  Speedup:")
        print(f"    Numba vs Py (SciPy): {speedup_nb_vs_py:.2f}x")
        print(f"    CyRK vs Py (SciPy):  {speedup_cyr_vs_py:.2f}x")
        print(f"    CyRK vs Numba (SciPy): {speedup_cyr_vs_nb:.2f}x")
    
    # Summary table
    print(f"\n{'='*80}")
    print("Summary Table:")
    print(f"{'Seed':<20} {'K':>4} {'SciPy-Py':>10} {'SciPy-Nb':>10} {'CyRK-Nb':>10} "
          f"{'Nb/Py':>8} {'CyRK/Py':>8} {'CyRK/Nb':>8}")
    print("-"*80)
    for r in results:
        speedup_nb_py = r["scipy_py_time_ms"] / r["scipy_nb_time_ms"] if r["scipy_nb_time_ms"] > 0 else 0
        speedup_cyr_py = r["scipy_py_time_ms"] / r["cyrk_time_ms"] if r["cyrk_time_ms"] > 0 else 0
        speedup_cyr_nb = r["scipy_nb_time_ms"] / r["cyrk_time_ms"] if r["cyrk_time_ms"] > 0 else 0
        seed_str = f"({r['seed'][0]:.1f},{r['seed'][1]:.1f},{r['seed'][2]:.1f})"
        print(f"{seed_str:<20} {r['K']:>4.1f} {r['scipy_py_time_ms']:>10.1f} "
              f"{r['scipy_nb_time_ms']:>10.1f} {r['cyrk_time_ms']:>10.1f} "
              f"{speedup_nb_py:>8.2f}x {speedup_cyr_py:>8.2f}x {speedup_cyr_nb:>8.2f}x")
    
    print(f"\n{'='*80}")
    print("Accuracy Summary (Max |Δr|):")
    print(f"{'Seed':<20} {'Py vs Nb':>12} {'Nb vs CyRK':>12} {'Py vs CyRK':>12}")
    print("-"*80)
    for r in results:
        seed_str = f"({r['seed'][0]:.1f},{r['seed'][1]:.1f},{r['seed'][2]:.1f})"
        print(f"{seed_str:<20} {r['dmax_sci_py_vs_nb']:>12.3e} {r['dmax_sci_nb_vs_cyr']:>12.3e} "
              f"{r['dmax_sci_py_vs_cyr']:>12.3e}")

