"""
Fair A/B:
A) SciPy solve_ivp (RK45), NO dense_output; get native steps, then linear-resample
B) Numba RK45 (Cash–Karp) with the same stopping rules
Both use the same resampler → apples-to-apples. Dipole field is corrected.
"""

import time
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from numba import njit

# ----------------------------
# Shared problem setup
# ----------------------------

def B_dipole(r):
    """Unit-dipole at origin aligned with z (up to constant factor)."""
    x, y, z = r
    rr = x*x + y*y + z*z
    r1 = np.sqrt(rr)
    if r1 == 0.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    r3 = rr * r1           # r^3
    r5 = rr * r3           # r^5
    mr = z                 # m=(0,0,1)·r
    # B = (3(m·r) r)/r^5 - m/r^3
    return 3.0*mr*np.array([x, y, z])/r5 - np.array([0.0, 0.0, 1.0])/r3

def Bhat(r, K):
    B = B_dipole(r)
    n = norm(B)
    if n == 0.0:
        return np.zeros(3)
    u = B / n
    return u if K >= 0.0 else -u

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
# A) SciPy solve_ivp (RK45, no dense_output)
# ----------------------------

def integrate_scipy_linear(r0, K):
    def rhs(_, r): return Bhat(r, K)
    def hit_surface(t, r): return r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - Rs2
    hit_surface.terminal = True; hit_surface.direction = -1
    def hit_Rmax(t, r):   return r[0]*r[0] + r[1]*r[1] + r[2]*r[2] - Rm2
    hit_Rmax.terminal = True; hit_Rmax.direction = +1

    t_span = (t0, t1_guess)
    t_start = time.perf_counter()
    res = solve_ivp(
        fun=rhs, t_span=t_span, y0=r0,
        method="RK45", rtol=rtol, atol=atol, max_step=max_step,
        dense_output=False, events=(hit_surface, hit_Rmax)
    )
    wall = time.perf_counter() - t_start
    # Build uniform arc-length grid up to actual stop
    t_stop = res.t[-1]
    if t_stop >= t0:
        t_eval = np.append(np.arange(t0, t_stop, ds), t_stop)
    else:
        t_eval = np.append(np.arange(t0, t_stop, -ds), t_stop)
    Y = resample_linear(res.t, res.y, t_eval)
    info = {"nfev": res.nfev, "time_s": wall, "t_stop": t_stop, "samples": t_eval.size}
    return t_eval, Y, info

# ----------------------------
# B) Numba Cash–Karp RK45 (same events)
# ----------------------------

@njit(fastmath=True, cache=True)
def B_dipole_numba(r):
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
    B = B_dipole_numba(r)
    n = np.sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2])
    if n == 0.0:
        u0 = 0.0; u1 = 0.0; u2 = 0.0
    else:
        inv = 1.0/n
        u0 = B[0]*inv; u1 = B[1]*inv; u2 = B[2]*inv
    if K >= 0.0:
        return np.array((u0, u1, u2))
    else:
        return np.array((-u0, -u1, -u2))

# Cash–Karp tableau
B5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50,  2/55], dtype=np.float64)
B4 = np.array([25/216,  0, 1408/2565, 2197/4104,   -1/5,   0   ], dtype=np.float64)
C2 = np.array([1/5], dtype=np.float64)
C3 = np.array([3/40, 9/40], dtype=np.float64)
C4 = np.array([3/10, -9/10, 6/5], dtype=np.float64)
C5 = np.array([-11/54, 5/2, -70/27, 35/27], dtype=np.float64)
C6 = np.array([1631/55296, 175/512, 575/13824, 44275/110592, 253/4096], dtype=np.float64)

@njit(fastmath=True, cache=True)
def rkck_step(r, h, K):
    k1 = Bhat_numba(r, K)
    r2 = r + h*(C2[0]*k1)
    k2 = Bhat_numba(r2, K)
    r3 = r + h*(C3[0]*k1 + C3[1]*k2)
    k3 = Bhat_numba(r3, K)
    r4 = r + h*(C4[0]*k1 + C4[1]*k2 + C4[2]*k3)
    k4 = Bhat_numba(r4, K)
    r5 = r + h*(C5[0]*k1 + C5[1]*k2 + C5[2]*k3 + C5[3]*k4)
    k5 = Bhat_numba(r5, K)
    r6 = r + h*(C6[0]*k1 + C6[1]*k2 + C6[2]*k3 + C6[3]*k4 + C6[4]*k5)
    k6 = Bhat_numba(r6, K)
    r5th = r + h*(B5[0]*k1 + B5[2]*k3 + B5[3]*k4 + B5[4]*k5 + B5[5]*k6)
    r4th = r + h*(B4[0]*k1 + B4[2]*k3 + B4[3]*k4 + B4[4]*k5)
    e0 = abs(r5th[0]-r4th[0]); e1 = abs(r5th[1]-r4th[1]); e2 = abs(r5th[2]-r4th[2])
    err = max(e0, e1, e2)
    return r5th, err

@njit(fastmath=True, cache=True)
def integrate_numba_native_steps(r0, K, t0, t1_guess, rtol, atol, h_max, R_SURF, R_MAX):
    """Return native step times T and states Y(3,N)."""
    Rs2 = R_SURF*R_SURF
    Rm2 = R_MAX*R_MAX
    # Over-allocate; shrink later
    T = np.empty(200000, dtype=np.float64)
    Y = np.empty((3, 200000), dtype=np.float64)
    r = r0.copy()
    t = t0
    h = min(0.25*h_max, h_max)
    i = 0
    T[i] = t; Y[:, i] = r; i += 1
    safety = 0.9; min_shrink = 0.2; max_grow = 5.0
    while True:
        if t >= t1_guess: break
        h = min(h, t1_guess - t)
        if h <= 0.0: break
        r5, err = rkck_step(r, h, K)
        tol = atol + rtol * max(abs(r5[0]), abs(r5[1]), abs(r5[2]))
        if err <= tol:
            t += h; r = r5
            # record step
            if i >= T.size:
                # grow (rare in tests)
                T = np.resize(T, T.size*2)
                Y = np.resize(Y, (3, Y.shape[1]*2))
            T[i] = t; Y[:, i] = r; i += 1
            # event?
            rr = r[0]*r[0] + r[1]*r[1] + r[2]*r[2]
            if rr <= Rs2 or rr >= Rm2:
                break
            # propose next h
            if err == 0.0: h = min(h_max, h*max_grow)
            else:          h = h * min(max_grow, max(min_shrink, safety*(tol/err)**0.2))
        else:
            h = h * max(min_shrink, safety*(tol/err)**0.25)
            if h < 1e-16: break
    return T[:i], Y[:, :i]

# ----------------------------
# Run & compare
# ----------------------------

if __name__ == "__main__":
    # Compile Numba once with first seed
    _ = integrate_numba_native_steps(seeds[0], 1.0, t0, t1_guess, rtol, atol, max_step, R_SURF, R_MAX)
    
    results = []
    
    for seed_idx, r0 in enumerate(seeds):
        K = 1.0 if np.dot(B_dipole(r0), r0) >= 0.0 else -1.0
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx+1}/{len(seeds)}: r0={r0}, K={K:.1f}")
        
        # --- SciPy ---
        t1 = time.perf_counter()
        T_sci, Y_sci, info = integrate_scipy_linear(r0, K)
        t_sci = time.perf_counter() - t1
        
        # --- Numba ---
        t2 = time.perf_counter()
        Tn, Yn = integrate_numba_native_steps(r0, K, t0, t1_guess, rtol, atol, max_step, R_SURF, R_MAX)
        t_numba = time.perf_counter() - t2
        
        # resample both to the SAME uniform ds grid up to the EARLIER stop
        t_stop = min(T_sci[-1], Tn[-1])
        t_eval = np.append(np.arange(t0, t_stop, ds), t_stop)
        Y_s = resample_linear(T_sci, Y_sci, t_eval)
        Y_n = resample_linear(Tn, Yn, t_eval)
        
        dmax = np.max(np.abs(Y_s - Y_n))
        
        results.append({
            "seed": r0,
            "K": K,
            "scipy_time_ms": info['time_s']*1e3,
            "numba_time_ms": t_numba*1e3,
            "scipy_nfev": info['nfev'],
            "numba_steps": Tn.size,
            "t_stop": t_stop,
            "samples": t_eval.size,
            "dmax": dmax
        })
        
        print(f"[SciPy RK45] nfev={info['nfev']} stop={t_stop:.6f} "
              f"samples={t_eval.size} time={info['time_s']*1e3:.1f} ms")
        print(f"[Numba RKCK] steps={Tn.size} time={t_numba*1e3:.1f} ms")
        print(f"Max |Δr| = {dmax:.3e}")
    
    # Summary table
    print(f"\n{'='*60}")
    print("Summary Table:")
    print(f"{'Seed':<20} {'K':>4} {'SciPy(ms)':>10} {'Numba(ms)':>10} "
          f"{'Speedup':>8} {'dmax':>12}")
    print("-"*60)
    for r in results:
        speedup = r["scipy_time_ms"] / r["numba_time_ms"] if r["numba_time_ms"] > 0 else 0
        seed_str = f"({r['seed'][0]:.1f},{r['seed'][1]:.1f},{r['seed'][2]:.1f})"
        print(f"{seed_str:<20} {r['K']:>4.1f} {r['scipy_time_ms']:>10.1f} "
              f"{r['numba_time_ms']:>10.1f} {speedup:>8.2f}x {r['dmax']:>12.3e}")
