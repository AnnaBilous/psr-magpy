"""
ODE solver abstraction for field line integration.

Provides unified interface for different ODE solvers (SciPy, Numba, CyRK).
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyTracingError

try:
    from CyRK import pysolve_ivp  # type: ignore[attr-defined]
    CYRK_AVAILABLE = True
except ImportError:
    CYRK_AVAILABLE = False
    pysolve_ivp = None

logger = get_logger(__name__, console_format_brief=True)


class SolverError(PSRMagPyTracingError):
    """Unified exception for ODE solver errors.
    
    Wraps solver-specific exceptions to provide consistent error handling
    across different solver backends (SciPy, CyRK, etc.).
    
    Attributes
    ----------
    solver_name : str
        Name of the solver that raised the error (e.g., "scipy", "cyrk").
    original_error : Exception
        The original exception raised by the solver (if any).
    """
    
    def __init__(self, message: str, solver_name: str, original_error: Optional[Exception] = None):
        self.solver_name = solver_name
        self.original_error = original_error
        # Log with solver context, then call parent with original message
        logger.error(f"[{solver_name}] {message}")
        # Call PSRMagPyTracingError.__init__ but skip its logging (already logged above)
        # We need to set the message directly to avoid double logging
        self.message = message
        Exception.__init__(self, message)


@dataclass
class Solution:
    """Unified solution interface for ODE solvers.
    
    Attributes
    ----------
    t : np.ndarray
        Time points (1D array).
    y : np.ndarray
        Solution values (shape: (n, m) where n is state dimension, m is number of time points).
    nfev : int
        Number of function evaluations.
    success : bool
        Whether integration was successful.
    message : str
        Status message from solver.
    sol : Optional[Callable]
        Dense output function (if available). Callable as sol(t) -> np.ndarray.
    t_events : Optional[list]
        List of arrays containing event times for each event (if events were provided).
    y_events : Optional[list]
        List of arrays containing solution values at event times (if events were provided).
    """
    
    t: np.ndarray
    y: np.ndarray
    nfev: int
    success: bool
    message: str
    sol: Optional[Callable] = None
    t_events: Optional[list] = None
    y_events: Optional[list] = None


class ODESolver(ABC):
    """Abstract base class for ODE solvers."""
    
    @abstractmethod
    def solve(
        self,
        fun: Callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        events: Optional[Tuple[Callable, ...]] = None,
        method: str = "RK45",
        rtol: float = 1e-9,
        atol: float = 1e-9,
        max_step: float = 0.01,
        dense_output: bool = True,
        **kwargs
    ) -> Solution:
        """Solve ODE system.
        
        Parameters
        ----------
        fun : Callable
            Right-hand side function: fun(t, y, *args) -> dy/dt
        t_span : Tuple[float, float]
            Integration time span (t_start, t_end).
        y0 : np.ndarray
            Initial condition (1D array).
        events : Optional[Tuple[Callable, ...]]
            Tuple of event functions. Each event function should be:
            event(t, y, *args) -> float, with optional .terminal and .direction attributes.
        method : str
            Integration method (solver-specific).
        rtol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        max_step : float
            Maximum step size.
        dense_output : bool
            Whether to provide dense output (if supported).
        **kwargs
            Additional solver-specific parameters.
            
        Returns
        -------
        Solution
            Unified solution object.
        """
        pass


class SciPySolver(ODESolver):
    """SciPy-based ODE solver wrapping solve_ivp."""
    
    def solve(
        self,
        fun: Callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        events: Optional[Tuple[Callable, ...]] = None,
        method: str = "RK45",
        rtol: float = 1e-9,
        atol: float = 1e-9,
        max_step: float = 0.01,
        dense_output: bool = True,
        **kwargs
    ) -> Solution:
        """Solve ODE using SciPy's solve_ivp.
        
        Parameters
        ----------
        fun : Callable
            Right-hand side function: fun(t, y, *args) -> dy/dt
        t_span : Tuple[float, float]
            Integration time span (t_start, t_end).
        y0 : np.ndarray
            Initial condition (1D array).
        events : Optional[Tuple[Callable, ...]]
            Tuple of event functions.
        method : str
            Integration method (e.g., "RK45", "DOP853", "BDF", "Radau").
        rtol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        max_step : float
            Maximum step size.
        dense_output : bool
            Whether to provide dense output.
        **kwargs
            Additional arguments passed to solve_ivp (e.g., args for fun).
            
        Returns
        -------
        Solution
            Unified solution object.
        """
        # Prepare solve_ivp arguments
        solve_ivp_kwargs = {
            "fun": fun,
            "t_span": t_span,
            "y0": y0,
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "max_step": max_step,
            "dense_output": dense_output,
        }
        
        # Add events if provided
        if events is not None:
            solve_ivp_kwargs["events"] = events
        
        # Add any additional kwargs (e.g., args for fun)
        solve_ivp_kwargs.update(kwargs)
        
        # Call solve_ivp with error handling
        try:
            scipy_sol = solve_ivp(**solve_ivp_kwargs)
        except (ValueError, RuntimeError) as e:
            raise SolverError(f"SciPy solver failed: {e}", solver_name="scipy", original_error=e) from e
        except Exception as e:
            raise SolverError(f"Unexpected error in SciPy solver: {e}", solver_name="scipy", original_error=e) from e
        
        # Check if integration was successful
        if not scipy_sol.success:
            raise SolverError(f"SciPy solver integration failed: {scipy_sol.message}", solver_name="scipy")
        
        # Convert to unified Solution object
        return Solution(
            t=scipy_sol.t,
            y=scipy_sol.y,
            nfev=scipy_sol.nfev,
            success=scipy_sol.success,
            message=scipy_sol.message,
            sol=scipy_sol.sol if dense_output else None,
            t_events=getattr(scipy_sol, 't_events', None),
            y_events=getattr(scipy_sol, 'y_events', None),
        )


class CyRKSolver(ODESolver):
    """CyRK-based ODE solver using pysolve_ivp."""
    
    def __init__(self):
        """Initialize CyRK solver."""
        if not CYRK_AVAILABLE:
            raise ImportError("CyRK is not available. Install it with: pip install CyRK>=0.16.0")
    
    def solve(
        self,
        fun: Callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        events: Optional[Tuple[Callable, ...]] = None,
        method: str = "RK45",
        rtol: float = 1e-9,
        atol: float = 1e-9,
        max_step: float = 0.01,
        dense_output: bool = True,
        **kwargs
    ) -> Solution:
        """Solve ODE using CyRK's pysolve_ivp.
        
        Parameters
        ----------
        fun : Callable
            Right-hand side function: fun(t, y, *args) -> dy/dt
        t_span : Tuple[float, float]
            Integration time span (t_start, t_end).
        y0 : np.ndarray
            Initial condition (1D array).
        events : Optional[Tuple[Callable, ...]]
            Tuple of event functions.
        method : str
            Integration method (e.g., "RK45").
        rtol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        max_step : float
            Maximum step size.
        dense_output : bool
            Whether to provide dense output (via interpolation).
        **kwargs
            Additional arguments passed to pysolve_ivp (e.g., args for fun).
            
        Returns
        -------
        Solution
            Unified solution object.
        """
        # Convert rtol, atol to Python types for CyRK compatibility
        rtol_py = float(rtol)
        atol_py = float(atol)
        # CyRK accepts method as string (e.g., "RK45") or integer
        method_py = method  # Keep as string, CyRK handles conversion internally
        
        # Prepare pysolve_ivp arguments
        cyrk_kwargs = {
            "py_diffeq": fun,
            "time_span": t_span,
            "y0": y0,
            "method": method_py,
            "rtol": rtol_py,
            "atol": atol_py,
            "max_step": max_step,
        }
        
        # Add events if provided
        if events is not None:
            cyrk_kwargs["events"] = events
        
        # Add any additional kwargs (e.g., args for fun)
        cyrk_kwargs.update(kwargs)
        
        # Call pysolve_ivp with error handling
        try:
            cyrk_sol = pysolve_ivp(**cyrk_kwargs)
        except (ValueError, RuntimeError) as e:
            raise SolverError(f"CyRK solver failed: {e}", solver_name="cyrk", original_error=e) from e
        except Exception as e:
            raise SolverError(f"Unexpected error in CyRK solver: {e}", solver_name="cyrk", original_error=e) from e
        
        # Extract solution arrays
        t = cyrk_sol.t
        y = cyrk_sol.y
        
        # Check if integration was successful
        success = getattr(cyrk_sol, 'success', True)
        if not success:
            message = getattr(cyrk_sol, 'message', 'Integration failed')
            raise SolverError(f"CyRK solver integration failed: {message}", solver_name="cyrk")
        
        # Handle shape: CyRK might return (n,) or (n, m), ensure (n, m)
        if y.ndim == 1:
            y = y.reshape((y.size, 1))
        # Ensure shape is (n, m) where n is state dimension
        if y.shape[0] != y0.size:
            y = y.T if y.shape[1] == y0.size else y
        
        # Get nfev if available
        nfev = getattr(cyrk_sol, 'nfev', None)
        
        # Get success and message
        success = getattr(cyrk_sol, 'success', True)
        message = getattr(cyrk_sol, 'message', '')
        
        # Extract event information
        t_events = getattr(cyrk_sol, 't_events', None)
        y_events = getattr(cyrk_sol, 'y_events', None)
        
        # Create dense output function if requested
        sol = None
        if dense_output and len(t) > 1:
            # Create interpolation function similar to SciPy's OdeSolution
            def dense_sol(t_eval):
                """Dense output via cubic interpolation."""
                t_eval = np.asarray(t_eval)
                if t_eval.ndim == 0:
                    t_eval = t_eval.reshape(1)
                # Use scipy's interp1d for interpolation
                # Use cubic for smoother curves (requires at least 4 points)
                # Fall back to linear if insufficient points
                if len(t) >= 4:
                    interp = interp1d(t, y, kind='cubic', axis=1, 
                                     bounds_error=False, fill_value='extrapolate')
                else:
                    interp = interp1d(t, y, kind='linear', axis=1, 
                                     bounds_error=False, fill_value='extrapolate')
                result = interp(t_eval)
                # Ensure correct shape: (n, m) where n is state dim, m is eval points
                if result.ndim == 1:
                    result = result.reshape((result.size, 1))
                return result
            
            sol = dense_sol
        
        # Convert to unified Solution object
        return Solution(
            t=t,
            y=y,
            nfev=nfev if nfev is not None else 0,
            success=success,
            message=message,
            sol=sol,
            t_events=t_events,
            y_events=y_events,
        )

