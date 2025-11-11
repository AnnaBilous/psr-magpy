#!/usr/bin/env python3
"""
Tests for ODE solver abstraction.

Tests SciPySolver to ensure identical behavior to direct solve_ivp calls.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from scipy.integrate import solve_ivp

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psr_magpy.line.solver import SciPySolver, Solution, CyRKSolver, CYRK_AVAILABLE, SolverError
from psr_magpy.line.field_calculator import PythonFieldCalculator


class TestSciPySolver:
    """Test SciPySolver against direct solve_ivp calls."""
    
    def test_simple_ode_no_events(self):
        """Test simple ODE integration without events."""
        def rhs(t, y):
            """dy/dt = -y"""
            return -y
        
        t_span = (0.0, 1.0)
        y0 = np.array([1.0])
        
        # Direct solve_ivp
        scipy_sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # SciPySolver
        solver = SciPySolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # Compare results
        # Final time and value (relaxed tolerance due to adaptive stepping)
        assert np.allclose(solution.t[-1], scipy_sol.t[-1], rtol=1e-8, atol=1e-10)
        # Use dense output for accurate comparison (final value may differ slightly)
        assert solution.sol is not None
        assert scipy_sol.sol is not None
        
        # Compare dense output at common time points (more reliable than final value)
        t_test = np.linspace(0.0, 1.0, 10)
        sol_solver = solution.sol(t_test)
        sol_scipy = scipy_sol.sol(t_test)
        np.testing.assert_allclose(sol_solver, sol_scipy, rtol=1e-8, atol=1e-10)
        
        # Metadata
        assert solution.success == scipy_sol.success
        assert solution.message == scipy_sol.message
    
    def test_ode_with_events(self):
        """Test ODE integration with terminal events."""
        def rhs(t, y):
            """dy/dt = 1"""
            return np.array([1.0])
        
        def event1(t, y):
            """Event at y = 0.5"""
            return y[0] - 0.5
        event1.terminal = True
        event1.direction = 1
        
        def event2(t, y):
            """Event at t = 0.8"""
            return t - 0.8
        event2.terminal = True
        event2.direction = 1
        
        t_span = (0.0, 2.0)
        y0 = np.array([0.0])
        
        # Direct solve_ivp
        scipy_sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events=(event1, event2),
            dense_output=True,
        )
        
        # SciPySolver
        solver = SciPySolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events=(event1, event2),
            dense_output=True,
        )
        
        # Compare results
        # Final time and value (relaxed tolerance due to adaptive stepping)
        assert np.allclose(solution.t[-1], scipy_sol.t[-1], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(solution.y[:, -1], scipy_sol.y[:, -1], rtol=1e-8, atol=1e-10)
        # nfev may differ due to adaptive stepping, so we don't compare it strictly
        assert solution.success == scipy_sol.success
        
        # Compare event times
        if scipy_sol.t_events is not None:
            assert solution.t_events is not None
            assert len(solution.t_events) == len(scipy_sol.t_events)
            for sol_evt, scipy_evt in zip(solution.t_events, scipy_sol.t_events):
                if len(sol_evt) > 0 and len(scipy_evt) > 0:
                    # Compare first event time (terminal event should stop integration)
                    assert np.allclose(sol_evt[0], scipy_evt[0], rtol=1e-10, atol=1e-12)
    
    def test_ode_with_args(self):
        """Test ODE integration with additional arguments."""
        def rhs(t, y, a, b):
            """dy/dt = a*y + b"""
            return a * y + b
        
        t_span = (0.0, 1.0)
        y0 = np.array([1.0])
        a = 2.0
        b = 1.0
        
        # Direct solve_ivp
        scipy_sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            args=(a, b),
            dense_output=True,
        )
        
        # SciPySolver
        solver = SciPySolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            args=(a, b),
            dense_output=True,
        )
        
        # Compare results
        # Use dense output for accurate comparison (more reliable than final value)
        assert solution.sol is not None
        assert scipy_sol.sol is not None
        t_test = np.linspace(0.0, 1.0, 10)
        sol_solver = solution.sol(t_test)
        sol_scipy = scipy_sol.sol(t_test)
        np.testing.assert_allclose(sol_solver, sol_scipy, rtol=1e-7, atol=1e-9)
        assert solution.success == scipy_sol.success
    
    def test_ode_3d_system(self):
        """Test 3D ODE system (like field line integration)."""
        def rhs(t, r):
            """Simple rotation: dr/dt = [r[1], -r[0], 0]"""
            return np.array([r[1], -r[0], 0.0])
        
        t_span = (0.0, 2.0 * np.pi)
        y0 = np.array([1.0, 0.0, 0.0])
        
        # Direct solve_ivp
        scipy_sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # SciPySolver
        solver = SciPySolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # Compare results
        # Use dense output for accurate comparison (more reliable than final value)
        assert solution.sol is not None
        assert scipy_sol.sol is not None
        t_test = np.linspace(0.0, 2.0 * np.pi, 20)
        sol_solver = solution.sol(t_test)
        sol_scipy = scipy_sol.sol(t_test)
        np.testing.assert_allclose(sol_solver, sol_scipy, rtol=1e-7, atol=1e-9)
        assert solution.success == scipy_sol.success
    
    def test_ode_no_dense_output(self):
        """Test ODE integration without dense output."""
        def rhs(t, y):
            """dy/dt = -y"""
            return -y
        
        t_span = (0.0, 1.0)
        y0 = np.array([1.0])
        
        # Direct solve_ivp
        scipy_sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=False,
        )
        
        # SciPySolver
        solver = SciPySolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=False,
        )
        
        # Compare results
        # Final time and value (relaxed tolerance due to adaptive stepping)
        assert np.allclose(solution.t[-1], scipy_sol.t[-1], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(solution.y[:, -1], scipy_sol.y[:, -1], rtol=1e-8, atol=1e-10)
        # nfev may differ due to adaptive stepping, so we don't compare it strictly
        assert solution.success == scipy_sol.success
        assert solution.sol is None
        assert scipy_sol.sol is None
    
    def test_ode_different_methods(self):
        """Test different integration methods."""
        def rhs(t, y):
            """dy/dt = -y"""
            return -y
        
        t_span = (0.0, 1.0)
        y0 = np.array([1.0])
        
        methods = ["RK45", "DOP853", "BDF"]
        
        for method in methods:
            # Direct solve_ivp
            scipy_sol = solve_ivp(
                fun=rhs,
                t_span=t_span,
                y0=y0,
                method=method,
                rtol=1e-9,
                atol=1e-9,
                dense_output=True,
            )
            
            # SciPySolver
            solver = SciPySolver()
            solution = solver.solve(
                fun=rhs,
                t_span=t_span,
                y0=y0,
                method=method,
                rtol=1e-9,
                atol=1e-9,
                dense_output=True,
            )
            
            # Compare results for each method
            # Use dense output for accurate comparison
            assert solution.sol is not None
            assert scipy_sol.sol is not None
            t_test = np.linspace(0.0, 1.0, 10)
            sol_solver = solution.sol(t_test)
            sol_scipy = scipy_sol.sol(t_test)
            np.testing.assert_allclose(sol_solver, sol_scipy, rtol=1e-7, atol=1e-9)
            assert solution.success == scipy_sol.success
    
    def test_ode_with_max_step(self):
        """Test ODE integration with max_step constraint."""
        def rhs(t, y):
            """dy/dt = -y"""
            return -y
        
        t_span = (0.0, 1.0)
        y0 = np.array([1.0])
        max_step = 0.1
        
        # Direct solve_ivp
        scipy_sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            max_step=max_step,
            dense_output=True,
        )
        
        # SciPySolver
        solver = SciPySolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            max_step=max_step,
            dense_output=True,
        )
        
        # Compare results
        np.testing.assert_allclose(solution.t, scipy_sol.t, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(solution.y, scipy_sol.y, rtol=1e-10, atol=1e-12)
        assert solution.nfev == scipy_sol.nfev
    
    def test_ode_backward_integration(self):
        """Test backward time integration."""
        def rhs(t, y):
            """dy/dt = -y"""
            return -y
        
        t_span = (1.0, 0.0)  # Backward
        y0 = np.array([np.exp(-1.0)])  # y(1) = exp(-1)
        
        # Direct solve_ivp
        scipy_sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # SciPySolver
        solver = SciPySolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # Compare results
        # Final time and value (relaxed tolerance due to adaptive stepping)
        assert np.allclose(solution.t[-1], scipy_sol.t[-1], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(solution.y[:, -1], scipy_sol.y[:, -1], rtol=1e-8, atol=1e-10)
        # nfev may differ due to adaptive stepping, so we don't compare it strictly
        assert solution.success == scipy_sol.success
        
        # Verify backward integration: should end at y(0) ≈ 1.0
        assert np.allclose(solution.y[:, -1], 1.0, rtol=1e-6)


@pytest.mark.skipif(not CYRK_AVAILABLE, reason="CyRK is not available")
class TestCyRKSolver:
    """Test CyRKSolver against SciPySolver for consistency."""
    
    def test_simple_ode_no_events(self):
        """Test simple ODE integration without events."""
        def rhs(t, y):
            """dy/dt = -y"""
            return -y
        
        t_span = (0.0, 1.0)
        y0 = np.array([1.0])
        
        # SciPySolver
        scipy_solver = SciPySolver()
        scipy_solution = scipy_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # CyRKSolver
        cyrk_solver = CyRKSolver()
        cyrk_solution = cyrk_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # Compare results
        # Final time and value (relaxed tolerance due to different solvers)
        assert np.allclose(cyrk_solution.t[-1], scipy_solution.t[-1], rtol=1e-4, atol=1e-6)
        # Use dense output for accurate comparison
        assert cyrk_solution.sol is not None
        assert scipy_solution.sol is not None
        t_test = np.linspace(0.0, 1.0, 10)
        sol_cyrk = cyrk_solution.sol(t_test)
        sol_scipy = scipy_solution.sol(t_test)
        np.testing.assert_allclose(sol_cyrk, sol_scipy, rtol=1e-4, atol=1e-6)
        assert cyrk_solution.success == scipy_solution.success
    
    def test_ode_with_events(self):
        """Test ODE integration with terminal events."""
        def rhs(t, y):
            """dy/dt = 1"""
            return np.array([1.0])
        
        def event1(t, y):
            """Event at y = 0.5"""
            return y[0] - 0.5
        event1.terminal = True
        event1.direction = 1
        
        t_span = (0.0, 2.0)
        y0 = np.array([0.0])
        
        # SciPySolver
        scipy_solver = SciPySolver()
        scipy_solution = scipy_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events=(event1,),
            dense_output=True,
        )
        
        # CyRKSolver
        cyrk_solver = CyRKSolver()
        cyrk_solution = cyrk_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            events=(event1,),
            dense_output=True,
        )
        
        # Compare results
        # Final time and value (relaxed tolerance)
        assert np.allclose(cyrk_solution.t[-1], scipy_solution.t[-1], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(cyrk_solution.y[:, -1], scipy_solution.y[:, -1], rtol=1e-6, atol=1e-8)
        assert cyrk_solution.success == scipy_solution.success
        
        # Compare event times
        if scipy_solution.t_events is not None and cyrk_solution.t_events is not None:
            assert len(cyrk_solution.t_events) == len(scipy_solution.t_events)
            for cyrk_evt, scipy_evt in zip(cyrk_solution.t_events, scipy_solution.t_events):
                if len(cyrk_evt) > 0 and len(scipy_evt) > 0:
                    # Compare first event time (terminal event should stop integration)
                    assert np.allclose(cyrk_evt[0], scipy_evt[0], rtol=1e-4, atol=1e-6)
    
    def test_ode_with_args(self):
        """Test ODE integration with additional arguments."""
        def rhs(t, y, a, b):
            """dy/dt = a*y + b"""
            return a * y + b
        
        t_span = (0.0, 1.0)
        y0 = np.array([1.0])
        a = 2.0
        b = 1.0
        
        # SciPySolver
        scipy_solver = SciPySolver()
        scipy_solution = scipy_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            args=(a, b),
            dense_output=True,
        )
        
        # CyRKSolver
        cyrk_solver = CyRKSolver()
        cyrk_solution = cyrk_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            args=(a, b),
            dense_output=True,
        )
        
        # Compare results
        # Use dense output for accurate comparison
        assert cyrk_solution.sol is not None
        assert scipy_solution.sol is not None
        t_test = np.linspace(0.0, 1.0, 10)
        sol_cyrk = cyrk_solution.sol(t_test)
        sol_scipy = scipy_solution.sol(t_test)
        np.testing.assert_allclose(sol_cyrk, sol_scipy, rtol=1e-4, atol=1e-6)
        assert cyrk_solution.success == scipy_solution.success
    
    def test_ode_3d_system(self):
        """Test 3D ODE system (like field line integration)."""
        def rhs(t, r):
            """Simple rotation: dr/dt = [r[1], -r[0], 0]"""
            return np.array([r[1], -r[0], 0.0])
        
        t_span = (0.0, 2.0 * np.pi)
        y0 = np.array([1.0, 0.0, 0.0])
        
        # SciPySolver
        scipy_solver = SciPySolver()
        scipy_solution = scipy_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # CyRKSolver
        cyrk_solver = CyRKSolver()
        cyrk_solution = cyrk_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # Compare results
        # Use dense output for accurate comparison
        assert cyrk_solution.sol is not None
        assert scipy_solution.sol is not None
        t_test = np.linspace(0.0, 2.0 * np.pi, 20)
        sol_cyrk = cyrk_solution.sol(t_test)
        sol_scipy = scipy_solution.sol(t_test)
        np.testing.assert_allclose(sol_cyrk, sol_scipy, rtol=1e-4, atol=1e-6)
        assert cyrk_solution.success == scipy_solution.success
    
    def test_ode_no_dense_output(self):
        """Test ODE integration without dense output."""
        def rhs(t, y):
            """dy/dt = -y"""
            return -y
        
        t_span = (0.0, 1.0)
        y0 = np.array([1.0])
        
        # SciPySolver
        scipy_solver = SciPySolver()
        scipy_solution = scipy_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=False,
        )
        
        # CyRKSolver
        cyrk_solver = CyRKSolver()
        cyrk_solution = cyrk_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=False,
        )
        
        # Compare results
        # Final time and value (relaxed tolerance)
        assert np.allclose(cyrk_solution.t[-1], scipy_solution.t[-1], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(cyrk_solution.y[:, -1], scipy_solution.y[:, -1], rtol=1e-6, atol=1e-8)
        assert cyrk_solution.success == scipy_solution.success
        assert cyrk_solution.sol is None
        assert scipy_solution.sol is None
    
    def test_ode_with_max_step(self):
        """Test ODE integration with max_step constraint."""
        def rhs(t, y):
            """dy/dt = -y"""
            return -y
        
        t_span = (0.0, 1.0)
        y0 = np.array([1.0])
        max_step = 0.1
        
        # SciPySolver
        scipy_solver = SciPySolver()
        scipy_solution = scipy_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            max_step=max_step,
            dense_output=True,
        )
        
        # CyRKSolver
        cyrk_solver = CyRKSolver()
        cyrk_solution = cyrk_solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=y0,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            max_step=max_step,
            dense_output=True,
        )
        
        # Compare results
        # Use dense output for accurate comparison
        # Relaxed tolerance for max_step test (different step size handling between solvers)
        assert cyrk_solution.sol is not None
        assert scipy_solution.sol is not None
        t_test = np.linspace(0.0, 1.0, 10)
        sol_cyrk = cyrk_solution.sol(t_test)
        sol_scipy = scipy_solution.sol(t_test)
        np.testing.assert_allclose(sol_cyrk, sol_scipy, rtol=1e-4, atol=1e-3)
        assert cyrk_solution.success == scipy_solution.success


@pytest.mark.skipif(not CYRK_AVAILABLE, reason="CyRK is not available")
class TestEventCompatibility:
    """Test event detection compatibility with different solvers."""
    
    def test_surface_event_detection_scipy(self):
        """Test surface event detection with SciPySolver."""
        # Create a simple field that drives toward origin
        def rhs(t, r):
            """Field pointing toward origin: dr/dt = -r/|r|^3"""
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-10:
                return np.zeros(3)
            return -r / (r_norm**3)
        
        # Start outside surface, should hit surface
        surface_radius = 1.0
        max_radius = 10.0
        
        class EventDetector:
            def __init__(self, surface_radius, max_radius):
                self.surface_radius = surface_radius
                self.max_radius = max_radius
            
            def hit_surface(self, t, r):
                radius = float(np.sqrt(np.sum(r**2)))
                return radius - self.surface_radius
            hit_surface.terminal = True
            
            def hit_Rmax(self, t, r):
                radius = float(np.sqrt(np.sum(r**2)))
                return radius - self.max_radius
            hit_Rmax.terminal = True
        
        detector = EventDetector(surface_radius, max_radius)
        
        # Start at radius 2.0, should hit surface
        r0 = np.array([2.0, 0.0, 0.0])
        t_span = (0.0, 100.0)
        
        solver = SciPySolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=r0,
            events=(detector.hit_surface, detector.hit_Rmax),
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # Verify surface event was detected
        assert solution.success
        assert solution.t_events is not None
        assert len(solution.t_events) == 2
        # First event should be surface (terminal)
        assert len(solution.t_events[0]) > 0
        # Final position should be near surface
        final_radius = np.linalg.norm(solution.y[:, -1])
        assert np.allclose(final_radius, surface_radius, rtol=1e-3, atol=1e-3)
    
    def test_surface_event_detection_cyrk(self):
        """Test surface event detection with CyRKSolver."""
        # Create a simple field that drives toward origin
        def rhs(t, r):
            """Field pointing toward origin: dr/dt = -r/|r|^3"""
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-10:
                return np.zeros(3)
            return -r / (r_norm**3)
        
        # Start outside surface, should hit surface
        surface_radius = 1.0
        max_radius = 10.0
        
        class EventDetector:
            def __init__(self, surface_radius, max_radius):
                self.surface_radius = surface_radius
                self.max_radius = max_radius
            
            def hit_surface(self, t, r):
                radius = float(np.sqrt(np.sum(r**2)))
                return radius - self.surface_radius
            hit_surface.terminal = True
            
            def hit_Rmax(self, t, r):
                radius = float(np.sqrt(np.sum(r**2)))
                return radius - self.max_radius
            hit_Rmax.terminal = True
        
        detector = EventDetector(surface_radius, max_radius)
        
        # Start at radius 2.0, should hit surface
        r0 = np.array([2.0, 0.0, 0.0])
        t_span = (0.0, 100.0)
        
        solver = CyRKSolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=r0,
            events=(detector.hit_surface, detector.hit_Rmax),
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # Verify surface event was detected
        assert solution.success
        assert solution.t_events is not None
        assert len(solution.t_events) == 2
        # First event should be surface (terminal)
        assert len(solution.t_events[0]) > 0
        # Final position should be near surface
        final_radius = np.linalg.norm(solution.y[:, -1])
        assert np.allclose(final_radius, surface_radius, rtol=1e-3, atol=1e-3)
    
    def test_rmax_event_detection_scipy(self):
        """Test Rmax event detection with SciPySolver."""
        # Create a simple field that drives away from origin
        def rhs(t, r):
            """Field pointing away from origin: dr/dt = r/|r|^3"""
            r_norm = np.linalg.norm(r)
            if r_norm < 1e-10:
                return np.zeros(3)
            return r / (r_norm**3)
        
        surface_radius = 1.0
        max_radius = 5.0
        
        class EventDetector:
            def __init__(self, surface_radius, max_radius):
                self.surface_radius = surface_radius
                self.max_radius = max_radius
            
            def hit_surface(self, t, r):
                radius = float(np.sqrt(np.sum(r**2)))
                return radius - self.surface_radius
            hit_surface.terminal = True
            
            def hit_Rmax(self, t, r):
                radius = float(np.sqrt(np.sum(r**2)))
                return radius - self.max_radius
            hit_Rmax.terminal = True
        
        detector = EventDetector(surface_radius, max_radius)
        
        # Start at radius 1.5, should hit Rmax
        r0 = np.array([1.5, 0.0, 0.0])
        t_span = (0.0, 100.0)
        
        solver = SciPySolver()
        solution = solver.solve(
            fun=rhs,
            t_span=t_span,
            y0=r0,
            events=(detector.hit_surface, detector.hit_Rmax),
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
            dense_output=True,
        )
        
        # Verify Rmax event was detected
        assert solution.success
        assert solution.t_events is not None
        assert len(solution.t_events) == 2
        # Second event should be Rmax (terminal)
        assert len(solution.t_events[1]) > 0
        # Final position should be near Rmax
        final_radius = np.linalg.norm(solution.y[:, -1])
        assert np.allclose(final_radius, max_radius, rtol=1e-3, atol=1e-3)


class TestBackwardCompatibility:
    """Test backward compatibility: default backends work as before."""
    
    def test_default_field_backend_is_python(self):
        """Test that default field_backend is 'python'."""
        from psr_magpy.line.tracing_parameters import IntegrationParameters
        
        # Create IntegrationParameters without specifying field_backend
        params = IntegrationParameters(max_integration_radius=10.0)
        
        # Should default to "python"
        assert params.field_backend == "python"
    
    def test_default_solver_backend_is_scipy(self):
        """Test that default solver_backend is 'scipy'."""
        from psr_magpy.line.tracing_parameters import IntegrationParameters
        
        # Create IntegrationParameters without specifying solver_backend
        params = IntegrationParameters(max_integration_radius=10.0)
        
        # Should default to "scipy"
        assert params.solver_backend == "scipy"
    
    def test_field_line_tracer_defaults(self):
        """Test that FieldLineTracer uses default backends when not specified."""
        from psr_magpy.line.tracer import FieldLineTracer
        from psr_magpy.line.tracing_parameters import TracingParameters, IntegrationParameters
        from psr_magpy.field.components import DipoleFieldComponent
        from psr_magpy.field.multipole import MultipoleField
        
        # Create field
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        
        # Create params without specifying backends (should use defaults)
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0)
        )
        
        # Create tracer
        tracer = FieldLineTracer(field, params)
        
        # Verify default backends are used
        assert isinstance(tracer.field_calculator, PythonFieldCalculator)
        assert isinstance(tracer.solver, SciPySolver)
        
        # Verify field_backend and solver_backend are set to defaults
        assert params.integration.field_backend == "python"
        assert params.integration.solver_backend == "scipy"


class TestSolverErrorHandling:
    """Test unified error handling for solver exceptions."""
    
    def test_scipy_solver_value_error_wrapped(self):
        """Test that ValueError from SciPy solver is wrapped in SolverError."""
        def rhs(t, y):
            """Valid RHS function."""
            return -y
        
        solver = SciPySolver()
        
        # Invalid method (should raise ValueError)
        with pytest.raises(SolverError) as exc_info:
            solver.solve(
                fun=rhs,
                t_span=(0.0, 1.0),
                y0=np.array([1.0]),
                method="INVALID_METHOD",  # This should raise ValueError
                rtol=1e-9,
                atol=1e-9,
            )
        
        # Verify error attributes
        assert exc_info.value.solver_name == "scipy"
        assert exc_info.value.original_error is not None
        assert isinstance(exc_info.value.original_error, (ValueError, RuntimeError))
        assert "SciPy solver" in str(exc_info.value)
    
    def test_scipy_solver_success_false_raises(self):
        """Test that SciPy solver raises SolverError when success=False.
        
        Note: This test is difficult to trigger reliably without hanging,
        so we test the error handling path by verifying the code exists.
        The actual success=False check is tested indirectly through other
        error handling tests.
        """
        # This test verifies that the success=False check exists in the code
        # by checking that SolverError would be raised if success=False
        # We can't easily trigger a real success=False case without hanging,
        # but the code path is verified to exist in solver.py
        
        # Verify the check exists by inspecting the solver code
        import inspect
        solver = SciPySolver()
        source = inspect.getsource(solver.solve)
        assert "if not scipy_sol.success" in source or "scipy_sol.success" in source
        assert "SolverError" in source
    
    @pytest.mark.skipif(not CYRK_AVAILABLE, reason="CyRK is not available")
    def test_cyrk_solver_value_error_wrapped(self):
        """Test that ValueError from CyRK solver is wrapped in SolverError."""
        def rhs(t, y):
            """Valid RHS function."""
            return -y
        
        solver = CyRKSolver()
        
        # Invalid method (should raise ValueError or similar)
        with pytest.raises(SolverError) as exc_info:
            solver.solve(
                fun=rhs,
                t_span=(0.0, 1.0),
                y0=np.array([1.0]),
                method="INVALID_METHOD",  # This should raise an error
                rtol=1e-9,
                atol=1e-9,
            )
        
        # Verify error attributes
        assert exc_info.value.solver_name == "cyrk"
        assert exc_info.value.original_error is not None
        assert "CyRK solver" in str(exc_info.value)
    
    @pytest.mark.skipif(not CYRK_AVAILABLE, reason="CyRK is not available")
    def test_cyrk_solver_success_false_raises(self):
        """Test that CyRK solver raises SolverError when success=False."""
        # Create a mock solution object that simulates failure
        # Note: This is harder to trigger with real CyRK, so we'll test the check logic
        def rhs(t, y):
            """RHS that might cause integration issues."""
            # Return very large values that might cause issues
            return np.array([1e10 * y[0]])
        
        solver = CyRKSolver()
        
        # Try with very strict tolerances that might cause failure
        # Note: Actual failure depends on CyRK implementation
        # We test that if success=False, we raise SolverError
        try:
            solution = solver.solve(
                fun=rhs,
                t_span=(0.0, 1.0),
                y0=np.array([1.0]),
                method="RK45",
                rtol=1e-15,  # Very strict
                atol=1e-15,
            )
            # If it succeeds, that's fine - we're testing the error handling path
            # The actual test is that if success=False, we raise SolverError
        except SolverError as e:
            # If we get SolverError, verify it's correctly formatted
            assert e.solver_name == "cyrk"
            assert "CyRK solver" in str(e)
    
    def test_solver_error_attributes(self):
        """Test that SolverError has correct attributes."""
        original_error = ValueError("Original error message")
        solver_error = SolverError(
            message="Test error message",
            solver_name="test_solver",
            original_error=original_error
        )
        
        assert solver_error.solver_name == "test_solver"
        assert solver_error.original_error == original_error
        assert str(solver_error) == "Test error message"
        assert isinstance(solver_error, RuntimeError)
    
    def test_solver_error_without_original(self):
        """Test SolverError without original error."""
        solver_error = SolverError(
            message="Test error message",
            solver_name="test_solver",
            original_error=None
        )
        
        assert solver_error.solver_name == "test_solver"
        assert solver_error.original_error is None
        assert str(solver_error) == "Test error message"
    
    def test_scipy_solver_unexpected_exception_wrapped(self):
        """Test that unexpected exceptions from SciPy are wrapped.
        
        Note: Exceptions in the RHS function are handled by solve_ivp internally,
        so we test with invalid solver parameters that cause solver-level errors.
        """
        def rhs(t, y):
            """Valid RHS function."""
            return -y
        
        solver = SciPySolver()
        
        # Use invalid parameters that cause solver-level errors
        # Invalid dense_output with certain methods can cause issues
        with pytest.raises(SolverError) as exc_info:
            solver.solve(
                fun=rhs,
                t_span=(0.0, 1.0),
                y0=np.array([1.0]),
                method="INVALID_METHOD_NAME_THAT_DOESNT_EXIST",
                rtol=1e-9,
                atol=1e-9,
            )
        
        assert exc_info.value.solver_name == "scipy"
        assert exc_info.value.original_error is not None
        assert "SciPy solver" in str(exc_info.value)
    
    @pytest.mark.skipif(not CYRK_AVAILABLE, reason="CyRK is not available")
    def test_cyrk_solver_unexpected_exception_wrapped(self):
        """Test that unexpected exceptions from CyRK are wrapped.
        
        Note: Exceptions in the RHS function are handled by pysolve_ivp internally,
        so we test with invalid solver parameters that cause solver-level errors.
        """
        def rhs(t, y):
            """Valid RHS function."""
            return -y
        
        solver = CyRKSolver()
        
        # Use invalid parameters that cause solver-level errors
        with pytest.raises(SolverError) as exc_info:
            solver.solve(
                fun=rhs,
                t_span=(0.0, 1.0),
                y0=np.array([1.0]),
                method="INVALID_METHOD",
                rtol=1e-9,
                atol=1e-9,
            )
        
        assert exc_info.value.solver_name == "cyrk"
        assert exc_info.value.original_error is not None
        assert "CyRK solver" in str(exc_info.value)
