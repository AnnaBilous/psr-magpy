"""
Field line tracer for magnetic field integration.

Provides efficient field line tracing with configurable properties
and integration parameters.
"""

from typing import Any
from astropy.coordinates import CartesianRepresentation
import numpy as np

from psr_magpy.line.tracing_parameters import TracingParameters
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.property_calculator import PropertyCalculator
from psr_magpy.line.field_calculator import (
    FieldCalculator,
    PythonFieldCalculator,
    NumbaFieldCalculator,
)
from psr_magpy.line.solver import ODESolver, SciPySolver, CyRKSolver
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyTracingError

logger = get_logger(__name__, console_format_brief=True)


class FieldLineTracer:
    """Main field line tracer for magnetic field analysis."""


    class _EventDetector:
        """Handles event detection for field line integration."""

        def __init__(self, surface_radius: float, max_integration_radius: float):
            """Initialize event detector."""
            self.surface_radius = surface_radius
            self.max_integration_radius = max_integration_radius

        def hit_surface(self, t: float, r: np.ndarray, K: int) -> float:
            """Event function for hitting neutron star surface."""
            radius = float(np.sqrt(np.sum(r**2)))

            return radius - self.surface_radius

        hit_surface.terminal = True

        def hit_Rmax(self, t: float, r: np.ndarray, K: int) -> float:
            """Event function for hitting maximum radius."""
            radius = float(np.sqrt(np.sum(r**2)))
            return radius - self.max_integration_radius

        hit_Rmax.terminal = True

    
    def __init__(self, field: Any, params: TracingParameters) -> None:
        """Initialize the field line tracer."""
        logger.info(
            f"Initializing FieldLineTracer with max_integration_radius={params.integration.max_integration_radius}, threshold_line_distance={params.threshold_line_distance}"
        )

        # Store parameters
        self.field = field
        self.params = params

        # Initialize field calculator based on backend
        # Backend validation is handled by TracingParameters validators
        field_backend = params.integration.field_backend
        if field_backend == "python":
            self.field_calculator = PythonFieldCalculator(field)
        elif field_backend == "numba":
            self.field_calculator = NumbaFieldCalculator(field)

        # Initialize solver based on backend
        # Backend validation is handled by TracingParameters validators
        solver_backend = params.integration.solver_backend
        if solver_backend == "scipy":
            self.solver = SciPySolver()
        elif solver_backend == "numba":
            # TODO: Implement NumbaSolver
            raise NotImplementedError("Numba solver backend not yet implemented")
        elif solver_backend == "cyrk":
            self.solver = CyRKSolver()

        # Initialize event detector
        self.event_detector = self._EventDetector(
            surface_radius=params.integration.surface_radius,
            max_integration_radius=params.integration.max_integration_radius,
        )

        # Initialize property calculator
        self.property_calculator = PropertyCalculator(
            params.threshold_line_distance, field=field
        )

        logger.debug(f"Field object: {field}")
        logger.debug(f"Tracing parameters: {params}")

    def _setup_event_detector(
        self, surface_radius: float, max_integration_radius: float
    ) -> None:
        """Update event detector parameters."""
        self.event_detector.surface_radius = surface_radius
        self.event_detector.max_integration_radius = max_integration_radius

    def _field_for_integration(self, t: float, r: np.ndarray, K: int) -> np.ndarray:
        """Wrapper for field calculation that includes unused time parameter required by ODE solvers."""
        return self.field_calculator.get_field(r, K, normalize=True)
    
    def _field_for_integration_with_args(self, t: float, r: np.ndarray, *args) -> np.ndarray:
        """Wrapper that extracts K from args for solver interface."""
        K = args[0] if len(args) > 0 else 1
        return self._field_for_integration(t, r, K)

    def integrate_field_line(
        self,
        r_initial: np.ndarray,
        K: int,
        max_integration_radius: float,
        trace_open_lines_backwards: bool,
    ) -> Any:
        """Integrate field line from initial position."""
        # Determine if this is backward integration based on starting radius
        r_initial_radius = np.sqrt(np.sum(r_initial**2))
        is_open_line = r_initial_radius > 1.001
        if not (is_open_line):
            r_initial = r_initial / np.linalg.norm(r_initial)

        # Setup event detector parameters
        surface_radius = self.params.integration.surface_radius
        
        # Calculate dynamic t_span based on max_integration_radius * t_coeff / dt
        dynamic_t_span_end = max_integration_radius * self.params.integration.t_coeff / self.params.integration.dt
        
        if (trace_open_lines_backwards) and (is_open_line):
            t_span = (
                self.params.integration.t_span[0],
                -dynamic_t_span_end,
            )
            effective_max_integration_radius = max_integration_radius * 1.1
            max_step = self.params.integration.max_step
        else:
            t_span = (self.params.integration.t_span[0], dynamic_t_span_end)
            effective_max_integration_radius = max_integration_radius
            max_step = self.params.integration.max_step

        self._setup_event_detector(surface_radius, effective_max_integration_radius)

        rtol = self.params.integration.rtol
        atol = self.params.integration.atol

        logger.debug(
            f"Integrating field line from {r_initial} with K={K}, max_integration_radius={max_integration_radius}"
        )

        # Use solver abstraction
        solution = self.solver.solve(
            fun=self._field_for_integration_with_args,
            t_span=t_span,
            y0=r_initial,
            events=(self.event_detector.hit_surface, self.event_detector.hit_Rmax),
            method=self.params.integration.method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
            args=(K,),  # Pass K as additional argument
        )

        return solution

    def trace_single_line(
        self,
        r_initial: CartesianRepresentation,
        footpoint_id: int,
        template: FieldLine,
        save_dR: bool = False,
    ) -> FieldLine:
        """Trace a single field line from the given initial position."""
        r_initial_array = r_initial.xyz.value

        # Create output with automatic reversed_error flag if verification is enabled
        if self.params.integration.verify_by_reversing:
            # Get template flags and ensure reversed_error is enabled
            template_flags = template.property_system.get_flags_dict()
            template_flags["save_reversed_error"] = True
            output = FieldLine(**template_flags)
        else:
            output = template.copy_structure()

        output.set_property(footpoint_id=int(footpoint_id))

        B_initial = self.field.get_total_field(r_initial_array, K=1, normalize=False)
        B_initial_norm = self.field.get_total_field(
            r_initial_array, K=1, normalize=True
        )
        dotprod = np.dot(B_initial_norm, r_initial_array)
        K = int(np.sign(dotprod))
        if K == 0:
            K = 1

        # Use internal integration method
        field_line = self.integrate_field_line(
            r_initial_array,
            K,
            self.params.integration.max_integration_radius,
            self.params.integration.trace_open_lines_backwards,
        )

        # Handle different integration systems
     
        dt = self.params.integration.dt
        #print(field_line.t[0], field_line.t[-1])
        if isinstance(dt, int) or dt > 1:
            t = np.linspace(field_line.t[0], field_line.t[-1], int(dt))
        else:
            # Handle both positive and negative time spans
            t_start, t_end = field_line.t[0], field_line.t[-1]
            if t_end >= t_start:
                # Forward time integration
                t = np.append(np.arange(t_start, t_end, dt), t_end)
            else:
                # Backward time integration (t_end < t_start)
                t = np.append(np.arange(t_start, t_end, -dt), t_end)

        # Ensure we don't have empty time array
        if len(t) == 0:
            # Fallback to linspace with minimum 2 points
            t = np.linspace(field_line.t[0], field_line.t[-1], max(int(dt), 2))

        r = field_line.sol(t)

        # Log final radius for debugging
        final_radius = np.sqrt(np.sum(r[:, -1] ** 2))
        logger.debug(
            f"Field line final radius: {final_radius:.6f}, integration success: {field_line.success}"
        )

        # Use property calculator
        # For System B, calculate dt from the actual time points
        if self.params.integration.integration_system == "A":
            dt_for_props = dt
        else:  # System B
            # Calculate dt from actual time points
            if len(t) > 1:
                dt_for_props = np.mean(np.diff(t))
            else:
                dt_for_props = self.params.integration.dt
        
        props = self.property_calculator.calculate_line_properties(
            footpoint_id, r, B_initial, dotprod, K, dt_for_props, template
        )

        output.set_property(**props)

        if save_dR:
            dr = r[:, 1:] - r[:, :-1]
            dR = np.median(np.sum(dr**2, axis=0) ** 0.5)
            # Note: dR is not a standard property, so we don't set it
            logger.debug(f"Calculated dR: {dR}")

        # Perform verification if enabled
        if self.params.integration.verify_by_reversing:

            r_initial_array_reversed = r[:, -1]
            B_initial_reversed = self.field.get_total_field(
                r_initial_array_reversed, K=1, normalize=False
            )
            B_initial_reversed_norm = self.field.get_total_field(
                r_initial_array_reversed, K=1, normalize=True
            )
            dotprod = np.dot(B_initial_reversed_norm, r_initial_array_reversed)
            K_reversed = int(np.sign(dotprod))
            if K_reversed == 0:
                K_reversed = 1

            field_line_reversed = self.integrate_field_line(
                r_initial_array_reversed,
                K_reversed,
                self.params.integration.max_integration_radius,
                True,
            )

            if field_line_reversed.success:
                # Use actual integration endpoint, not interpolated value
                # This ensures accuracy regardless of interpolation method
                r_reversed_final = field_line_reversed.y[:, -1]
                # Calculate distance from original starting point
                distance_from_start = np.sqrt(
                    np.sum((r_reversed_final - r_initial_array) ** 2)
                )
                output.set_property(reversed_error=distance_from_start)
            else:
                # If reversed integration failed, set a large error
                output.set_property(reversed_error=float("inf"))

        return output
