"""Field line tracing module for magnetic field analysis.

Provides classes for tracing magnetic field lines from neutron star surfaces
using ODE integration. Supports different integration methods, event detection,
and property calculation for field line analysis.

Classes:
    FieldLineTracer: Main tracer class for field line calculations

Example:
    >>> from psr_magpy.line.tracer import FieldLineTracer
    >>> from psr_magpy.line.parameters import TracingParameters
    >>> params = TracingParameters(R_max=10.0, min_R_max_line=1.0)
    >>> tracer = FieldLineTracer(field, params)
    >>> field_line = tracer.trace_single_line(r_initial, footpoint_id, template)
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from astropy.coordinates.representation import CartesianRepresentation, SphericalRepresentation

from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.property_system import PropertyName
from psr_magpy.line.tracing_parameters import TracingParameters, PhysicsParameters
from psr_magpy.line.property_calculator import PropertyCalculator
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class FieldLineTracer:
    """Main field line tracer for magnetic field analysis.
    
    Orchestrates the field line tracing process including ODE integration,
    event detection, and property calculation. Provides a high-level interface
    for tracing field lines from neutron star surfaces.
    
    Attributes:
        field: Object that calculates the magnetic field
        params: Unified tracing parameters
        property_calculator: Handler for property calculations
    """
    
    class _EventDetector:
        """Handles event detection for field line integration.
        
        Manages event functions used by scipy.integrate.solve_ivp to detect
        when field lines hit the neutron star surface or maximum radius.
        
        Attributes:
            surface_radius: Radius of neutron star surface
            R_max: Maximum radius to trace field lines to
        """
        
        def __init__(self, surface_radius: float, R_max: float):
            """Initialize event detector.
            
            Parameters
            ----------
            surface_radius : float
                Radius of neutron star surface
            R_max : float
                Maximum radius to trace field lines to
            """
            self.surface_radius = surface_radius
            self.R_max = R_max
        
        def hit_surface(self, t: float, r: np.ndarray, K: float) -> float:
            """Event function for hitting neutron star surface.
            
            Parameters
            ----------
            t : float
                Current integration time (unused)
            r : np.ndarray
                Current position vector
            K : float
                Field direction multiplier (unused)
                
            Returns
            -------
            float
                Distance from surface (positive = outside, negative = inside)
            """
            radius = float(np.sqrt(np.sum(r**2)))
            return radius - self.surface_radius

        hit_surface.terminal = True

        def hit_Rmax(self, t: float, r: np.ndarray, K: float) -> float:
            """Event function for hitting maximum radius.
            
            Parameters
            ----------
            t : float
                Current integration time (unused)
            r : np.ndarray
                Current position vector
            K : float
                Field direction multiplier (unused)
                
            Returns
            -------
            float
                Distance from maximum radius (positive = beyond, negative = within)
            """
            radius = float(np.sqrt(np.sum(r**2)))
            return radius - self.R_max

        hit_Rmax.terminal = True
    
    def __init__(self, field: Any, params: TracingParameters) -> None:
        """Initialize the field line tracer.

        Parameters
        ----------
        field : MultipoleField
            Object that calculates the magnetic field
        params : TracingParameters
            Unified tracing parameters containing all configuration
        """
        logger.info(f"Initializing FieldLineTracer with R_max={params.R_max}, min_R_max_line={params.min_R_max_line}")
        
        # Store parameters
        self.field = field
        self.params = params
        
        # Initialize event detector
        self.event_detector = self._EventDetector(
            surface_radius=params.physics.surface_radius,
            R_max=params.physics.surface_radius  # Will be set during integration
        )
        
        # Initialize property calculator
        self.property_calculator = PropertyCalculator(params.min_R_max_line)
        
        logger.debug(f"Field object: {field}")
        logger.debug(f"Tracing parameters: {params}")

    def _field_for_integration(self, t: float, r: np.ndarray, K: float) -> np.ndarray:
        """Wrapper for field calculation that includes unused time parameter required by solve_ivp.

        Parameters
        ----------
        t : float
            Time parameter (unused, required by solve_ivp)
        r : ndarray
            Position vector
        K : float
            Field direction multiplier

        Returns
        -------
        ndarray
            Magnetic field vector
        """
        return self.field.get_total_field(r, K, normalize=True)
    
    def integrate_field_line(self, r_initial: np.ndarray, K: float, R_max: float) -> Any:
        """Integrate field line from initial position.
        
        Parameters
        ----------
        r_initial : np.ndarray
            Initial position vector
        K : float
            Field direction multiplier
        R_max : float
            Maximum radius to trace to
            
        Returns
        -------
        solve_ivp result
            Integration result from scipy.integrate.solve_ivp
        """
        # Update event detector with current R_max
        self.event_detector.R_max = R_max
        
        logger.debug(f"Integrating field line from {r_initial} with K={K}, R_max={R_max}")
        
        return solve_ivp(
            self._field_for_integration,
            self.params.physics.t_span,
            r_initial,
            args=[K],
            dense_output=True,
            events=(self.event_detector.hit_surface, self.event_detector.hit_Rmax),
            rtol=self.params.physics.rtol,
            atol=self.params.physics.atol,
        )

    def trace_single_line(
        self,
        r_initial: CartesianRepresentation,
        footpoint_id: int,
        template: FieldLine,
        save_dR: bool = False,
    ) -> FieldLine:
        """Trace a single field line from the given initial position.

        Parameters
        ----------
        r_initial : CartesianRepresentation
            Initial position for field line tracing
        footpoint_id : int
            ID of the footpoint
        template : FieldLine
            Template specifying which properties to save
        save_dR : bool, optional
            Whether to save the median step size, default False

        Returns
        -------
        FieldLine
            FieldLine object containing requested variables for the field line
        """
        output = template.copy_structure()
        output.set_property(footpoint_id=int(footpoint_id))
        r_initial_array = r_initial.xyz.value
        B_initial = self.field.get_total_field(r_initial_array, K=1, normalize=False)
        B_initial_norm = self.field.get_total_field(r_initial_array, K=1, normalize=True)
        dotprod = np.dot(B_initial_norm, r_initial_array)
        K = np.sign(dotprod)
        if K == 0:
            K = 1.
        
        # Use internal integration method
        field_line = self.integrate_field_line(r_initial_array, K, self.params.R_max)

        # Use dt from integration parameters
        dt = self.params.physics.dt
        if isinstance(dt, int) or dt > 1:
            t = np.linspace(field_line.t[0], field_line.t[-1], int(dt))
        else:
            t = np.arange(field_line.t[0], field_line.t[-1] + dt, dt)

        r = field_line.sol(t)
       
        # Use property calculator
        props = self.property_calculator.calculate_line_properties(
            footpoint_id, r, B_initial, dotprod, K, template
        )
      
        output.set_property(**props)

        if save_dR:
            dr = r[:, 1:] - r[:, :-1]
            dR = np.median(np.sum(dr**2, axis=0) ** 0.5)
            # Note: dR is not a standard property, so we don't set it
            logger.debug(f"Calculated dR: {dR}")
        return output

    

