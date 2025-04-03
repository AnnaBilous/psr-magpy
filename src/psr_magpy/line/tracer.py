import logging

import numpy as np
from scipy.integrate import solve_ivp

from astropy.coordinates.representation import CartesianRepresentation

from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)

class FieldLineTracer:
    def __init__(self, field, R_max=10.0, R_min=1.0, dt=0.01):
        """Initialize the field line tracer.

        Parameters
        ----------
        field : MultipoleField
            Object that calculates the magnetic field
        R_max : float
            Maximum radius to trace field lines to (in NS radii)
        R_min : float
            Minimum radius to trace field lines to (in NS radii)
        """
        self.field = field
        self.R_max = R_max
        self.R_min = R_min
        self.dt = dt
        
        logger.info(f"Initialized FieldLineTracer with R_max={R_max}, R_min={R_min}, dt={dt}")
        logger.debug(f"Field object: {field}")

    def hit_surface(self, t, r, K):
        r = np.sum(r**2) ** 0.5
        return r - 0.999999

    hit_surface.terminal = True

    def hit_Rmax(self, t, r, K):
        r = np.sum(r**2) ** 0.5
        return r - self.R_max

    hit_Rmax.terminal = True

    def hit_Rmin(self, t, r, K):
        r = np.sum(r**2) ** 0.5
        return r - self.R_min

    hit_Rmin.terminal = True

    def _field_for_integration(self, t, r, K):
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

    def trace_single_line(
        self,
        r_initial: CartesianRepresentation,
        footpoint_id: int,
        template: FieldLine,
        save_dR=False,
    ) -> FieldLine:
        """Trace a single field line from the given initial position.

        Parameters
        ----------
        r_initial : CartesianRepresentation
            Initial position for field line tracing
        template : FieldLine
            Template specifying which properties to save
        dt : float
            Integration time step
        save_dR : bool
            Whether to save the median step size

        Returns
        -------
        FieldLine
            FieldLine object containing requested variables for the field line
        """
        output = template.copy_structure()
        output.set_property(footpoint_id=footpoint_id)
        r_initial = r_initial.xyz.value
        B_initial = self.field.get_total_field(r_initial, K=1, normalize=False)
        B_initial_norm = self.field.get_total_field(r_initial, K=1, normalize=True)
        dotprod = np.dot(B_initial_norm, r_initial)
        K = np.sign(dotprod)
        if K == 0:
            K = 1.0
        # Include hit_Rmin event to ensure we don't trace too far down?
        field_line = solve_ivp(
            self._field_for_integration,
            [0, 1e8],
            r_initial,
            args=[K],
            dense_output=True,
            events=(self.hit_surface, self.hit_Rmax),
            rtol=5.0e-8,
            atol=5.0e-8,
        )

        if isinstance(self.dt, int):
            t = np.linspace(field_line.t[0], field_line.t[-1] + self.dt, self.dt)
        else:
            t = np.arange(field_line.t[0], field_line.t[-1] + self.dt, self.dt)

        r = field_line.sol(t)
        R_max = np.max(np.sum(r**2, axis=0) ** 0.5)

        if template.save_flags.save_r_line:
            if R_max >= self.R_min:
                r_line_arg = r.T
            else:
                r_line_arg = np.array([])  # Line traced but didn't reach R_min
        else:
            r_line_arg = None  # Don't save line at all

        kwargs = {
            "footpoint_id": footpoint_id,
            "r_line": r_line_arg,
            "r_max": R_max if template.save_flags.save_r_max else None,
            "r_cyl_max": (
                np.max(np.sum(r[:2,] ** 2, axis=0) ** 0.5)
                if template.save_flags.save_r_cyl_max
                else None
            ),
            "z_abs_max": (
                np.max(np.abs(r[2, :])) if template.save_flags.save_z_abs_max else None
            ),
            "b_ini": (
                np.sum(B_initial**2) ** 0.5 if template.save_flags.save_b_ini else None
            ),
            "b_r_ini_angle": (
                np.arccos(dotprod) if template.save_flags.save_b_r_ini_angle else None
            ),
        }

        # Remove None values from kwargs
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        output.set_property(**kwargs)

        if save_dR:
            dr = r[:, 1:] - r[:, :-1]
            dR = np.median(np.sum(dr**2, axis=0) ** 0.5)
            output.dR = dR
        return output
