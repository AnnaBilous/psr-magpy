import numpy as np
from astropy.coordinates.representation import CartesianRepresentation
from scipy.integrate import solve_ivp
from typing import Dict
from Constants import R_NS

class FieldLineTracer:
    def __init__(self, field_calculator, R_max=10.0, R_min=1.0):
        """Initialize the field line tracer.
        
        Parameters
        ----------
        field_calculator : MultipoleField
            Object that calculates the magnetic field
        R_max : float
            Maximum radius to trace field lines to (in NS radii)
        R_min : float
            Minimum radius to trace field lines to (in NS radii)
        """
        self.field_calculator = field_calculator
        self.R_max = R_max
        self.R_min = R_min

    def hit_surface(self, t, r, K):
        r = np.sum(r**2) ** 0.5
        return r - 0.999999 * R_NS

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
        return self.field_calculator.get_total_field(r, K, normalize=True)

    def trace_single_line(
        self, r_initial: CartesianRepresentation, vars_to_save: Dict, dt=0.01, save_dR=False
    ) -> Dict:
        """Trace a single field line from the given initial position.
        
        Parameters
        ----------
        r_initial : CartesianRepresentation
            Initial position for field line tracing
        vars_to_save : Dict
            Dictionary specifying which variables to save
        dt : float
            Integration time step
        save_dR : bool
            Whether to save the median step size
            
        Returns
        -------
        Dict
            Dictionary containing requested variables for the field line
        """
        r_initial = r_initial.xyz.value
        B_initial = self.field_calculator.get_total_field(r_initial, K=1, normalize=False)
        B_initial_norm = self.field_calculator.get_total_field(r_initial, K=1, normalize=True)
        dotprod = np.dot(B_initial_norm, r_initial)
        K = np.sign(dotprod)
        if K==0: K = 1.0
        
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

        if isinstance(dt, int):
            t = np.linspace(field_line.t[0], field_line.t[-1] + dt, dt)
        else:
            t = np.arange(field_line.t[0], field_line.t[-1] + dt, dt)
        
        r = field_line.sol(t)
        R_max = np.max(np.sum(r**2, axis=0) ** 0.5)

        output_dict = {}
        if "B_R_ini_angle" in vars_to_save:
            output_dict["B_R_ini_angle"] = np.arccos(dotprod)
        if "R_max" in vars_to_save:
            output_dict["R_max"] = R_max
        if "R_cyl_max" in vars_to_save:
            output_dict["R_cyl_max"] = np.max(np.sum(r[:2,] ** 2, axis=0) ** 0.5)
        if "z_abs_max" in vars_to_save:
            output_dict["z_abs_max"] = np.max(np.abs(r[2, :]))
        if "B_ini" in vars_to_save:
            output_dict["B_ini"] = np.sum(B_initial**2) ** 0.5
        if "r_ini" in vars_to_save:
            output_dict["r_ini"] = r_initial
        if "field_lines" in vars_to_save:
            if R_max >= self.R_min:
                output_dict["field_lines"] = r.T
            else:
                output_dict["field_lines"] = []

        if save_dR:
            dr = r[:, 1:] - r[:, :-1]
            dR = np.median(np.sum(dr**2, axis=0) ** 0.5)
            output_dict["dR"] = dR
            
        return output_dict 