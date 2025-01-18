import numpy as np
from astropy.coordinates.representation import CartesianRepresentation
from scipy.integrate import solve_ivp
from typing import Dict
from Constants import R_NS
import cProfile
import pstats
from pstats import SortKey

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from MultipoleField import MultipoleField
    from FootpointGrid import FootpointGrid, GridType, SphericalGridParams
    from FieldComponent import FieldComponent, MultipoleType
    
    # Create magnetic field (example with dipole)
    dipole = FieldComponent(moment_type=MultipoleType.QUADRUPOLE, B_magnitude=3.0)
    field = MultipoleField([dipole])  
    # Create profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Initialize field line tracer
    tracer = FieldLineTracer(field, R_max=5.0)
    
    # Create footpoints using FootpointGrid
    params_standard = SphericalGridParams(nlon=5, nlat=15)
    grid = FootpointGrid(grid_type=GridType.SPHERICAL, params=params_standard)
    footpoints = grid.footpoint_coordinates
    
    # Variables to save
    vars_to_save = {"field_lines": True}
    
    field_lines = []
    for footpoint in footpoints:
        result = tracer.trace_single_line(footpoint, vars_to_save)
        if "field_lines" in result and len(result["field_lines"]) > 0:
            field_lines.append(result["field_lines"])
    
    # Plot results
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot field lines
    for line in field_lines:
        ax.plot(line[:, 0], line[:, 1], line[:, 2], 'b-', alpha=0.5)
    
    # Plot NS surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='r', alpha=0.1)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    
    # Disable profiler and print stats
    profiler.disable()
    
    # Print stats sorted by cumulative time
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Show top 20 time-consuming functions
    
    # Print stats sorted by total time
    print("\nStats sorted by total time:")
    stats.sort_stats(SortKey.TIME).print_stats(20) 