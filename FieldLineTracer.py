import numpy as np
from astropy.coordinates.representation import CartesianRepresentation
from scipy.integrate import solve_ivp
from typing import Dict
from Constants import R_NS
import cProfile
import pstats
from pstats import SortKey
from multiprocessing import Pool
from FieldLineDefinition import FieldLine


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
        output.footpoint_id = footpoint_id
        r_initial = r_initial.xyz.value
        B_initial = self.field.get_total_field(r_initial, K=1, normalize=False)
        B_initial_norm = self.field.get_total_field(r_initial, K=1, normalize=True)
        dotprod = np.dot(B_initial_norm, r_initial)
        K = np.sign(dotprod)
        if K == 0:
            K = 1.0

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

 
        if template.save_b_r_ini_angle:
            output.b_r_ini_angle = np.arccos(dotprod)
        if template.save_r_max:
            output.r_max = R_max
        if template.save_r_cyl_max:
            output.r_cyl_max = np.max(np.sum(r[:2,] ** 2, axis=0) ** 0.5)
        if template.save_z_abs_max:
            output.z_abs_max = np.max(np.abs(r[2, :]))
        if template.save_b_ini:
            output.b_ini = np.sum(B_initial**2) ** 0.5
        if template.save_r_line:
            if R_max >= self.R_min:
                output.r_line = r.T
            else:
                output.r_line = np.empty((0, 3))

        if save_dR:
            dr = r[:, 1:] - r[:, :-1]
            dR = np.median(np.sum(dr**2, axis=0) ** 0.5)
            output.dR = dR
        return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from MultipoleField import MultipoleField
    from FootpointGrid import FootpointGrid, GridType, SphericalGridParams
    from FieldComponent import DipoleFieldComponent, QuadrupoleFieldComponent,MultipoleType
    from multiprocessing import Pool

    # Create magnetic field (example with dipole)
    dipole = DipoleFieldComponent(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=45.0,
    )
    quadrupole = FieldComponent(moment_type=MultipoleType.QUADRUPOLE, B_magnitude=3.0)
    field = MultipoleField([dipole, quadrupole])

    def trace_wrapper(footpoint, vars_to_save):
        """Wrapper function for parallel processing"""
        return tracer.trace_single_line(footpoint, vars_to_save)

    # Create profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Initialize field line tracer
    tracer = FieldLineTracer(field, R_max=5.0)

    # Create footpoints using FootpointGrid
    params_standard = SphericalGridParams(nlon=4, nlat=11)
    grid = FootpointGrid(grid_type=GridType.SPHERICAL, params=params_standard)
    footpoints = grid.footpoint_coordinates

    # Variables to save
    template = FieldLine(save_r_line=True)

    # Create list of arguments for each footpoint
    args = [(footpoint, template) for footpoint in footpoints]

    # Parallel processing of field lines
    with Pool() as pool:
        results = pool.starmap(trace_wrapper, args)

    # Filter and collect results
    field_lines = [
        a.r_line for a in results
    ]  # if a.r_line # is not None and len(a.r_line) > 0]

    # Plot results
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot field lines
    for line in field_lines:
        ax.plot(line[:, 0], line[:, 1], line[:, 2], "b-", alpha=0.5)

    # Plot NS surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="r", alpha=0.1)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

    # Disable profiler and print stats
    profiler.disable()

    # Print stats sorted by cumulative time
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(10)  # Show top 20 time-consuming functions

    # Print stats sorted by total time
    print("\nStats sorted by cumulative time:")
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(20)
