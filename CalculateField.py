import time
import datetime

import numpy as np
from astropy.coordinates.representation import CartesianRepresentation
from scipy.integrate import solve_ivp
from tqdm import tqdm

from FieldComponent import *
from FootpointGrid import *
from Constants import R_NS
import WriteToFits as io

class MultipoleField:
    def __init__(self, field_components: list):
        self.field_components = field_components

    def set_vars_to_save(self, vars: str):
        allowed_variables = set(
            (
                "R_max",
                "R_cyl_max",
                "z_abs_max",
                "B_ini",
                "B_R_ini_angle",
                "r_ini",
                "footpoint_id",
                "field_lines",
            )
        )
        if vars == "":
            vars_to_save = allowed_variables
        else:
            vars = set(vars.replace(" ", "").split(","))
            vars_to_save = vars & allowed_variables
            not_allowed_vars = vars - allowed_variables
            assert len(not_allowed_vars) == 0, print(
                f"Following variables are not recognized: {not_allowed_vars}.\nPlease use the following: {allowed_variables}"
            )
        vars_to_save_dict = {}
        vars_to_save_dict = {var: [] for var in vars_to_save}
        return vars_to_save_dict

    def get_total_field(self, t, r, K, normalize=True):
        F = np.array([0.0, 0.0, 0.0])
        for multipole in self.field_components:
            F += multipole.get_field(r)
        F = F * K
        if normalize:
            F /= np.sum(F**2) ** 0.5
        return F

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

    def get_field_lines(
        self,
        footpoints_grid,
        filename="",
        vars_to_save="",
        R_max=10.0,
        R_min=1.0,
        dt=0.01,
        save_file=True,
        max_file_size_Mb=200,
    ):
        """Calculate field lines starting from given footpoints.

        Parameters
        ----------
        footpoints_grid : FootpointGrid
            Grid of initial points for field line tracing
        filename : str, optional
            Base filename to save results, by default ""
        vars_to_save : str, optional
            Comma-separated list of variables to save, by default ""
        R_max : float, optional
            Maximum radius to trace field lines to (in NS radii), by default 10.0
        R_min : float, optional
            Minimum radius to trace field lines to (in NS radii), by default 1.0
        dt : float, optional
            Integration time step, by default 0.01
        save_file : bool, optional
            Whether to save results to file, by default True
        max_file_size_Mb : float, optional
            Maximum size of output files in MB, by default 200.0

        Returns
        -------
        dict
            Dictionary containing requested variables for each field line
        """
        self.R_min = R_min
        self.R_max = R_max
        if max_file_size_Mb < 0.02:
            print(
                "Max file size is smaller than the size of header. Setting it to 20 kB."
            )
            max_file_size_Mb = 0.02
        r_initial_grid = footpoints_grid.footpoint_coordinates
        grid_indices = footpoints_grid.indices
        vars_to_save_dict = self.set_vars_to_save(vars_to_save)

        completion_time_sec = self.estimate_computing_time(
            footpoints_grid, vars_to_save_dict
        )
        too_long_sec = 300.0
        if (completion_time_sec > too_long_sec) & (save_file == False):
            print(
                f"Warning: calculations will take {completion_time_sec/60.} min. According to input parameters, files will not be saved to disk. Are you sure?"
            )

        save_field_lines = "field_lines" in vars_to_save_dict.keys()

        vars_to_save_size_increment_kB = 0.06
        vars_to_save_size_kB = 12  # Size of header
        field_lines_size_kB = 0
        file_no = 0
        filename_cur = filename + "_%03d.fits" % file_no

        print(f"Starting field line calculations at {datetime.datetime.now()}")
        for i in tqdm(np.arange(footpoints_grid.nline)):
            vars_to_save_size_kB += vars_to_save_size_increment_kB
            r_initial = r_initial_grid[i]
            stat_dict = self.get_single_field_line(
                r_initial, vars_to_save_dict.keys(), dt=dt
            )
            stat_dict["footpoint_id"] = grid_indices[i]
            for key in vars_to_save_dict.keys():
                vars_to_save_dict[key].append(stat_dict[key])

            if save_field_lines:
                if len(stat_dict["field_lines"]) > 0:
                    field_lines_size_kB += len(stat_dict["field_lines"][:, 0]) * 0.012

            file_size_Mb = (vars_to_save_size_kB + field_lines_size_kB) / 1.0e3
            if save_file:
                if file_size_Mb > max_file_size_Mb:
                    io.write_to_fits(
                        self.field_components,
                        footpoints_grid,
                        vars_to_save_dict,
                        filename_cur,
                        self.R_min,
                        self.R_max,
                    )
                    file_no += 1
                    filename_cur = filename + "_" + "%03d.fits" % file_no
                    # initialize columns
                    vars_to_save_dict = self.set_vars_to_save(vars_to_save)
                    vars_to_save_size_kB = 0
                    field_lines_size_kB = 0

        if save_file:
            io.write_to_fits(
                self.field_components,
                footpoints_grid,
                vars_to_save_dict,
                filename_cur,
                self.R_min,
                self.R_max,
            )
            file_no += 1
            filename_cur = filename + "_" + "%03d.fits" % file_no

        # del vars_to_save_dict
        return footpoints_grid, self.field_components, vars_to_save_dict

    def get_single_field_line(
        self, r_initial: CartesianRepresentation, vars_to_save, dt=0.01, save_dR=False
    ):
        r_initial = r_initial.xyz.value
        B_initial = self.get_total_field(0, r_initial, 1, normalize=False)
        B_initial_norm = self.get_total_field(0, r_initial, 1, normalize=True)
        dotprod = np.dot(B_initial_norm, r_initial)
        K = np.sign(dotprod)
        field_line = solve_ivp(
            self.get_total_field,
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
        # Change to Cartesian representation
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
                # r = np.random.normal(size=(3,1000))
                output_dict["field_lines"] = r.T
            else:
                output_dict["field_lines"] = []

        if save_dR:
            dr = r[:, 1:] - r[:, :-1]
            dR = np.median(np.sum(dr**2, axis=0) ** 0.5)
            output_dict["dR"] = dR
        return output_dict

    def estimate_computing_time(
        self, footpoints_grid, vars_to_save_dict: dict, n_lines_estimate=20, dt=0.01
    ):
        n_lines_total = footpoints_grid.nline
        if n_lines_estimate >= n_lines_total:
            n_lines_estimate = n_lines_total
            indices_estimate = np.arange(n_lines_estimate)
        else:
            indices_estimate = np.arange(n_lines_total)
            np.random.shuffle(indices_estimate)
            indices_estimate = indices_estimate[:n_lines_estimate]
        r_initial_grid = footpoints_grid.footpoint_coordinates
        print(
            f"Estimating completion time and file size using {n_lines_estimate} lines with dt = {dt}..."
        )
        for i in [0]:
            r_initial = r_initial_grid[i]
            stat_dict = self.get_single_field_line(
                r_initial, vars_to_save_dict.keys(), dt=dt, save_dR=True
            )
            dR = stat_dict["dR"]
            R_span = 3 * (self.R_max - R_NS)
            n_points_max = int(R_span / dR)
            print(f"Distance between points on a field line: {dR: .1e}")
            print(
                f"Estimated maximum number of points for 3*(R_max - R_NS) = {R_span}: {n_points_max}"
            )

        start_time = time.time()
        for i in indices_estimate:
            stat_dict = self.get_single_field_line(
                r_initial_grid[i], vars_to_save_dict.keys(), dt=dt
            )
        end_time = time.time()
        estimated_duration = (end_time - start_time) / n_lines_estimate * n_lines_total
        print(
            f"Based on {n_lines_estimate} line calculations, computing {n_lines_total} lines will take:\n{estimated_duration: .0f} s\nor\n{estimated_duration/3600.: .1f} hr."
        )
        return estimated_duration


if __name__ == "__main__":
    
    params = SphericalGridParams(nlon=36, nlat=18, pole_lat=70.)
    grid = FootpointGrid(GridType.SPHERICAL, params)
        
    component1 = FieldComponent(
        moment_type=MultipoleType.DIPOLE  # Only required parameter
    )

    # Full usage with all parameters
    component2 = FieldComponent(
        moment_type=MultipoleType.QUADRUPOLE,
        B_magnitude=2.0,
        moment_orientation_latitude_degree=45.0,
        moment_orientation_longitude_degree=90.0,
        offset_x_r_ns=0.1,
        offset_y_r_ns=0.2,
        offset_z_r_ns=0.3,
        component_number=1
    )

    component_list = [component1, component2]

    magnetic_field = MultipoleField()
    magnetic_field.initialize(
        [
            {
                "moment_type": "quadrupole",
                "moment_orientation_latitude_degree": 0.0,
                "moment_orientation_longitude_degree": 0.0,
            },
            {
                "moment_type": "dipole",
                "moment_orientation_latitude_degree": 90.0,
                "moment_orientation_longitude_degree": 0.0,
            },
        ],
    )
    magnetic_field.get_field_lines(
        grid,
        "test_file",
        vars_to_save="footpoint_id,R_max,R_cyl_max,z_abs_max,r_ini,field_lines",  # "footpoint_id, field_lines",
        R_max=3.0,
        R_min=1.0,
        max_file_size_Mb=2,
    )
