import time
import datetime
import sys
import numpy as np
from tqdm import tqdm
from astropy import units as u
from astropy.io import fits as pf
from scipy.integrate import solve_ivp
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation
from scipy.spatial.transform import Rotation as rot
from InitializeFootpointGrid import FootpointGrid
from FieldComponent import FieldComponent
import InputOutput as io

r_ns = u.def_unit("Neutron star radius, 10 km", 10 * u.km)
u.add_enabled_units([r_ns])

class MultipoleField:
    def initialize(self, component_params_list: list, R_max: float, R_min: float):
        """
        Gets list of dictionaries
        """
        multipole_components = []
        for component_params in component_params_list:
            multipole_component = FieldComponent(**component_params)
            multipole_components.append(multipole_component)

        self.multipole_components = multipole_components
        self.R_NS = 1
        self.R_min = R_min
        self.R_max = R_max

    def load_from_file(self):
        multipole_components = io.read_from_fits(filename)
        return 


    def set_saved_vars(self, vars: str):
        allowed_variables = set(
            ("R_max", "R_cyl_max", "z_abs_max", "B_ini", "B_R_ini_angle")
        )
        if vars == "all":
            vars_to_save = allowed_variables
        else:
            vars = vars.replace(" ", "").split(",")
            vars_to_save = vars & allowed_variables
            not_allowed_vars = vars - allowed_variables
            assert len(not_allowed_vars) == 0, print(
                f"Following variables are not recognized: {not_allowed_vars}.\nPlease use the following: {allowed_variables}"
            )
        if len(vars_to_save) == 0:
            raise ValueError(f"No variables selected for saving. Aborting.")
        vars_to_save_dict = {var: [] for var in vars_to_save}
        vars_to_save_dict["footpoint_id"] = []
        return vars_to_save_dict

    def get_total_field(self, t, r, K, normalize=True):
        F = np.array([0.0, 0.0, 0.0])
        for multipole in self.multipole_components:
            F += multipole.get_field(r)
        F = F * K
        if normalize:
            F /= np.sum(F**2) ** 0.5
        return F

    def hit_surface(self, t, r, K):
        r = np.sum(r**2) ** 0.5
        return r - 0.999 * self.R_NS

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
        filename,
        vars_to_save="all",
        save_footpoint_grid=True,
        save_field_lines=True,
        dt=0.01,
        max_file_size_Mb=200,
    ):
        self.estimate_computing_time(footpoints_grid)

        grid_indices = np.arange(footpoints_grid.nline)
        r_initial_grid = footpoints_grid.footpoint_coordinates

        vars_to_save_dict = self.set_saved_vars(vars_to_save)
        field_lines = []
        vars_to_save_size = 0
        field_lines_size = 0
        file_no = 0
        filename_cur = filename + "%03d.fits" % file_no
        print(f"Starting field line calculations at {datetime.datetime.now()}")
        for i in tqdm(grid_indices):
            vars_to_save_size += 100.
            r_initial = r_initial_grid[i].xyz.value
            stat_dict = self.get_single_field_line(
                r_initial, dt=dt, save_field_line=True
            )
            stat_dict["footpoint_id"] = i
            for key in vars_to_save_dict.keys():
                vars_to_save_dict[key].append(stat_dict[key])
            if save_field_lines:
                if stat_dict["R_max"] >= self.R_min:
                    field_lines.append(stat_dict["field_line"].T)
                    # Empirically measured size of one entry in fits file
                    field_lines_size += len(stat_dict["field_line"][0,:]) * 12.
                    
                else:
                    field_lines.append([])
                
            file_size_Mb = (vars_to_save_size + field_lines_size) / 1.0e6 
            if file_size_Mb > max_file_size_Mb:
                io.write_to_fits(
                    footpoints_grid,
                    vars_to_save_dict,
                    filename_cur,
                    field_lines=field_lines,
                )
                file_no += 1
                filename_cur = filename + "%03d.fits" % file_no
                # initialize columns
                vars_to_save_dict = self.set_saved_vars(vars_to_save)
                field_lines = []
                vars_to_save_size = 0
                field_lines_size = 0

        io.write_to_fits(
            footpoints_grid, vars_to_save_dict, filename_cur, field_lines=field_lines, save_footpoint_grid,
        )
        file_no += 1
        filename_cur = filename + "%03d.fits" % file_no
        del vars_to_save_dict
        if save_field_lines:
            del field_lines

        return 0


    def estimate_computing_time(self, footpoints_grid):
        n_lines_total = footpoints_grid.nline
        n_lines_estimate = 20
        dt = 0.01
        if n_lines_estimate >= n_lines_total:
            trial_indices = np.arange(n_lines_total)
            n_lines_estimate = n_lines_total
        else:
            trial_indices = np.linspace(0, n_lines_total - 1, n_lines_estimate).astype(
                int
            )
        r_initial_grid = footpoints_grid.footpoint_coordinates
        print(
            f"Estimating completion time and file size using {n_lines_estimate} lines with dt = {dt}..."
        )
        for i in [trial_indices[0]]:
            r_initial = r_initial_grid[i]
            stat_dict = self.get_single_field_line(r_initial, dt=dt, save_dR=True)
            dR = stat_dict["dR"]
            R_span = 3 * (self.R_max - self.R_NS)
            n_points_max = int(R_span / dR)
            print(f"Distance between points on a field line: {dR: .1e}")
            print(
                f"Estimated maximum number of points for 3*(R_max - R_NS) = {R_span}: {n_points_max}"
            )

        start_time = time.time()
        for i in trial_indices:
            stat_dict = self.get_single_field_line(
                r_initial_grid[i], dt=dt, save_field_line=True
            )
        end_time = time.time()
        estimated_duration = (end_time - start_time) / n_lines_estimate * n_lines_total
        print(
            f"Based on {n_lines_estimate} line calculations, computing {n_lines_total} lines will take:\n{estimated_duration: .0f} s\nor\n{estimated_duration/3600.: .1f} hr."
        )
        return 0

    def get_single_field_line(
        self, r_initial: CartesianRepresentation, dt=0.01, save_dR=False, save_field_line=False
    ):
        theta_ini, phi_ini = r_initial. # Add spherical representation
        r_initial = r_initial.xyz.value
        B_initial = self.get_total_field(0, r_initial, 1, normalize=False)
        dotprod = np.dot(B_initial, r_initial)
        chi = np.arccos(dotprod)
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

        t = np.arange(field_line.t[0], field_line.t[-1] + dt, dt)
        r = field_line.sol(t)
        R_max = np.max(np.sum(r**2, axis=0) ** 0.5)
        R_cyl_max = np.max(np.sum(r[:2,] ** 2, axis=0) ** 0.5)
        z_abs_max = np.max(np.abs(r[2, :]))

        if save_dR:
            dr = r[:, 1:] - r[:, :-1]
            dR = np.median(np.sum(dr**2, axis=0) ** 0.5)

        stat_dict = {"R_max": R_max, "R_cyl_max": R_cyl_max, "z_abs_max": z_abs_max}
        stat_dict["B_R_ini_angle"] = chi
        stat_dict["B_ini"] = np.sum(B_initial**2) ** 0.5
        stat_dict["r_ini"] = r_initial
        if save_dR:
            stat_dict["dR"] = dR
        if save_field_line:
            stat_dict["field_line"] = r #np.random.uniform(size=(3,2000))

        return stat_dict


if __name__ == "__main__":

    grid = FootpointGrid()
    r0 = 1 * r_ns
    grid.calculate_grid("healpix", {"nside": 8}, r0)
    # grid.calculate_grid("spherical", {"pole_lat": 0, "nlon": 2, "nlat": 20}, r0)
    # grid.load_grid_from_fits("test.fits")
    # grid.plot_footpoint_coordinates()
    magnetic_field = MultipoleField(
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
        R_max=300.0,
        R_min=2.,
    )
    magnetic_field.get_field_lines(
        grid, "test_file", vars_to_save="all", save_field_lines=True, max_file_size_Mb=2
    )
