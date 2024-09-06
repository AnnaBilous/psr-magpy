import time
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy import units as u
from astropy.io import fits as pf
from scipy.integrate import solve_ivp
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation
from scipy.spatial.transform import Rotation as rot
from InitializeFootpointGrid import FootpointGrid


r_ns = u.def_unit("Neutron star radius, 10 km", 10 * u.km)
u.add_enabled_units([r_ns])


class field_component:
    def __init__(self, **kwargs):

        default_values = {
            "component_number": 0,
            "moment_type": "dipole",
            "B_magnitude": 1,
            "moment_orientation_latitude_degree": 45.0,
            "moment_orientation_longitude_degree": 0,
            "offset_x_r_ns": 0,
            "offset_y_r_ns": 0,
            "offset_z_r_ns": 0,
        }

        expected_types = {
            "component_number": int,
            "moment_type": str,
            "B_magnitude": float,
            "moment_orientation_latitude_degree": float,
            "moment_orientation_longitude_degree": float,
            "offset_x_r_ns": float,
            "offset_y_r_ns": float,
            "offset_z_r_ns": float,
        }

        for key in kwargs:
            if key not in default_values:
                raise ValueError(f"Invalid parameter name: {key}")

        for key, value in kwargs.items():
            expected_type = expected_types[key]
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Incorrect type for parameter {key}: expected {expected_type}"
                )

        assert kwargs["moment_type"] in [
            "dipole",
            "quadrupole",
        ], "Multipole must be dipole or quadrupole"

        print("Setting up field components with the following vars:")
        for key in default_values:
            print(f"\t\t{key}: {default_values[key]}")
        print("--------------------------------------")
        default_values.update(kwargs)

        self.component_number = default_values["component_number"]
        self.moment_type = default_values["moment_type"]
        self.B_magnitude = default_values["B_magnitude"]
        self.moment_orientation = PhysicsSphericalRepresentation(
            phi=default_values["moment_orientation_longitude_degree"] * u.deg,
            theta=default_values["moment_orientation_latitude_degree"] * u.deg,
            r=0,
        )
        self.moment_offset_vector = CartesianRepresentation(
            x=default_values["offset_x_r_ns"],
            y=default_values["offset_y_r_ns"],
            z=default_values["offset_z_r_ns"],
        )

        self.rotate_magn_to_spin_frame = rot.from_euler(
            "yzx",
            [self.moment_orientation.theta.rad, self.moment_orientation.phi.rad, 0],
        )
        self.rotate_spin_to_magn_frame = self.rotate_magn_to_spin_frame.inv()

    def get_field(self, r_spin_frame: np.array):
        r_spin_frame = r_spin_frame * 1
        r_spin_frame = CartesianRepresentation(x=r_spin_frame)
        r_offset_spin_frame = r_spin_frame - self.moment_offset_vector
        r_magn_frame_value = self.rotate_spin_to_magn_frame.apply(
            r_offset_spin_frame.xyz.T
        )
        r_magn_frame = CartesianRepresentation(x=r_magn_frame_value)
        r_magn_frame_cyl = CartesianRepresentation(
            x=r_magn_frame.x, y=r_magn_frame.y, z=0 * r_magn_frame.z
        )
        r = r_magn_frame.norm()
        r_cyl = r_magn_frame_cyl.norm()

        x = r_magn_frame.x
        y = r_magn_frame.y
        z = r_magn_frame.z

        # Magneic field defined in cartesian coordinates
        if self.moment_type == "dipole":
            B_magnetic_frame = (
                self.B_magnitude
                * r**-5
                * np.array(
                    [
                        1.5 * z * x,
                        1.5 * z * y,
                        z**2 - 0.5 * r_cyl**2,
                    ]
                )
            )
        if self.moment_type == "quadrupole":
            denom = (0.5 * (5 * z**2 / r**2 - 1) / r) ** -1
            B_magnetic_frame = (
                self.B_magnitude
                * r**-4
                * np.array(
                    [
                        x / denom,
                        y / denom,
                        z / denom - z / r,
                    ]
                )
            )

        B_spin_frame = self.rotate_magn_to_spin_frame.apply(B_magnetic_frame)

        return B_spin_frame


class MultipoleField:
    def __init__(self, component_params_list: list, R_max: float, R_min: float):
        """
        Gets list of dictionaries
        """
        multipole_components = []
        for component_params in component_params_list:
            multipole_component = field_component(**component_params)
            multipole_components.append(multipole_component)

        self.multipole_components = multipole_components
        self.R_NS = 1
        self.R_min = R_min
        self.R_max = R_max

    def set_saved_vars(self, vars: str):
        allowed_variables = set(
            ("r_ini", "R_max", "R_cyl_max", "z_abs_max", "B_ini", "B_R_ini_angle")
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
                self.write_to_fits(
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

        self.write_to_fits(
            footpoints_grid, vars_to_save_dict, filename_cur, field_lines=field_lines
        )
        file_no += 1
        filename_cur = filename + "%03d.fits" % file_no
        del vars_to_save_dict
        if save_field_lines:
            del field_lines

        return 0

    def write_to_fits(
        self, footpoints_grid, vars_to_save_dict, filename, field_lines=None
    ):
        print(f"Saving {vars_to_save_dict.keys()}")
        print(f"Saving field lines: {not(field_lines is None)}")
        hdu_main = pf.PrimaryHDU()
        header_main = hdu_main.header
        header_main.set("GRIDTYPE", footpoints_grid.grid_type, "Grid type")
        header_main.set("NLINE", footpoints_grid.nline, "Number of field lines")
        header_main.set(
            "R0", footpoints_grid.r0.value, "Radius of the sphere with footpoints"
        )
        header_main.set("R0UNITS", footpoints_grid.r0.unit.name, "Unit of r0")
        if footpoints_grid.grid_type == "spherical":
            header_main.set(
                "NLON", footpoints_grid.nlon, "Number of longitude divisions"
            )
            header_main.set(
                "NLAT", footpoints_grid.nlat, "Number of latitude divisions"
            )
            header_main.set(
                "POLELAT", footpoints_grid.pole_lat, "Latitude of grid pole, degree"
            )
        if footpoints_grid.grid_type == "healpix":
            header_main.set("NSIDE", footpoints_grid.nside, "Healpix parameter nside")

        magnetic_field_list = [
            vars(component) for component in self.multipole_components
        ]
        column_names = ["component_number", "moment_type", "B_magnitude"]
        column_formats = ["K", "10A", "D"]
        magnetic_field_columns = []
        for i, key in enumerate(column_names):
            value = [v[key] for v in magnetic_field_list]
            col = pf.Column(name=key, format=column_formats[i], array=value)
            magnetic_field_columns.append(col)

        value = [v["moment_orientation"].phi.degree for v in magnetic_field_list]
        col = pf.Column(name="orientation_phi_deg", format="D", array=value)
        magnetic_field_columns.append(col)
        value = [v["moment_orientation"].theta.degree for v in magnetic_field_list]
        col = pf.Column(name="orientation_theta_deg", format="D", array=value)
        magnetic_field_columns.append(col)

        value = [v["moment_offset_vector"].x for v in magnetic_field_list]
        col = pf.Column(name="offset_x_R_NS", format="D", array=value)
        magnetic_field_columns.append(col)
        value = [v["moment_offset_vector"].y for v in magnetic_field_list]
        col = pf.Column(name="offset_y_R_NS", format="D", array=value)
        magnetic_field_columns.append(col)
        value = [v["moment_offset_vector"].z for v in magnetic_field_list]
        col = pf.Column(name="offset_z_R_NS", format="D", array=value)
        magnetic_field_columns.append(col)
        hdu_field_components = pf.BinTableHDU.from_columns(
            magnetic_field_columns, name="field_components"
        )

        fits_columns = []
        for key, value in vars_to_save_dict.items():
            if key == "footpoint_id":
                col_format = "K"
            elif key == "r_ini":
                col_format = "3D"
            else:
                col_format = "D"
            col = pf.Column(name=key, format=col_format, array=value)
            fits_columns.append(col)
        if field_lines is not None:
            col = pf.Column(name="field_lines", format="PE", array=field_lines)
            fits_columns.append(col)
        hdu = pf.BinTableHDU.from_columns(fits_columns, name="field_data")
        print(f"Writing {filename}.")
        hdulist = pf.HDUList([hdu_main, hdu_field_components, hdu])
        hdulist.writeto(filename, overwrite=True)
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
            r_initial = r_initial_grid[i].xyz.value
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
            r_initial = r_initial_grid[i].xyz.value
            stat_dict = self.get_single_field_line(
                r_initial, dt=dt, save_field_line=True
            )
        end_time = time.time()
        estimated_duration = (end_time - start_time) / n_lines_estimate * n_lines_total
        print(
            f"Based on {n_lines_estimate} line calculations, computing {n_lines_total} lines will take:\n{estimated_duration: .0f} s\nor\n{estimated_duration/3600.: .1f} hr."
        )
        return 0

    def get_single_field_line(
        self, r_initial, dt=0.01, save_dR=False, save_field_line=False
    ):

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
