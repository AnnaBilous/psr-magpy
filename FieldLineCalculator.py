import time
import datetime
import numpy as np
from astropy.coordinates.representation import CartesianRepresentation
from tqdm import tqdm
from typing import Dict, Tuple

from Constants import R_NS
import WriteToFits as io
from FieldLineTracer import FieldLineTracer
from FootpointGrid import FootpointGrid

class FieldLineCalculator:
    def __init__(self, field):
        """Initialize field line calculator.
        
        Parameters
        ----------
        field : MultipoleField
            Object that calculates the magnetic field
        """
        self.field = field
        self._field_line_tracer = None

    def set_vars_to_save(self, vars: str) -> Dict:
        """Set which variables to save during field line calculations.
        
        Parameters
        ----------
        vars : str
            Comma-separated list of variables to save
            
        Returns
        -------
        Dict
            Dictionary with empty lists for each variable to save
        """
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
        return {var: [] for var in vars_to_save}

    def get_single_field_line(self, r_initial: CartesianRepresentation, vars_to_save: Dict, 
                            dt=0.01, save_dR=False, R_max=2.0, R_min=1.0) -> Dict:
        """Calculate a single field line.
        
        Parameters
        ----------
        r_initial : CartesianRepresentation
            Starting point for field line
        vars_to_save : Dict
            Dictionary specifying which variables to save
        dt : float, optional
            Integration time step, by default 0.01
        save_dR : bool, optional
            Whether to save step size info, by default False
        R_max : float, optional
            Maximum radius to trace to, by default 2.0
        R_min : float, optional
            Minimum radius to trace to, by default 1.0
            
        Returns
        -------
        Dict
            Dictionary containing requested variables
        """
        if not self._field_line_tracer:
            self._field_line_tracer = FieldLineTracer(self.field, R_max, R_min)
        return self._field_line_tracer.trace_single_line(r_initial, vars_to_save, dt, save_dR)

    def get_field_lines(self, footpoints_grid: FootpointGrid, filename="", vars_to_save="", 
                       R_max=10.0, R_min=1.0, dt=0.01, save_file=True, 
                       max_file_size_Mb=200) -> Tuple[FootpointGrid, list, Dict]:
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
        Tuple[FootpointGrid, list, Dict]
            Footpoint grid, field components, and calculated variables
        """
        if max_file_size_Mb < 0.02:
            print("Max file size is smaller than the size of header. Setting it to 20 kB.")
            max_file_size_Mb = 0.02

        r_initial_grid = footpoints_grid.footpoint_coordinates
        grid_indices = footpoints_grid.indices
        vars_to_save_dict = self.set_vars_to_save(vars_to_save)

        completion_time_sec = self.estimate_computing_time(footpoints_grid, vars_to_save_dict)
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
                r_initial, vars_to_save_dict.keys(), dt=dt, R_max=R_max, R_min=R_min
            )
            stat_dict["footpoint_id"] = grid_indices[i]
            for key in vars_to_save_dict.keys():
                vars_to_save_dict[key].append(stat_dict[key])

            if save_field_lines and len(stat_dict["field_lines"]) > 0:
                field_lines_size_kB += len(stat_dict["field_lines"][:, 0]) * 0.012

            file_size_Mb = (vars_to_save_size_kB + field_lines_size_kB) / 1.0e3
            if save_file and file_size_Mb > max_file_size_Mb:
                io.write_to_fits(
                    self.field.field_components,
                    footpoints_grid,
                    vars_to_save_dict,
                    filename_cur,
                    R_min,
                    R_max,
                )
                file_no += 1
                filename_cur = filename + "_" + "%03d.fits" % file_no
                vars_to_save_dict = self.set_vars_to_save(vars_to_save)
                vars_to_save_size_kB = 0
                field_lines_size_kB = 0

        if save_file:
            io.write_to_fits(
                self.field.field_components,
                footpoints_grid,
                vars_to_save_dict,
                filename_cur,
                R_min,
                R_max,
            )

        return footpoints_grid, self.field.field_components, vars_to_save_dict

    def estimate_computing_time(self, footpoints_grid: FootpointGrid, 
                              vars_to_save_dict: Dict, n_lines_estimate=20, 
                              dt=0.01) -> float:
        """Estimate time needed to compute field lines.
        
        Parameters
        ----------
        footpoints_grid : FootpointGrid
            Grid of initial points
        vars_to_save_dict : Dict
            Dictionary of variables to save
        n_lines_estimate : int, optional
            Number of lines to use for estimate, by default 20
        dt : float, optional
            Integration time step, by default 0.01
            
        Returns
        -------
        float
            Estimated computation time in seconds
        """
        n_lines_total = footpoints_grid.nline
        if n_lines_estimate >= n_lines_total:
            n_lines_estimate = n_lines_total
            indices_estimate = np.arange(n_lines_estimate)
        else:
            indices_estimate = np.random.choice(
                np.arange(n_lines_total), n_lines_estimate, replace=False
            )

        r_initial_grid = footpoints_grid.footpoint_coordinates
        print(f"Estimating completion time using {n_lines_estimate} lines with dt = {dt}...")

        r_initial = r_initial_grid[0]
        stat_dict = self.get_single_field_line(
            r_initial, vars_to_save_dict.keys(), dt=dt, save_dR=True
        )
        dR = stat_dict["dR"]
        R_span = 3 * (self._field_line_tracer.R_max - R_NS)
        n_points_max = int(R_span / dR)
        print(f"Distance between points on a field line: {dR: .1e}")
        print(f"Estimated maximum number of points for 3*(R_max - R_NS) = {R_span}: {n_points_max}")

        start_time = time.time()
        for i in indices_estimate:
            self.get_single_field_line(
                r_initial_grid[i], vars_to_save_dict.keys(), dt=dt
            )
        end_time = time.time()
        
        estimated_duration = (end_time - start_time) / n_lines_estimate * n_lines_total
        print(
            f"Based on {n_lines_estimate} line calculations, computing {n_lines_total} lines will take:\n"
            f"{estimated_duration: .0f} s\nor\n{estimated_duration/3600.: .1f} hr."
        )
        return estimated_duration 