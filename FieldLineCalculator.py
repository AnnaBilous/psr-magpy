import time
import datetime
import numpy as np
from astropy.coordinates.representation import CartesianRepresentation
from tqdm import tqdm
from typing import Dict, Tuple, Set, Optional, Sequence, NDArray
from enum import Enum
from dataclasses import dataclass

from Constants import R_NS
import WriteToFits as io
from FieldLineTracer import FieldLineTracer
from FootpointGrid import FootpointGrid

class Var(Enum):
    """Variables that can be saved during field line calculations."""
    R_MAX = "r_max"
    R_CYL_MAX = "r_cyl_max"
    Z_ABS_MAX = "z_abs_max"
    B_INI = "b_ini"
    B_R_INI_ANGLE = "b_r_ini_angle"
    R_INI = "r_ini"
    FOOTPOINT_ID = "footpoint_id"
    FIELD_LINES = "field_lines"

@dataclass
class FieldLine:
    """Single field line and its properties."""
    # Required property
    footpoint_id: int
    
    # Optional path of the field line in 3D space
    points: Optional[NDArray[np.float64]] = None  # Shape: (n_points, 3)
    
    # Optional scalar properties at the initial point
    r_max: Optional[float] = None
    r_cyl_max: Optional[float] = None
    z_abs_max: Optional[float] = None
    b_ini: Optional[float] = None
    b_r_ini_angle: Optional[float] = None
    r_ini: Optional[float] = None

@dataclass
class FieldLineSet:
    """Collection of field lines."""
    lines: Sequence[FieldLine]

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    # Perhaps not needed
    @property
    def footpoint_ids(self) -> NDArray[np.int64]:
        return np.array([line.footpoint_id for line in self.lines])


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

    def prepare_variables_dict(self, variables: Optional[Set[Var]] = None) -> Dict:
        """Prepare dictionary for variables to save.
        
        Parameters
        ----------
        variables : Set[Var], optional
            Set of variables to save. If None, saves all variables.
            
        Returns
        -------
        Dict
            Dictionary with empty lists for each variable to save
        """
        if variables is None:
            variables = set(Var)
            
        return {var.value: [] for var in variables}

    def get_single_field_line(self, r_initial: CartesianRepresentation, 
                            variables: Set[Var], dt=0.01, save_dR=False, 
                            R_max=2.0, R_min=1.0) -> Dict:
        """Calculate a single field line.
        
        Parameters
        ----------
        r_initial : CartesianRepresentation
            Starting point for field line
        variables : Set[Var]
            Set of variables to save
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
        return self._field_line_tracer.trace_single_line(r_initial, {var.value for var in variables}, dt, save_dR)

    def get_field_lines(self, footpoints_grid: FootpointGrid, 
                       template: FieldLine,
                       filename: str = "", 
                       R_max: float = 10.0, 
                       R_min: float = 1.0, 
                       dt: float = 0.01) -> FieldLineSet:
        """Calculate field lines based on template.
        
        Parameters
        ----------
        footpoints_grid : FootpointGrid
            Grid of initial points
        template : FieldLine
            Template specifying which properties to calculate
        """
        # Properties to calculate are those that are not None in template
        properties_to_calculate = {
            name for name, value in vars(template).items() 
            if value is not None and name != 'footpoint_id'
        }

        if max_file_size_Mb < 0.02:
            print("Max file size is smaller than the size of header. Setting it to 20 kB.")
            max_file_size_Mb = 0.02

        r_initial_grid = footpoints_grid.footpoint_coordinates
        grid_indices = footpoints_grid.indices
        vars_to_save_dict = self.prepare_variables_dict(properties_to_calculate)

        completion_time_sec = self.estimate_computing_time(footpoints_grid, properties_to_calculate)
        too_long_sec = 300.0
        if (completion_time_sec > too_long_sec) & (save_file == False):
            print(
                f"Warning: calculations will take {completion_time_sec/60.} min. According to input parameters, files will not be saved to disk. Are you sure?"
            )

        save_field_lines = Var.FIELD_LINES in (properties_to_calculate or set(Var))
        vars_to_save_size_increment_kB = 0.06
        vars_to_save_size_kB = 12  # Size of header
        field_lines_size_kB = 0
        file_no = 0
        filename_cur = filename + "_%03d.fits" % file_no

        print(f"Starting field line calculations at {datetime.datetime.now()}")
        field_lines = []
        for i in tqdm(np.arange(footpoints_grid.nline)):
            vars_to_save_size_kB += vars_to_save_size_increment_kB
            r_initial = r_initial_grid[i]
            stat_dict = self.get_single_field_line(
                r_initial, properties_to_calculate, dt=dt, R_max=R_max, R_min=R_min
            )
            stat_dict[Var.FOOTPOINT_ID.value] = grid_indices[i]
            for key in vars_to_save_dict:
                vars_to_save_dict[key].append(stat_dict[key])

            if save_field_lines and len(stat_dict[Var.FIELD_LINES.value]) > 0:
                field_lines_size_kB += len(stat_dict[Var.FIELD_LINES.value][:, 0]) * 0.012

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
                vars_to_save_dict = self.prepare_variables_dict(properties_to_calculate)
                vars_to_save_size_kB = 0
                field_lines_size_kB = 0

            field_lines.append(FieldLine(**stat_dict))

        if save_file:
            io.write_to_fits(
                self.field.field_components,
                footpoints_grid,
                vars_to_save_dict,
                filename_cur,
                R_min,
                R_max,
            )

        return FieldLineSet(field_lines)

    def estimate_computing_time(self, footpoints_grid: FootpointGrid, 
                              variables: Set[Var], n_lines_estimate=20, 
                              dt=0.01) -> float:
        """Estimate time needed to compute field lines.
        
        Parameters
        ----------
        footpoints_grid : FootpointGrid
            Grid of initial points
        variables : Set[Var]
            Set of variables to save
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
            r_initial, variables, dt=dt, save_dR=True
        )
        dR = stat_dict["dR"]
        R_span = 3 * (self._field_line_tracer.R_max - R_NS)
        n_points_max = int(R_span / dR)
        print(f"Distance between points on a field line: {dR: .1e}")
        print(f"Estimated maximum number of points for 3*(R_max - R_NS) = {R_span}: {n_points_max}")

        start_time = time.time()
        for i in indices_estimate:
            self.get_single_field_line(
                r_initial_grid[i], variables, dt=dt
            )
        end_time = time.time()
        
        estimated_duration = (end_time - start_time) / n_lines_estimate * n_lines_total
        print(
            f"Based on {n_lines_estimate} line calculations, computing {n_lines_total} lines will take:\n"
            f"{estimated_duration: .0f} s\nor\n{estimated_duration/3600.: .1f} hr."
        )
        return estimated_duration 