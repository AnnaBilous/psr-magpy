import numpy as np
import matplotlib.pyplot as plt
from psr_magpy import *
import matplotlib.patches as patches
import argparse
import os
from pathlib import Path


import sys
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from psr_magpy import *


def filter_by_tangent_theta(
    field_line_set, 
    target_theta: float, 
    tolerance: float
) -> tuple:
    """Filter points by tangent_theta and aggregate across all lines.
    
    Args:
        field_line_set: FieldLineSet with tangent_theta, tangent_phi, and pa_line properties
        target_theta: Target tangent_theta value (degrees)
        tolerance: Tolerance around target_theta (degrees)
    
    Returns:
        Tuple of (filtered_tangent_theta, filtered_tangent_phi, filtered_pa_line) as numpy arrays
    """
    tangent_theta_data = field_line_set.get_property('tangent_theta')
    tangent_phi_data = field_line_set.get_property('tangent_phi')
    pa_line_data = field_line_set.get_property('pa_line')
    
    
    if tangent_theta_data is None:
        error_msg = "tangent_theta data not available in field line set"
        print(f"ERROR: {error_msg}")
        return None, None, None
    
    filtered_theta = []
    filtered_phi = []
    filtered_pa = []
    
    n_lines = len(tangent_theta_data)
    
    for i in range(n_lines):
        theta_line = tangent_theta_data[i]
        phi_line = tangent_phi_data[i] if tangent_phi_data is not None else None
        pa_line = pa_line_data[i] if pa_line_data is not None else None
        """
        if i%100 == 0 :
            plt.scatter(np.arange(len(pa_line)), pa_line, s=5, c=theta_line, cmap='hsv', vmin=-180, vmax=180)
            plt.twinx()
            plt.plot(phi_line, 'g.-', label=f'Line {i}')
            plt.legend(loc='upper right')

            plt.show()
        """
        if theta_line is None or len(theta_line) == 0:
            continue
        
        # Filter points where tangent_theta is within tolerance
        mask = np.abs(theta_line - target_theta) <= tolerance
        
        if np.any(mask):
            filtered_theta.extend(theta_line[mask])
            if phi_line is not None:
                filtered_phi.extend(phi_line[mask])
            if pa_line is not None:
                filtered_pa.extend(pa_line[mask])
    
    filtered_theta = np.array(filtered_theta) if filtered_theta else np.array([])
    filtered_phi = np.array(filtered_phi) if filtered_phi else np.array([])
    filtered_pa = np.array(filtered_pa) if filtered_pa else np.array([])
    
    return filtered_theta, filtered_phi, filtered_pa


if __name__ == "__main__":

    series_config = ExperimentSeriesConfig.load("/home/bilous/Research/Codebase/psr_magpy/PA_check/configs/PA_check_config.yaml")
    print(series_config)
    
  
    # Create minimal template that only saves the properties we need
    from psr_magpy.line.field_line import FieldLine
    from psr_magpy.line.field_line_set import FieldLineSet
    minimal_template = FieldLine(
        save_footpoint_id=True,
        save_r_line=True,
        save_pa_line=True,
        save_tangent_phi=True,
        save_tangent_theta=True,
        # All other properties default to False
    )
    
    # Load data from all experiments in the series
    lines_dir = series_config.get_lines_dir()
    n_experiments = series_config.get_total_experiments_from_summary()
    
    combined_field_line_set = FieldLineSet(minimal_template, copy_data=False)
    target_theta = 85.0
    tolerance = 0.1
        
    for exp_num in [0,1]:
        exp_config = series_config.load_experiment_config(exp_num)
        exp_data = exp_config.load_combined_batch_data(minimal_template, str(lines_dir))
        for target_theta in np.arange(0.0, 95.0, 2.0):
            filt_theta, filt_phi, filt_pa = filter_by_tangent_theta(
                exp_data, 
                target_theta=target_theta, 
                tolerance=tolerance
            )
            print(filt_theta, filt_phi, filt_pa)

            plt.scatter(filt_phi, filt_pa, s=5, c=filt_pa, cmap='hsv')
            plt.colorbar()
            plt.xlabel('Tangent Phi')
            plt.ylabel('PA Line')
            plt.title(f'PA vs Tangent Phi for Target Theta {target_theta} with Tolerance {tolerance}')
            plt.grid(True, alpha=0.3)
            plt.show()
        
        
        