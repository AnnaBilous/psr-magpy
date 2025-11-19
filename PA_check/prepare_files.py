import numpy as np
import matplotlib.pyplot as plt
from psr_magpy import *
import matplotlib.patches as patches
import argparse
import os
from pathlib import Path

def plot_field_lines(
    stat_dict, R_LC, cap_heights, cl_closed, cl_open_fl, cl_open_lc, lw_closed, savefig
):
    field_lines = stat_dict["field_lines"]
    nline = len(field_lines)

    plt.figure(figsize=(4, 4 * 2.6))
    ax = plt.subplot(111)
    ax.set_aspect = "equal"

    line_OFL_cap_height_num = np.zeros(nline) - 1

    # OFLs are nested, hence two cycles
    # 1
    for c, cap_height in enumerate(cap_heights):
        OFL_footpoints = tls.get_OFL_footpoints(stat_dict, R_LC, h_LC_coeff=cap_height)
        line_OFL_cap_height_num[OFL_footpoints] = c

    # 2
    for c in np.arange(-1, len(cap_heights)):
        OFL_footpoints = np.array(stat_dict["footpoint_id"])[
            line_OFL_cap_height_num == c
        ]
        for j in OFL_footpoints:
            if np.shape(field_lines[j])[0] > 0:
                fl_cart = CartesianRepresentation(x=field_lines[j].T)
                if c == -1:
                    ax.plot(
                        fl_cart.x,
                        fl_cart.z,
                        color=cl_closed,
                        lw=lw_closed,
                        zorder=2,
                    )
                else:
                    fl_before_lc, fl_after_lc = tls.trim_field_line(
                        fl_cart, R_LC, h_LC_coeff=cap_heights[c]
                    )
                    ax.plot(
                        fl_before_lc.x,
                        fl_before_lc.z,
                        color=cl_open_fl[c],
                        lw=lw_open[c],
                        zorder=3,
                    )
                    ax.plot(
                        fl_after_lc.x,
                        fl_after_lc.z,
                        color=cl_closed,
                        lw=lw_closed,
                        zorder=2,
                    )

    for c, cap_height in enumerate(cap_heights):
        left = -R_LC
        bottom = -cap_height * R_LC
        width = 2 * R_LC
        height = 2 * cap_height * R_LC
        p = plt.Rectangle(
            (left, bottom),
            width,
            height,
            fill=False,
            color=cl_open_lc[c],
            ls="-",
            lw=2.5 - c / 2,
            zorder=5 - c,
        )
        ax.add_patch(p)
        ax.axis("off")

    lim = R_LC
    k = 1.2
    ax.set_xlim(-lim * k, lim * k)
    ax.set_ylim(-lim * 2.6 * k, lim * 2.6 * k)
    # ax.set_xlabel("y, $R_\mathrm{NS}$")
    # ax.set_ylabel("z, $R_\mathrm{NS}$")
    if savefig:
        plt.savefig(
            "/home/bilous/Research/2024/Multipole_RVM/tex/figures/02_OFL_BQ_field_lines.pdf",
            dpi=150,
        )
    else:
        plt.show()

import sys
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from psr_magpy import *

def create_field_config():


   
    dipole_config = FieldComponentConfig(
        moment_type="dipole",
        B_magnitude=1, 
        moment_orientation_latitude_degree=90.0, 
    )
    

    
    quadrupole_config = FieldComponentConfig(
        moment_type="quadrupole",
         B_magnitude=1, 
        moment_orientation_latitude_degree=90, 
    )
    field_config = FieldFactoryConfig(components=[
        [dipole_config], 
        [quadrupole_config, dipole_config],
        ]
    )
  
    return field_config

def create_grid_config():
    seed_grid = SphericalGrid(nlon=2, nlat=30)      
        
    #seed_grid = HealpixGrid(nside=32)
    if seed_grid.footpoint_coordinates is None:
        raise ValueError("Grid coordinates not initialized")
    
    #longitude_mask = (seed_grid.longitude > 0) & (seed_grid.longitude < 360)
    #latitude_mask = (seed_grid.colatitude > 89.9) & (seed_grid.colatitude < 90.1)
    #latitude_mask = (seed_grid.colatitude > 89.6) & (seed_grid.colatitude < 90.4)
    #mask = longitude_mask & latitude_mask
    #mask = latitude_mask
    #mask = (seed_grid.colatitude > 0.0) & (seed_grid.colatitude < 0.4)
    coords = seed_grid.footpoint_coordinates
    masked_coords = coords#[mask]
    if masked_coords is None:
        raise ValueError("Failed to mask coordinates")
        
    npoints = len(masked_coords.x)
    # Now we can pass CartesianRepresentation directly
    grid_config = CustomGridConfig(nline_ori=npoints, coordinates=masked_coords)
    return grid_config


if __name__ == "__main__":

   
    field_factory = create_field_config()
    #grid = HealpixGrid(nside=32)
    #grid = create_grid_config()
    #grid_factory = GridFactoryConfig(grids=[grid])
    grid_factory = GridFactoryConfig(grids=[HealpixGridFactoryConfig(nside=32)])

    tracing_config = TracingFactoryConfig(
        max_integration_radius=10,
        save_results_to_disk=True,
        max_memory_mb=100,
        skip_memory_estimation=True,
        max_step=1.0,
        field_backend="numba",
        solver_backend="cyrk",
    )

    # Create experiment series config
    series_config = ExperimentSeriesConfig(
        name="PA_check",
        field=field_factory,
        grid=grid_factory,
        tracing=tracing_config,
        template=FieldLine.template_all(),
        description="Check PA calculation",
    )

    series_result = run_experiment_series(
        series_config=series_config,
        save_individual_configs=True,
        save_grids=True,
        continue_on_failure=True,
    )