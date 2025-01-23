import time
import datetime
import numpy as np
from dataclasses import dataclass

from Constants import R_NS
import WriteToFits as io
from FieldLineTracer import FieldLineTracer
from FootpointGrid import FootpointGrid
from field_line_types import FieldLine

import matplotlib.pyplot as plt
from MultipoleField import MultipoleField
from FootpointGrid import FootpointGrid, GridType, SphericalGridParams
from FieldComponent import FieldComponent, MultipoleType
from multiprocessing import Pool
import cProfile


class FieldLineCalculator:
    def __init__(self, field: MultipoleField, grid: FootpointGrid):
        """Initialize field line calculator.
        
        Parameters
        ----------
        field : MultipoleField
            Object that calculates the magnetic field
        """
        self.field = field
        self.grid = grid
      
    def get_field_lines(self, 
                       template: FieldLine,
                       filename: str = "", 
                       R_max: float = 10.0, 
                       R_min: float = 1.0, 
                       dt: float = 0.01):
     
        # Initialize field line tracer
        tracer = FieldLineTracer(self.field, R_max=R_max, R_min=R_min)
     
        footpoints = self.grid.footpoint_coordinates
        
        # Create list of arguments for each footpoint
        args = [(footpoint, template) for footpoint in footpoints]
    
        # Parallel processing of field lines
        with Pool() as pool:
            results = pool.starmap(tracer.trace_single_line, args)

        #field_line_tracer = FieldLineTracer(self.field, R_max, R_min)
        #args = [(footpoint, template) for footpoint in footpoints]
        return results

if __name__ == "__main__":

    
    # Create magnetic field (example with dipole)
    dipole = FieldComponent(moment_type=MultipoleType.DIPOLE, B_magnitude=1.0, moment_orientation_latitude_degree=0.0)
    quadrupole = FieldComponent(moment_type=MultipoleType.QUADRUPOLE, B_magnitude=3.0)
    field = MultipoleField([dipole]) #, quadrupole])  

   
    # Create footpoints using FootpointGrid
    params_standard = SphericalGridParams(nlon=4, nlat=11)
    grid = FootpointGrid(grid_type=GridType.SPHERICAL, params=params_standard)
    footpoints = grid.footpoint_coordinates
    
    # Variables to save
    template = FieldLine.create_template(save_r_line=True)
    
    calculator = FieldLineCalculator(field, grid)
    results =calculator.get_field_lines(template, R_max=2.0)
    
    # Create list of arguments for each footpoint
    args = [(footpoint, template) for footpoint in footpoints]

    # Filter and collect results
    field_lines = [a.r_line for a in results] # if a.r_line # is not None and len(a.r_line) > 0]
       
    
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
    
    
