import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import logging
from astropy.coordinates import CartesianRepresentation
import astropy.units as u

# Suppress matplotlib debug outputs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from FieldLineTracer import FieldLineTracer
from MultipoleField import MultipoleField
from FootpointGrid import SphericalGrid
from FieldComponent import DipoleFieldComponent
from FieldLineDefinition import FieldLine

# Set up logging for our debug script
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def find_target_footpoint(footpoints):
    """Find footpoint with smallest positive x and largest z."""
    xyz = footpoints.xyz.value
    
    # Get points with positive x
    pos_x_mask = xyz[0] > 0
    pos_x_points = xyz[:, pos_x_mask]
    
    if len(pos_x_points[0]) == 0:
        raise ValueError("No points with positive x found")
    
    # Find point with smallest x among positive x points
    max_z_idx = np.argmax(pos_x_points[2])
    target_point = pos_x_points[:, max_z_idx]
    
    logger.info(f"\nTarget footpoint coordinates:")
    logger.info(f"x={target_point[0]:.6f}, y={target_point[1]:.6f}, z={target_point[2]:.6f}")
    
    # Find original index
    original_idx = np.where((xyz[0] == target_point[0]) & 
                          (xyz[1] == target_point[1]) & 
                          (xyz[2] == target_point[2]))[0][0]
    
    return original_idx

def plot_field_directions(ax, line_points, field, scale=0.2):
    """Plot magnetic field direction vectors at each point of the field line.
    
    Args:
        ax: Matplotlib axis (3D)
        line_points: Array of points along field line
        field: MultipoleField instance
        scale: Length of direction vectors (default 0.2)
    """
    for point in line_points:
        # Convert point to CartesianRepresentation
        point_cart = CartesianRepresentation(
            x=point[0] * u.dimensionless_unscaled,
            y=point[1] * u.dimensionless_unscaled,
            z=point[2] * u.dimensionless_unscaled
        )
        
        # Get magnetic field direction at this point
        B = field.get_total_field(point_cart.xyz, K=1, normalize=False)
        B_xyz = B
        
        # Normalize to get unit vector
        B_norm = np.linalg.norm(B_xyz)
        if B_norm > 0:
            B_unit = B_xyz / B_norm
            
            # Plot direction vector
            ax.quiver(point[0], point[1], point[2],
                     B_unit[0], B_unit[1], B_unit[2],
                     length=scale, color='red', alpha=0.5)

def main():
    # Create dipole field aligned with z-axis
    dipole = DipoleFieldComponent(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=90.0,
        moment_orientation_longitude_degree=0.0
    )
    field = MultipoleField([dipole])
    
    # Initialize field line tracer with detailed logging
    tracer = FieldLineTracer(field, R_max=1.2, dt=0.1)
    
    # Create grid and get footpoints
    #grid = SphericalGrid(nlon=2, nlat=20, pole_lat=0.0)
    #footpoints = grid.footpoint_coordinates
    
    # Find our target footpoint
    #target_idx = find_target_footpoint(footpoints)
    #target_footpoint = footpoints[target_idx]
    #print("target_footpoint: ", target_footpoint)
    
    #target_footpoint = CartesianRepresentation(y=0.0, x=1.0/np.sqrt(2), z=1.0/np.sqrt(2))
    target_footpoint = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
    print("target_footpoint: ", target_footpoint)
    # Create template
    template = FieldLine(save_r_line=True)
    
    # Trace single line with detailed logging
    logger.info("\nStarting field line trace:")
    logger.info("Initial conditions:")
    logger.info(f"Starting position: {target_footpoint.xyz.value}")
    
    # Trace the line
    line = tracer.trace_single_line(target_footpoint, 0, template)
    
    if line.r_line is not None:
        logger.info(f"\nField line traced successfully:")
        logger.info(f"Number of points: {len(line.r_line)}")
        logger.info(f"Final position: {line.r_line[-1]}")
        
        # Analyze y-components
        y_components = line.r_line[:, 1]
        max_y = np.max(np.abs(y_components))
        logger.info(f"\nMaximum absolute y-component: {max_y:.6f}")
        
        # Plot the results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(line.r_line[:, 0], line.r_line[:, 1], line.r_line[:, 2], 'b-')
        
        # Add field direction vectors
        plot_field_directions(ax1, line.r_line, field)
        
        # Plot unit sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x, y, z, color='r', alpha=0.1)
        
        # Plot starting point
        ax1.scatter(target_footpoint.xyz.value[0], 
                   target_footpoint.xyz.value[1], 
                   target_footpoint.xyz.value[2], 
                   c='green', s=100, label='Start')
        
        ax1.set_box_aspect([1,1,1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D View with Field Directions')
        ax1.legend()
        
        # XZ plane plot with field directions projected 
        ax2 = fig.add_subplot(122)
        ax2.plot(line.r_line[:, 0], line.r_line[:, 2], 'b-')  
        # Add field directions in YZ plane
        for point in line.r_line:
            point_cart = CartesianRepresentation(
                x=point[0],
                y=point[1],
                z=point[2]
            )
            B = field.get_total_field(point_cart.xyz, K=1, normalize=False)
            B_xyz = B
            B_norm = np.linalg.norm(B_xyz)
            if B_norm > 0:
                B_unit = B_xyz / B_norm
                ax2.quiver(point[0], point[2],  
                          B_unit[0], B_unit[2], 
                          scale=10, color='red', alpha=0.5)
        
        # Plot unit circle in YZ plane
        theta = np.linspace(0, 2*np.pi, 100)
        ax2.plot(np.sin(theta), np.cos(theta), 'r-', alpha=0.3)
        
        # Plot starting point
        ax2.scatter(target_footpoint.xyz.value[0], 
                   target_footpoint.xyz.value[2], 
                   c='green', s=100, label='Start')
        
        ax2.set_aspect('equal')
        ax2.set_xlabel('X')  # Changed from X
        ax2.set_ylabel('Z')
        ax2.set_title('XZ Plane View with Field Directions')  # Changed from XZ
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    else:
        logger.error("Field line tracing failed")

if __name__ == "__main__":
    main() 