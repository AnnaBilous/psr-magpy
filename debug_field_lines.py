import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import logging

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

def analyze_field_lines(field_lines: List[np.ndarray]):
    """Analyze x-components of field lines."""
    max_x_components = []
    for line in field_lines:
        max_x = np.max(np.abs(line[:, 1]))
        max_x_components.append(max_x)
        logger.debug(f"Max x-component: {max_x:.6f}")
    
    return np.array(max_x_components)

def main():
    # Create dipole field aligned with z-axis (should give field lines in XZ plane)
    dipole = DipoleFieldComponent(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=90.0, 
        moment_orientation_longitude_degree=0.0
    )
    field = MultipoleField([dipole])
    
    # Initialize field line tracer
    tracer = FieldLineTracer(field, R_max=2.0, dt=0.25)
    
    # Create minimal spherical grid (2 longitudes = XZ plane)
    grid = SphericalGrid(nlon=2, nlat=20, pole_lat=90.0)
    footpoints = grid.footpoint_coordinates
    
    # Print footpoint coordinates for verification
    xyz = footpoints.xyz.value
    logger.info("\nFootpoint coordinates:")
    for i in range(len(xyz[0])):
        logger.info(f"Point {i}: x={xyz[0,i]:.6f}, y={xyz[1,i]:.6f}, z={xyz[2,i]:.6f}")
    
    # Create template for field line properties
    template = FieldLine(save_r_line=True)
    
    # Calculate field lines
    field_lines = []
    for i, footpoint in enumerate(footpoints):
        line = tracer.trace_single_line(footpoint, i, template)
        if line.r_line is not None and len(line.r_line) > 0:
            field_lines.append(line.r_line)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 3D view
    ax1 = fig.add_subplot(121, projection="3d")
    # Plot field lines
    for line in field_lines:
        ax1.plot(line[:, 0], line[:, 1], line[:, 2], "b-", alpha=0.5)
    
    # Plot NS surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color="r", alpha=0.1)
    
    # Plot footpoints
    ax1.scatter(xyz[0], xyz[1], xyz[2], c='black', s=20, alpha=1.0)
    
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D View")
    
    # Plot XZ plane view
    ax2 = fig.add_subplot(122)
    for line in field_lines:
        ax2.plot(line[:, 0], line[:, 2], "b.-", alpha=0.5)
    
    # Plot unit circle in XZ plane
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.sin(theta), np.cos(theta), 'r-', alpha=0.3)
    
    # Plot footpoints in XZ plane
    ax2.scatter(xyz[0], xyz[2], c='black', s=20, alpha=1.0)
    
    ax2.set_aspect('equal')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title("XZ Plane View")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 