import sys
from pathlib import Path
from setup_directories import setup_output_directories

# Add project root to PYTHONPATH only if not already present
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from setup_directories import setup_output_directories

import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.field.components import (
    DipoleFieldComponent,
    QuadrupoleFieldComponent
)
from src.field.multipole import MultipoleField

# Create output directories relative to script location
_, fits_dir, plot_dir, log_dir = setup_output_directories(__file__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_field_grid(x_range: tuple, z_range: tuple, n_points: int = 20):
    """Create a 2D grid of points for field visualization in OX plane."""
    x = np.linspace(*x_range, n_points)
    z = np.linspace(*z_range, n_points)
    
    X, Z = np.meshgrid(x, z)
    # Create points with y=0 for OX plane
    points = np.zeros((X.size, 3))
    points[:, 0] = X.flatten()  # X coordinates
    points[:, 2] = Z.flatten()  # Z coordinates
    return points, (X, Z)

def plot_field_lines(ax, field: MultipoleField, grid_points, scale: float = 0.5):
    """
    Plot magnetic field lines as arrows in the OX plane (2D view).
    
    Args:
        ax: Matplotlib axis for 2D plot
        field: MultipoleField instance
        grid_points: Array of points where to evaluate the field
        scale: Scaling factor for arrow sizes
    """
    positions = grid_points.reshape(-1, 3)
    field_vectors = np.array([field.get_total_field(pos) for pos in positions])
    
    # We only need X and Z components for 2D visualization
    X = positions[:, 0].reshape(-1)
    Z = positions[:, 2].reshape(-1)
    U = field_vectors[:, 0].reshape(-1)  # X component
    W = field_vectors[:, 2].reshape(-1)  # Z component
    
    # Calculate magnitude for color mapping
    magnitude = np.sqrt(U**2 + W**2)
    
    # Plot arrows
    ax.quiver(X, Z, U, W,
              magnitude,  # Color arrows by field strength
              scale=scale,
              cmap='viridis',
              width=0.003,
              alpha=0.8)

def main():
    # Create field components
    dipole = DipoleFieldComponent(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=30.0,
        moment_orientation_longitude_degree=0.0,
        offset_r_ns=np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    logger.info("Created dipole component: %s", dipole)

    quadrupole = QuadrupoleFieldComponent(
        B_magnitude=0.5,
        moment_orientation_latitude_degree=60.0,
        moment_orientation_longitude_degree=0.0,
        offset_r_ns=np.array([0.1, 0.1, 0.0], dtype=np.float32)
    )
    logger.info("Created quadrupole component: %s", quadrupole)

    # Create multipole field
    field = MultipoleField([dipole, dipole, dipole])    

    # Save field configuration to the fits directory
    fits_path = fits_dir / 'field_configuration.fits'
    field.save_to_fits(fits_path)
    logger.info("Field configuration saved to '%s'", fits_path)

    # Load field from FITS and verify
    loaded_field = MultipoleField.load_from_fits(fits_path)
    logger.info("Loaded field configuration from FITS")

    # Create visualization grid
    grid_points, (X, Z) = create_field_grid(
        x_range=(-2, 2),
        z_range=(-2, 2),
        n_points=20
    )

    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original field
    plot_field_lines(ax1, field, grid_points, scale=30)
    circle1 = plt.Circle((0, 0), 1, color='blue', fill=True, alpha=0.2)
    ax1.add_patch(circle1)
    ax1.set_xlabel('X (NS radii)')
    ax1.set_ylabel('Z (NS radii)')
    ax1.set_title('Original Field Configuration')
    ax1.set_aspect('equal')
    plt.colorbar(ax1.collections[0], label='Field Strength')

    # Plot loaded field
    plot_field_lines(ax2, loaded_field, grid_points, scale=30)
    circle2 = plt.Circle((0, 0), 1, color='blue', fill=True, alpha=0.2)
    ax2.add_patch(circle2)
    ax2.set_xlabel('X (NS radii)')
    ax2.set_ylabel('Z (NS radii)')
    ax2.set_title('Loaded Field Configuration')
    ax2.set_aspect('equal')
    plt.colorbar(ax2.collections[0], label='Field Strength')

    # Save comparison plot
    plot_path = plot_dir / 'magnetic_field_configuration_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Comparison plot saved as '%s'", plot_path)

if __name__ == "__main__":
    main() 