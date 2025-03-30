#!/usr/bin/env python
"""Demo for multipole magnetic field visualization."""

from pathlib import Path

# Import directly from the package
from psr_magpy.cli.setup_environment import setup_environment

# Set up environment
env = setup_environment(output_dir=Path(__file__).parent / "output")
logger = env['logger']
fits_dir = env['directories']['fits']
plot_dir = env['directories']['plots']
log_dir = env['directories']['logs']

import numpy as np
import matplotlib.pyplot as plt
from src.psr_magpy.field.components import DipoleFieldComponent, QuadrupoleFieldComponent
from src.psr_magpy.field.multipole import MultipoleField

def plot_field_lines_2d(field, title, save_path=None, show=True):
    """Plot magnetic field lines in the x-z plane.
    
    Parameters
    ----------
    field : MultipoleField
        Field to plot
    title : str
        Plot title
    save_path : Path, optional
        Path to save the plot
    show : bool, optional
        Whether to display the plot, by default True
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid of starting points in x-z plane
    x = np.linspace(-2, 2, 20)
    z = np.linspace(-2, 2, 20)
    X, Z = np.meshgrid(x, z)
    
    # Calculate field at each point
    B_field = np.zeros((len(x), len(z), 3))
    for i in range(len(x)):
        for j in range(len(z)):
            r = np.array([X[i,j], 0, Z[i,j]])
            B_field[i,j] = field.get_total_field(r)
    
    # Plot streamlines
    ax.streamplot(X, Z, B_field[:,:,0], B_field[:,:,2], 
                 density=2, color='b', linewidth=1, arrowsize=1)
    
    # Plot unit circle representing star surface
    circle = plt.Circle((0, 0), 1, color='r', fill=False)
    ax.add_artist(circle)
    
    # Customize plot
    ax.set_xlabel('X (NS radii)')
    ax.set_ylabel('Z (NS radii)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
    if show:
        plt.show()
    
    return fig, ax

def main():
    """Run the multipole field demo."""
    # 1. Demonstrate dipole field
    logger.info("Creating dipole field")
    dipole = DipoleFieldComponent(
        moment_orientation_latitude_degree=45.0,
    )
    dipole_field = MultipoleField([dipole])
    plot_field_lines_2d(
        dipole_field,
        "Dipole Field (45° tilt)",
        plot_dir / "dipole_field.png"
    )
    
    # 2. Demonstrate quadrupole field
    logger.info("Creating quadrupole field")
    quadrupole = QuadrupoleFieldComponent(
         moment_orientation_latitude_degree=90.0,
    )
    quadrupole_field = MultipoleField([quadrupole])
    plot_field_lines_2d(
        quadrupole_field,
        "Quadrupole Field",
        plot_dir / "quadrupole_field.png"
    )
    
    # 3. Demonstrate combined field
    logger.info("Creating combined dipole + quadrupole field")
    # Reduce quadrupole strength for visualization
    
    combined_field = MultipoleField([dipole, quadrupole])
    plot_field_lines_2d(
        combined_field,
        "Combined Field (Dipole + 0.3 Quadrupole)",
        plot_dir / "combined_field.png"
    )
    
    # 4. Demonstrate offset dipole
    logger.info("Creating offset dipole field")
    offset_dipole = DipoleFieldComponent(
        moment_orientation_latitude_degree=45.0,
        offset_r_ns=np.array([0.2, 0.0, 0.0])  # Offset in x direction
    )
    offset_field = MultipoleField([offset_dipole])
    plot_field_lines_2d(
        offset_field,
        "Offset Dipole Field (0.2 R_NS in x)",
        plot_dir / "offset_dipole.png"
    )
    
    # Show plots if running interactively
    plt.show()

if __name__ == "__main__":
    try:
        main()
        logger.info("Demo completed successfully")
    except Exception as e:
        logger.exception("Demo failed with error")
        raise
