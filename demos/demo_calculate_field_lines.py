import sys
from pathlib import Path
from setup_directories import setup_output_directories
from datetime import datetime

# Add project root to PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.logging_config import setup_logging, LogLevels
import logging

from src.field.multipole import MultipoleField
from src.field.components import DipoleFieldComponent, QuadrupoleFieldComponent
from src.footpoint_grid.grid import HealpixGrid
from src.line.field_line import FieldLine
from src.line.calculator import FieldLineCalculator, TracingConfig
from src.line.field_line import PropertyName
from src.line.field_line_set import FieldLineSet
from src.line.io import FieldLineSetFitsIO

# Create output directories
_, fits_dir, plot_dir, log_dir = setup_output_directories(__file__)

# Setup logging with different levels for console and file
log_file = log_dir / f"field_lines_{datetime.now():%Y%m%d_%H%M%S}.log"
setup_logging(
    log_level='DEBUG',  # Base level
    log_file=log_file,
    handler_levels={
        'console': LogLevels.CONSOLE,  # Less verbose for console
        'file': LogLevels.FILE  # More detailed for file
    },
    max_bytes=50 * 1024 * 1024,  # 50MB per file
    backup_count=3  # Keep 3 backup files
)

logger = logging.getLogger(__name__)
logger.info("Starting field line calculation demo")

def plot_field_lines(field_lines, save_path=None):
    """Plot the calculated field lines in 3D."""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    r_lines = field_lines.get_property("r_line")
    
    # Plot field lines
    for r_line in r_lines:
        if r_line is not None and len(r_line) > 0:
            ax.plot(r_line[:, 0], r_line[:, 1], r_line[:, 2], 
                   'b-', alpha=0.3)

    # Plot unit sphere (star surface)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='r', alpha=0.1)

    # Customize plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Magnetic Field Lines')
    ax.set_box_aspect([1, 1, 1])
    plt.show()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    # Generate unique filename based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(fits_dir, f"field_lines_{timestamp}")

    # Configure tracing parameters
    config = TracingConfig(
        R_max=10.0,          # Maximum radius for tracing
        R_min=1.0,           # Minimum radius (star surface)
        dt=0.01,             # Integration time step
        max_memory_mb=50.0, # Maximum memory usage per batch
        n_cores=2,           # Number of CPU cores to use
        save_results=True,   # Save results to disk
        filename=output_filename
    )

    # Create magnetic field configuration
    dipole = DipoleFieldComponent(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=0.0,
        moment_orientation_longitude_degree=0.0
    )
    
    quadrupole = QuadrupoleFieldComponent(
        B_magnitude=0.5,
        moment_orientation_latitude_degree=45.0,
        moment_orientation_longitude_degree=0.0
    )
    
    field = MultipoleField([dipole, quadrupole])

    # Create footpoint grid using HEALPix
    grid = HealpixGrid(nside=64)
    logger.info(f"Number of field lines to calculate: {grid.nline}")
    
    # Configure which field line properties to save
    template = FieldLine(
        save_r_max=True,           # Save maximum radius reached
        save_b_r_ini_angle=True,   # Save initial B_r angle
        save_r_line=True           # Save full field line coordinates
    )
    logger.info(f"Template configuration: {template}")

    # Initialize calculator
    calculator = FieldLineCalculator(field, grid)
    
    try:
        # Calculate field lines
        logger.info("Starting field line calculations...")
        results = calculator.get_field_lines(
            template=template,
            config=config
        )

        if config.save_results:
            logger.info(f"Results saved to: {output_filename}")
            
            # Load the saved results and create visualization
            logger.info("Loading saved results for verification...")
            loaded_field_lines = FieldLineSetFitsIO.load_from_fits(output_filename)
            logger.info(f"Successfully loaded {loaded_field_lines.n_field_lines} field lines")
            
            # Print information about loaded data
            logger.info("Loaded field line properties:")
            for prop in loaded_field_lines.save_flags.__dict__:
                if getattr(loaded_field_lines.save_flags, prop):
                    logger.info(f"  - {prop}")
            
            # Create visualization of loaded data
            plot_path = os.path.join(plot_dir, f"field_lines_{timestamp}.png")
            #plot_field_lines(loaded_field_lines, save_path=plot_path)
            logger.info(f"Plot saved to: {plot_path}")
            
    except MemoryError:
        logger.error("Memory limit exceeded during calculation. Try reducing max_memory_mb or nside.")
        raise
    except Exception as e:
        logger.error(f"Error during field line calculation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 