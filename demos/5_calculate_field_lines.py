#!/usr/bin/env python
"""Demo for multipole magnetic field visualization."""

import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments for logging configuration."""
    parser = argparse.ArgumentParser(description="Field line calculation demo with configurable logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for all modules")
    parser.add_argument("--debug-field", action="store_true", help="Enable debug mode for field module only")
    parser.add_argument("--debug-line", action="store_true", help="Enable debug mode for line module only")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output (warnings and errors only)")
    parser.add_argument("--no-filename", action="store_true", help="Don't show filename in logs")
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Import directly from the package
from psr_magpy.cli.setup_environment import setup_environment
from psr_magpy.utils.logging_config import set_debug_for_modules

# Set up environment with debug options
env = setup_environment(
    output_dir=Path(__file__).parent / "output",
    debug_mode=args.debug,
    show_filename=not args.no_filename
)
logger = env["logger"]

# Set module-specific debug levels if requested
if args.debug_field:
    set_debug_for_modules(["psr_magpy.field"])
    logger.info("Debug mode enabled for field module")
    
if args.debug_line:
    set_debug_for_modules(["psr_magpy.line"])
    logger.info("Debug mode enabled for line module")

fits_dir = env["directories"]["fits"]
plot_dir = env["directories"]["plots"]
log_dir = env["directories"]["logs"]

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Use consistent import paths
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.field.components import (
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
)
from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.calculator import FieldLineCalculator, TracingConfig
from psr_magpy.line.io import FieldLineSetFitsIO


def plot_field_lines(field_lines, save_path=None):
    """Plot the calculated field lines in 3D."""
    logger.info("Creating 3D visualization of field lines")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    r_lines = field_lines.get_property("r_line")
    
    # Plot field lines
    line_count = 0
    for r_line in r_lines:
        if r_line is not None and len(r_line) > 0:
            ax.plot(r_line[:, 0], r_line[:, 1], r_line[:, 2], "b-", alpha=0.3)
            line_count += 1
    
    logger.debug(f"Plotted {line_count} field lines")

    # Plot unit sphere (star surface)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="r", alpha=0.1)

    # Customize plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Magnetic Field Lines")
    ax.set_box_aspect([1, 1, 1])

    if save_path:
        logger.info(f"Saving plot to {save_path}")
        plt.savefig(save_path)
    else:
        logger.info("Displaying plot")
        plt.show()


def main():
    # Generate unique filename based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #output_filename = os.path.join(fits_dir, f"field_lines_{timestamp}")
    output_filename = os.path.join(fits_dir, f"field_lines")
    logger.info(f"Output filename: {output_filename}")

    # Configure tracing parameters
    config = TracingConfig(
        R_max=20.0,  # Maximum radius for tracing
        min_R_max_line=2.0,  # Save lines with R_max > min_R_max_line
        dt=0.01,  # Integration time step
        max_memory_mb=5.0,  # Maximum memory usage per batch
        n_cores=4,  # Number of CPU cores to use
        max_batch_size=100,  # Number of field lines to calculate per batch
        save_results=True,  # Save results to disk
        filename=output_filename,
    )
    logger.debug(f"Tracing configuration: {config}")

    # Create magnetic field configuration
    logger.info("Creating magnetic field configuration")
    dipole = DipoleFieldComponent(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=90.0,
        moment_orientation_longitude_degree=0.0,
    )
    logger.debug(f"Dipole component: {dipole}")

    quadrupole = QuadrupoleFieldComponent(
        B_magnitude=0.5,
        moment_orientation_latitude_degree=90.0,
        moment_orientation_longitude_degree=0.0,
    )
    logger.debug(f"Quadrupole component: {quadrupole}")

    #field = MultipoleField([dipole]) #, quadrupole])
    field = MultipoleField([quadrupole])
    logger.debug(f"Combined field: {field}")

    # Create footpoint grid using HEALPix
    logger.info("Creating HEALPix grid for footpoints")
    grid = HealpixGrid(nside=8)
    logger.info(f"Number of field lines to calculate: {grid.nline}")

    # Configure which field line properties to save
    logger.info("Configuring field line properties to save")
    #template = FieldLine(
    #    save_r_max=True,  # Save maximum radius reached
    #    save_b_r_ini_angle=True,  # Save initial B_r angle
    #    save_r_line=True,  # Save full field line coordinates
    #)
    template = FieldLine(save_all=True)
    logger.info(f"Template configuration: {template}")

    # Initialize calculator
    logger.info("Initializing field line calculator")
    calculator = FieldLineCalculator(field, grid)

    try:
        # Calculate field lines
        logger.info("Starting field line calculations...")
        results = calculator.get_field_lines(template=template, config=config)

        if config.save_results:
            logger.info(f"Results saved to: {output_filename}")

            # Load the saved results and create visualization
            logger.info("Loading saved results for verification...")
            loaded_field_lines = FieldLineSetFitsIO.load_from_fits(output_filename)
            logger.info(
                f"Successfully loaded {loaded_field_lines.n_field_lines} field lines"
            )

            # Print information about loaded data
            logger.info("Loaded field line properties:")
            for prop in loaded_field_lines.save_flags.__dict__:
                if getattr(loaded_field_lines.save_flags, prop):
                    logger.info(f"  - {prop}")

            # Create visualization of loaded data
            plot_path = os.path.join(plot_dir, f"field_lines_{timestamp}.png")
            plot_field_lines(loaded_field_lines, save_path=plot_path)
            plt.show()
            logger.info(f"Plot saved to: {plot_path}")

    except MemoryError:
        logger.error(
            "Memory limit exceeded during calculation. Try reducing max_memory_mb or nside."
        )
        raise
    except Exception as e:
        logger.error(f"Error during field line calculation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
