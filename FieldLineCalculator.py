import time
import datetime
import numpy as np

import psutil
from typing import Tuple
import os

import WriteToFits as io
from FieldLineTracer import FieldLineTracer
from FootpointGrid import FootpointGrid
from FieldLineDefinition import FieldLine, FieldLineSet

import matplotlib.pyplot as plt
from MultipoleField import MultipoleField
from FootpointGrid import (
    FootpointGrid,
    GridType,
    SphericalGridParams,
    HealpixGridParams,
)
from FieldComponent import FieldComponent, MultipoleType
from multiprocessing import Pool
from ResultsManager import ResultsManager, MemoryConfig


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

    def get_field_lines(
        self,
        template: FieldLine,
        filename: str = "",
        R_max: float = 10.0,
        R_min: float = 1.0,
        dt: float = 0.01,
        max_memory_mb: float = 100.0,
        n_cores: int = None,
        save_results: bool = False,
    ):
        """Calculate field lines in batches and save to FITS file."""
        start_time = time.time()
        # Initialize components
        tracer = FieldLineTracer(self.field, R_max, R_min, dt)
        # Initialize results manager
        memory_config = MemoryConfig(max_memory_mb=max_memory_mb, n_cores=n_cores)
        results_manager = ResultsManager(memory_config, tracer)

        self._validate_inputs(filename, save_results)

        batch_size = results_manager.find_batch_size(template)
        batch_size = min(batch_size, self.grid.nline)

        # Process in batches
        all_indices = np.arange(self.grid.nline)
        for start_idx in range(0, self.grid.nline, batch_size):

            # Create subset grid
            batch_indices = all_indices[start_idx : start_idx + batch_size]
            subset_grid = self.grid.create_subset_grid(batch_indices)

            batch_results = self.process_batch(subset_grid, template, tracer, n_cores)

            # Let results manager handle the results
            results = results_manager.add_results(batch_results, filename, start_idx)

            # Print memory stats
            stats = results_manager.get_memory_stats()
            print(
                f"Memory usage: {stats['percent_used']}% "
                f"(Available: {stats['available_gb']:.1f} GB)"
            )

        return None if save_results else results

    def process_batch(
        self,
        subset_grid: FootpointGrid,
        template: FieldLine,
        tracer: FieldLineTracer,
        n_cores: int = None,
    ) -> list:

        batch_args = [
            (subset_grid.footpoint_coordinates[idx], subset_grid.indices[idx], template)
            for idx in range(subset_grid.nline)
        ]

        # Process batch in parallel
        with Pool(n_cores) as pool:
            batch_results = pool.starmap(tracer.trace_single_line, batch_args)

        return batch_results

    def _validate_inputs(self, filename: str, save_results: bool) -> None:
        """Validate input parameters.

        Args:
            filename: Output file path
            save_results: Whether to save results to file

        Raises:
            ValueError: If inputs are invalid
        """
        if save_results and not filename:
            raise ValueError("Filename must be specified when save_results is True")

        if filename:
            # Check if directory exists
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                raise ValueError(f"Output directory does not exist: {directory}")

            # Check file extension
            if not filename.endswith(".fits"):
                raise ValueError("Output filename must have .fits extension")


if __name__ == "__main__":
    # Configuration
    output_filename = "field_lines.fits"
    memory_config = MemoryConfig(
        max_memory_mb=20.0,
        safety_factor=0.8,  # Use 80% of available memory
        min_batch_size=10,
    )

    # Create magnetic field components
    dipole = FieldComponent(
        moment_type=MultipoleType.DIPOLE,
        B_magnitude=1.0,
        moment_orientation_latitude_degree=90.0,
    )
    quadrupole = FieldComponent(moment_type=MultipoleType.QUADRUPOLE, B_magnitude=3.0)
    field = MultipoleField([dipole, quadrupole])

    # Create footpoint grid
    params_healpix = HealpixGridParams(nside=32)
    grid = FootpointGrid(grid_type=GridType.HEALPIX, params=params_healpix)

    # Configure field line template
    template = FieldLine(save_r_max=True, save_b_r_ini_angle=True, save_r_line=True)

    # Initialize calculator
    calculator = FieldLineCalculator(field, grid)

    try:
        # Calculate field lines with memory management
        results = calculator.get_field_lines(
            template=template,
            filename=output_filename,
            R_max=100,
            R_min=1.0,
            dt=0.01,
            max_memory_mb=memory_config.max_memory_mb,
            n_cores=4,
            save_results=False,
        )

        # Visualize results if available
        if results and results.field_lines:
            # Filter valid field lines
            field_lines = [
                line.r_line
                for line in results.field_lines
                if line.r_line is not None and len(line.r_line) > 0
            ]

            if field_lines:
                # Create 3D plot
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection="3d")

                # Plot field lines
                for line in field_lines:
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], "b-", alpha=0.5)

                # Plot NS surface
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color="r", alpha=0.1)

                # Configure plot
                ax.set_box_aspect([1, 1, 1])
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                plt.show()
            else:
                print("No valid field lines to plot")
        else:
            print("No results available for visualization")

    except Exception as e:
        print(f"Error during calculation: {str(e)}")
        raise
