import time
import datetime
import numpy as np

import psutil
from typing import Tuple, List, Optional, Dict, Any
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
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class TracingConfig:
    """Configuration for field line tracing parameters and output settings."""
    R_max: float = 10.0
    R_min: float = 1.0
    dt: float = 0.01
    max_memory_mb: float = 100.0
    n_cores: Optional[int] = None
    save_results: bool = False
    filename: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.save_results and not self.filename:
            raise ValueError("filename must be specified when save_results is True")
        if not self.save_results and self.filename:
            raise ValueError("filename should not be specified when save_results is False")
        
        if self.filename:
            directory = os.path.dirname(self.filename)
            if directory and not os.path.exists(directory):
                raise ValueError(f"Output directory does not exist: {directory}")

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
        config: Optional[TracingConfig] = None,
    ) -> Optional[FieldLineSet]:
        """Calculate field lines in batches and save to FITS file."""
        config = config or TracingConfig()
        start_time = time.time()
        
        tracer = FieldLineTracer(self.field, config.R_max, config.R_min, config.dt)
        memory_config = MemoryConfig(max_memory_mb=config.max_memory_mb, n_cores=config.n_cores)
        results_manager = ResultsManager(memory_config, tracer)

        print(f"Starting field line calculations at {time.strftime('%H:%M:%S')}")
        results = self._process_all_batches(
            template, config.filename or "", tracer, results_manager
        )
        elapsed = time.time() - start_time
        print(f"\nCalculations completed in {elapsed:.1f} seconds")
        return results

    def _process_all_batches(
        self,
        template: FieldLine,
        filename: str,
        tracer: FieldLineTracer,
        results_manager: ResultsManager,
    ) -> Optional[FieldLineSet]:
        batch_size = min(
            results_manager.find_batch_size(template),
            self.grid.nline
        )

        all_indices = np.arange(self.grid.nline)
        results = None
        total_batches = (self.grid.nline + batch_size - 1) // batch_size

        with tqdm(total=self.grid.nline, desc="Processing field lines") as pbar:
            for start_idx in range(0, self.grid.nline, batch_size):
                batch_indices = all_indices[start_idx : start_idx + batch_size]
                subset_grid = self.grid.create_subset_grid(batch_indices)
                
                # Process batch using multiprocessing
                batch_args = [
                    (subset_grid.footpoint_coordinates[idx], subset_grid.indices[idx], template)
                    for idx in range(subset_grid.nline)
                ]
                
                with Pool(results_manager.memory_config.n_cores) as pool:
                    batch_results = pool.starmap(tracer.trace_single_line, batch_args)
                
                results = results_manager.add_results(batch_results, filename, start_idx)
                #self._print_memory_stats(results_manager)
                
                # Update progress bar
                pbar.update(len(batch_indices))
                pbar.set_postfix({
                    'Batch': f"{(start_idx // batch_size) + 1}/{total_batches}",
                    'Lines': f"{min(start_idx + batch_size, self.grid.nline)}/{self.grid.nline}"
                })

        if config.save_results:
            results_manager.rename_batch_files(filename)
            return None            
        else:
            results_set = FieldLineSet(results)
            return results_set

    def _print_memory_stats(self, results_manager: ResultsManager) -> None:
        stats = results_manager.get_memory_stats()
        tqdm.write(
            f"Memory usage: {stats['percent_used']}% "
            f"(Available: {stats['available_gb']:.1f} GB)"
        )

if __name__ == "__main__":
    # Configuration
    output_filename = "field_lines"
    config = TracingConfig(
        R_max=100,
        R_min=1.0,
        dt=0.01,
        max_memory_mb=20.0,
        n_cores=4,
        save_results=True,
        filename=output_filename
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
            config=config
        )


    except Exception as e:
        print(f"Error during calculation: {str(e)}")
        raise
