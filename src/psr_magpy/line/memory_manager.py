import glob
import os
import psutil
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from multiprocessing import Pool

import numpy as np

from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.io import FieldLineSetFitsIO
from psr_magpy.line.tracer import FieldLineTracer
from psr_magpy.line.tracing_parameters import PerformanceParameters
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class MemoryManager:
    """Manages memory usage and result storage for field line calculations."""

    def __init__(self, config: PerformanceParameters, tracer: FieldLineTracer, max_file_size_mb: float = 100.0, save_results: bool = True) -> None:
        """Initialize memory manager."""
        self.config = config
        self.tracer = tracer
        self.max_file_size_mb = max_file_size_mb
        self.save_results = save_results
        self._results_in_memory: Optional[FieldLineSet] = None
        self._results_in_memory_size: float = 0.0
        
        logger.debug(f"Initialized MemoryManager with max_file_size_mb={max_file_size_mb}, save_results={save_results}")
        
        if not save_results:
            logger.info("Disk writing disabled - all results will be kept in memory")
        elif max_file_size_mb <= 0:
            logger.warning("max_file_size_mb <= 0 - results will accumulate in memory indefinitely")

    def __repr__(self) -> str:
        """String representation of MemoryManager."""
        memory_mb = self._results_in_memory_size / (1024 * 1024)
        result_count = self._results_in_memory.n_field_lines if self._results_in_memory else 0
        return (
            f"MemoryManager(config={self.config}, "
            f"results_size={memory_mb:.2f} MB, "
            f"field_lines={result_count})"
        )

    def estimate_memory_requirements(
        self, 
        template: FieldLine, 
        grid: FootpointGrid, 
        n_trial: int = 10
    ) -> Tuple[float, float]:
        """Estimate memory requirements using simplified heuristics."""
        logger.info(f"Estimating memory requirements with {n_trial} trial lines")
        time_start = time.time()
        
        if grid.nline is None:
            error_msg = "Grid nline is None, cannot estimate memory"
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_trial = min(grid.nline, n_trial)
        
        # Process trial batch
        trial_indices = np.arange(n_trial)
        trial_grid = grid.subset_grid(trial_indices)
        trial_results = self._process_trial_batch(trial_grid, template)
        
        # Calculate size per line
        trial_size = trial_results.estimate_size() / n_trial
        trial_size_mb = trial_size / (1024 * 1024)
        
        # Estimate completion time
        elapsed = time.time() - time_start
        completion_time = elapsed * grid.nline / n_trial
        
        logger.info(f"Estimated {trial_size_mb:.2f} MB per line, {completion_time/60:.1f} min total")
        return trial_size, completion_time

    def calculate_batch_size(self, estimated_line_size: float) -> int:
        """Calculate optimal batch size using simplified heuristics."""
        # Simple heuristic: use 50% of available memory or max_batch_size, whichever is smaller
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        safe_memory_mb = min(available_mb * 0.5, self.config.max_memory_mb)
        
        estimated_line_size_mb = estimated_line_size / (1024 * 1024)
        batch_size = int(safe_memory_mb / estimated_line_size_mb)
        batch_size = min(batch_size, self.config.max_batch_size)
        
        logger.info(f"Batch size: {batch_size} lines ({batch_size * estimated_line_size_mb:.1f} MB)")
        return batch_size

    def add_results(self, batch_results: FieldLineSet, filename: str, batch_idx: int) -> FieldLineSet:
        """Add batch results and manage memory."""
        logger.debug(f"Adding {batch_results.n_field_lines} field lines to memory")
        
        if self._results_in_memory is None:
            self._results_in_memory = batch_results
        else:
            self._results_in_memory.append(batch_results)

        self._results_in_memory_size = self._results_in_memory.estimate_size()
        memory_mb = self._results_in_memory_size / (1024 * 1024)
        
        # Only write to disk if save_results is True and file size exceeds threshold
        if self.save_results and memory_mb > self.max_file_size_mb:
            logger.info(f"File size threshold reached ({memory_mb:.1f} MB > {self.max_file_size_mb} MB), writing to disk")
            self._write_results_to_disk(filename, batch_idx)
        elif not self.save_results and memory_mb > self.max_file_size_mb:
            logger.warning(f"Memory usage ({memory_mb:.1f} MB) exceeds threshold ({self.max_file_size_mb} MB) but save_results=False - keeping in memory")

        return self._results_in_memory

    def _process_trial_batch(
        self, 
        subset_grid: FootpointGrid, 
        template: FieldLine
    ) -> FieldLineSet:
        """Process trial batch for memory estimation."""
        if subset_grid.nline is None:
            error_msg = "Subset grid nline is None"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if self.config.n_cores is None or self.config.n_cores == 1:
            # Single-threaded
            field_lines = []
            coordinates = subset_grid.footpoint_coordinates
            indices = subset_grid.indices
            assert coordinates is not None and indices is not None  # Type assertion
            for i in range(subset_grid.nline):
                r_initial = coordinates[i]
                footpoint_id = indices[i]
                assert r_initial is not None  # Type assertion for individual coordinate
                field_line = self.tracer.trace_single_line(r_initial, footpoint_id, template)
                field_lines.append(field_line)
            return FieldLineSet(field_lines)
        else:
            # Multi-threaded
            coordinates = subset_grid.footpoint_coordinates
            indices = subset_grid.indices
            assert coordinates is not None and indices is not None  # Type assertion
            with Pool(processes=self.config.n_cores) as pool:
                args = [
                    (coordinates[i], indices[i], template)
                    for i in range(subset_grid.nline)
                ]
                field_lines = pool.starmap(self.tracer.trace_single_line, args)
                return FieldLineSet(field_lines)

    def _write_results_to_disk(self, filename: str, batch_idx: int) -> None:
        """Write results to disk and reset memory."""
        if self._results_in_memory is None:
            return
        
        # Create filename with batch info
        batch_filename = f"{filename}_batch_{batch_idx:03d}.fits"
        
        # Write to FITS
        io_handler = FieldLineSetFitsIO(self._results_in_memory)
        io_handler.save_to_fits(batch_filename)
            
        logger.info(f"Wrote {self._results_in_memory.n_field_lines} field lines to {batch_filename}")
        
        # Reset memory
        self._results_in_memory = None
        self._results_in_memory_size = 0.0

    def rename_batch_files(self, filename: str) -> None:
        """Rename batch files to final format."""
        pattern = f"{filename}_batch_*.fits"
        batch_files = glob.glob(pattern)
        
        if not batch_files:
            logger.warning(f"No batch files found matching {pattern}")
            return

        logger.info(f"Found {len(batch_files)} batch files to rename")
        
        # For now, just log the files - actual renaming logic can be added later
        for batch_file in sorted(batch_files):
            logger.debug(f"Batch file: {batch_file}")

    @property
    def results_in_memory(self) -> Optional[FieldLineSet]:
        """Get results currently in memory."""
        return self._results_in_memory

    @property
    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self._results_in_memory_size / (1024 * 1024)
