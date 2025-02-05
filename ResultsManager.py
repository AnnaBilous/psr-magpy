from dataclasses import dataclass
from typing import Optional, List, Tuple
import psutil
import numpy as np
from FieldLineDefinition import FieldLine, FieldLineSet
from FieldLineTracer import FieldLineTracer
import os
import glob
import logging

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    max_memory_mb: float
    safety_factor: float = 0.9  # Leave 10% buffer
    min_batch_size: int = 10  # Minimum batch size to process
    check_interval_sec: float = 1.0  # How often to check memory usage
    n_cores: Optional[int] = None  # Number of CPU cores to use

    def __post_init__(self):
        """Setup number of cores after initialization."""
        self.n_cores = self._setup_cores(self.n_cores)

    def _setup_cores(self, n_cores: Optional[int] = None) -> int:
        """Setup and validate the number of CPU cores to use.

        Args:
            n_cores: Requested number of cores, or None for auto-detection

        Returns:
            Number of cores to use
        """
        if n_cores is None:
            # Try to get physical cores first, fall back to logical cores
            n_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()

        # Ensure we have at least one core
        n_cores = max(1, n_cores)

        # Check against system limits
        max_cores = psutil.cpu_count()
        if n_cores > max_cores:
            logger.warning("Requested %d cores, but only %d available. Using %d cores.", n_cores, max_cores, max_cores)
            n_cores = max_cores

        logger.info("Using %d CPU cores", n_cores)
        return n_cores


class ResultsManager:
    def __init__(self, config: MemoryConfig, tracer: FieldLineTracer):
        """Initialize results manager.

        Args:
            config: Memory management configuration
        """
        self.memory_config = config
        self._total_memory = psutil.virtual_memory().total
        self._results_in_memory: FieldLineSet = FieldLineSet([])  # Initialize as an empty FieldLineSet
        self._results_in_memory_size = 0
        self.tracer = tracer
        logger.info("ResultsManager initialized with memory config: %s", config)

    def _calculate_n_points(self, tracer: FieldLineTracer) -> int:
        return int((tracer.R_max - tracer.R_min) / tracer.dt)

    @property
    def available_memory(self) -> float:
        """Get available memory in MB."""
        return psutil.virtual_memory().available / (1024 * 1024)

    def estimate_single_line_size(self, template: FieldLine, n_points: int) -> float:
        """Estimate memory size for a single field line."""
        float_size = 4  # FITS uses float32 (4 bytes)
        total_bytes = 8  # footpoint_id (int64)

        # Add size for optional parameters
        for attr in ["r_max", "r_cyl_max", "z_abs_max", "b_ini", "b_r_ini_angle"]:
            if getattr(template, attr) is not None:
                total_bytes += float_size

        # Add size for field line coordinates
        if template.r_line is not None:
            total_bytes += n_points * 3 * float_size

        return total_bytes

    def find_batch_size(
        self,
        template: FieldLine,
    ) -> Tuple[int, float]:
        """Determine optimal batch size based on performance testing.

        Args:
            template: Template for field line calculation
            n_cores: Number of CPU cores to use
            n_points: Number of points per field line
            memory_margin: Fraction of available memory to use (safety factor)

        Returns:
            Tuple of (optimal_batch_size, estimated_line_size)

        Raises:
            ValueError: If requested memory exceeds system limits
        """
        # Check available system memory
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        logger.info("Available memory: %.1f MB", available_memory_mb)
        logger.info("Number of CPU cores to use: %d", self.memory_config.n_cores)

        # Adjust max memory to account for multiprocessing
        max_memory_total = self.memory_config.max_memory_mb * self.memory_config.n_cores
        logger.info("Max memory total: %.1f MB", max_memory_total)

        if max_memory_total > self.memory_config.safety_factor * available_memory_mb:
            raise ValueError(
                f"Requested memory ({max_memory_total:.1f} MB) exceeds "
                f"{self.memory_config.safety_factor*100:.1f}% of available system memory "
                f"({available_memory_mb:.1f} MB)"
            )

        # Estimate memory usage for a single field line
        n_points = self._calculate_n_points(self.tracer)
        estimated_line_size = self.estimate_single_line_size(template, n_points)
        logger.info("Estimated memory per field line: %.2f KB", estimated_line_size / 1024)

        # Calculate max batch size based on memory constraints
        max_batch_size = int(
            (self.memory_config.max_memory_mb * 1024 * 1024) / estimated_line_size
        )
        logger.info("Max batch size based on memory limit: %d", max_batch_size)

        return max_batch_size

    def should_write_batch(self, current_size_mb: float) -> bool:
        """Determine if current batch should be written to disk."""
        return (
            current_size_mb >= self.memory_config.max_memory_mb
            or self.available_memory < self.memory_config.max_memory_mb
        )

    def add_results(
        self, batch_results: FieldLineSet, filename: str, batch_idx: int
    ) -> FieldLineSet:
        """Add batch results and manage memory/output.

        Args:
            batch_results: New batch of field lines to add
            filename: Base filename for output
            batch_idx: Current batch index

        Returns:
            Current results as a FieldLineSet
        """
        self._results_in_memory = self._results_in_memory.concat(batch_results)
        current_size = sum(
            self.estimate_single_line_size(
                line, len(line.r_line) if line.r_line is not None else 0
            )
            for line in batch_results.field_lines
        )
        self._results_in_memory_size += current_size

        logger.debug("Added %d field lines, current in-memory size: %.2f MB", len(batch_results.field_lines), self._results_in_memory_size / (1024 * 1024))

        if self.should_write_batch(self._results_in_memory_size / (1024 * 1024)):
            logger.info("Writing batch %d to disk", batch_idx)
            self.write_results_to_disk(filename, batch_idx)

        return self._results_in_memory

    def write_results_to_disk(self, filename: str, batch_idx: int) -> None:
        """Write results to disk and reset in-memory results."""
        if self._results_in_memory:
            self._results_in_memory.write_to_fits(f"{filename}_batch_{batch_idx}.fits")
            logger.info("Results written to %s_batch_%d.fits", filename, batch_idx)
            self._results_in_memory_size = 0
            self._results_in_memory = FieldLineSet([])




    def rename_batch_files(self, filename: str) -> None:
        """Rename batch output FITS files to show progress (e.g., '1_of_10.fits').
        
        Parameters
        ----------
        filename : str
            The base filename without extension used for the output files
        """
        # Get directory and base name
        directory = os.path.dirname(filename)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        # Find all related FITS files
        pattern = f"{base_name}_batch_*.fits"
        if directory:
            pattern = os.path.join(directory, pattern)
        
        files = sorted(
            glob.glob(pattern),
            key=lambda x: int(x.split('_batch_')[1].split('.fits')[0])
        )
        total_files = len(files)
        
        if total_files == 0:
            logger.warning("No batch files found to rename for %s", filename)
            return
        
        # Rename files
        for i, old_file in enumerate(files, 1):
            directory = os.path.dirname(old_file)
            new_name = f"{base_name}_{i}_of_{total_files}.fits"
            if directory:
                new_name = os.path.join(directory, new_name)
            
            try:
                os.rename(old_file, new_name)
                logger.info("Renamed %s to %s", old_file, new_name)
            except OSError as e:
                logger.error("Error renaming %s to %s: %s", old_file, new_name, e)
