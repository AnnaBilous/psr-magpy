import glob
import logging
import os
import psutil
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.io import FieldLineSetFitsIO
from psr_magpy.line.tracer import FieldLineTracer
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


@dataclass
class MemoryConfig:
    """Configuration for memory management.

    Attributes
    ----------
    max_memory_mb : float
        Maximum memory allowed for processing in megabytes.
    safety_factor : float
        Fraction of available memory to leave as a buffer (default is 0.9).
    max_batch_size : int
        Maximum batch size to process (default is 100).
    check_interval_sec : float
        How often to check memory usage in seconds (default is 1.0).
    n_cores : Optional[int]
        Number of CPU cores to use (default is None, which auto-detects).
    """

    max_memory_mb: float
    safety_factor: float = 0.9
    max_batch_size: int = 100
    check_interval_sec: float = 1.0
    n_cores: Optional[int] = None

    def __post_init__(self) -> None:
        """Setup number of cores after initialization."""
        self.n_cores = self._setup_cores(self.n_cores)
        logger.debug(f"MemoryConfig initialized: {self}")

    def _setup_cores(self, n_cores: Optional[int] = None) -> int:
        """Setup and validate the number of CPU cores to use.

        Args:
            n_cores: Requested number of cores, or None for auto-detection.

        Returns:
            Number of cores to use.

        Raises:
            ValueError: If the requested number of cores exceeds available cores.
        """
        available_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
        logger.debug(f"Available physical cores: {available_cores}")
        
        if n_cores is None:
            n_cores = available_cores
            logger.debug(f"No core count specified, using all available cores: {n_cores}")

        n_cores = max(1, n_cores)
        max_cores = psutil.cpu_count()
        if n_cores > max_cores:
            warning_msg = f"Requested {n_cores} cores, but only {max_cores} available. Using {max_cores} cores."
            logger.warning(warning_msg)
            n_cores = max_cores

        logger.info(f"Using {n_cores} CPU cores for processing")
        return n_cores
        
    def __repr__(self) -> str:
        """Return string representation of memory configuration."""
        return (f"MemoryConfig(max_memory_mb={self.max_memory_mb}, "
                f"safety_factor={self.safety_factor}, "
                f"max_batch_size={self.max_batch_size}, "
                f"n_cores={self.n_cores})")


class ResultsManager:
    """Manages the results of field line calculations, including memory management and file I/O.

    Attributes
    ----------
    memory_config : MemoryConfig
        Configuration for memory management.
    _total_memory : float
        Total system memory available.
    _results_in_memory : FieldLineSet
        Collection of field lines currently in memory.
    _results_in_memory_size : float
        Total size of results in memory in bytes.
    tracer : FieldLineTracer
        Tracer object used for field line calculations.
    """

    def __init__(self, config: MemoryConfig, tracer: FieldLineTracer) -> None:
        """Initialize results manager.

        Args:
            config: Memory management configuration.
            tracer: FieldLineTracer instance for tracing field lines.
        """
        self.memory_config = config
        self._total_memory = psutil.virtual_memory().total
        self._results_in_memory: FieldLineSet = None
        self._results_in_memory_size: float = 0.0
        self.tracer = tracer
        
        logger.info(f"ResultsManager initialized with {config.max_memory_mb} MB memory limit")
        logger.debug(f"Total system memory: {self._total_memory / (1024*1024*1024):.2f} GB")
        logger.debug(f"Memory safety factor: {config.safety_factor}")

    def __repr__(self) -> str:
        """Return a string representation of the ResultsManager instance."""
        memory_mb = self._results_in_memory_size / (1024 * 1024)
        result_count = 0
        if self._results_in_memory is not None:
            result_count = self._results_in_memory.n_field_lines
            
        return (
            f"ResultsManager(memory_config={self.memory_config}, "
            f"results_size={memory_mb:.2f} MB, "
            f"field_lines_in_memory={result_count})"
        )

    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics.
        
        Returns:
            dict: Dictionary with memory statistics
        """
        vm = psutil.virtual_memory()
        stats = {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent_used": vm.percent,
            "results_mb": self._results_in_memory_size / (1024**2) if self._results_in_memory_size else 0
        }
        
        logger.debug(f"Memory stats: {stats['percent_used']:.1f}% used, "
                    f"{stats['available_gb']:.2f} GB available")
        return stats

    @property
    def available_memory(self) -> float:
        """Get available memory in MB.

        Returns:
            float: Available memory in megabytes.
        """
        available = psutil.virtual_memory().available / (1024 * 1024)
        logger.debug(f"Available memory: {available:.2f} MB")
        return available

    def find_batch_size(self, estimated_line_size: float) -> int:
        """Determine optimal batch size based on memory constraints.

        Args:
            estimated_line_size: Estimated size of a single field line in bytes.

        Returns:
            int: Optimal batch size.

        Raises:
            ValueError: If requested memory exceeds system limits.
        """
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        logger.info(f"Available memory: {available_memory_mb:.1f} MB")
        
        # Calculate total memory needed across all cores
        max_memory_total = self.memory_config.max_memory_mb * self.memory_config.n_cores
        logger.info(f"Maximum memory allocation: {max_memory_total:.1f} MB ({self.memory_config.n_cores} cores)")

        # Check if we're exceeding safe memory limits
        safe_memory_limit = self.memory_config.safety_factor * available_memory_mb
        if max_memory_total > safe_memory_limit:
            error_msg = (
                f"Requested memory ({max_memory_total:.1f} MB) exceeds "
                f"{self.memory_config.safety_factor*100:.1f}% of available system memory "
                f"({available_memory_mb:.1f} MB)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate batch size based on memory constraints
        estimated_line_size_mb = estimated_line_size / (1024 * 1024)
        logger.debug(f"Estimated size per field line: {estimated_line_size_mb:.4f} MB")
        
        max_batch_size = int(self.memory_config.max_memory_mb / estimated_line_size_mb)
        max_batch_size = min(max_batch_size, self.memory_config.max_batch_size)
        
        logger.info(f"Calculated batch size: {max_batch_size} field lines")
        logger.debug(f"Estimated memory per batch: {max_batch_size * estimated_line_size_mb:.2f} MB")
        
        return max_batch_size

    def should_write_batch(self, current_size_mb: float) -> bool:
        """Determine if current batch should be written to disk.

        Args:
            current_size_mb: Current size of the batch in megabytes.

        Returns:
            bool: True if the batch should be written to disk, False otherwise.
        """
        memory_limit_exceeded = current_size_mb >= self.memory_config.max_memory_mb
        available_memory_low = self.available_memory < self.memory_config.max_memory_mb
        
        if memory_limit_exceeded:
            logger.debug(f"Memory limit exceeded: {current_size_mb:.2f} MB > {self.memory_config.max_memory_mb:.2f} MB")
        
        if available_memory_low:
            logger.debug(f"Available memory low: {self.available_memory:.2f} MB < {self.memory_config.max_memory_mb:.2f} MB")
            
        return memory_limit_exceeded or available_memory_low

    def add_results(
        self, batch_results: FieldLineSet, filename: str, batch_idx: int
    ) -> FieldLineSet:
        """Add batch results and manage memory/output.

        Args:
            batch_results: New batch of field lines to add.
            filename: Base filename for output.
            batch_idx: Current batch index.

        Returns:
            FieldLineSet: Current results as a FieldLineSet.
        """
        logger.debug(f"Adding batch results with {batch_results.n_field_lines} field lines")
        
        if self._results_in_memory is None:
            self._results_in_memory = batch_results
            logger.debug("Initialized results in memory with first batch")
        else:
            logger.debug(f"Appending {batch_results.n_field_lines} field lines to existing {self._results_in_memory.n_field_lines}")
            self._results_in_memory.append(batch_results)

        # Update size tracking
        self._results_in_memory_size = self._results_in_memory.estimate_size()
        memory_mb = self._results_in_memory_size / (1024 * 1024)
        logger.info(f"Results in memory: {self._results_in_memory.n_field_lines} field lines, {memory_mb:.2f} MB")

        # Check if we need to write to disk
        if self.should_write_batch(memory_mb):
            logger.info(f"Memory threshold reached, writing batch {batch_idx} to disk")
            
            # Get range of footpoint IDs for filename
            footpoint_ids = self._results_in_memory.get_property("footpoint_id")
            suffix = f"{int(np.min(footpoint_ids))}_{int(np.max(footpoint_ids))}"
            
            self.write_results_to_disk(filename, suffix)

        return self._results_in_memory

    def write_results_to_disk(self, filename: str, suffix: str) -> None:
        """Write results to disk and reset in-memory results.

        Args:
            filename: Base filename for output.
            suffix: Suffix to append to filename (typically footpoint ID range).
        """
        if self._results_in_memory is None or self._results_in_memory.n_field_lines == 0:
            logger.warning("No results in memory to write to disk")
            return
        
        output_filename = f"{filename}_batch_{suffix}.fits"
        
        logger.debug(f"Writing {self._results_in_memory.n_field_lines} field lines to {output_filename}")
        
        try:
            line_io = FieldLineSetFitsIO(self._results_in_memory)
            line_io.save_to_fits(output_filename)
            
            logger.info(f"Successfully wrote results to {output_filename}")
            logger.info("Clearing in-memory results after writing to disk")
            
            # Reset in-memory storage
            self._results_in_memory_size = 0
            self._results_in_memory = None
            
        except Exception as e:
            logger.error(f"Error writing results to {output_filename}: {e}")
            logger.debug("Write error details", exc_info=True)
            raise

    def rename_batch_files(self, filename: str) -> None:
        """Rename batch output FITS files to show progress (e.g., '1_of_10.fits').

        Args:
            filename: The base filename without extension used for the output files.
        """
        
        directory = os.path.dirname(filename)
        base_name = os.path.splitext(os.path.basename(filename))[0]

        pattern = f"{base_name}_batch_*.fits"
        if directory:
            pattern = os.path.join(directory, pattern)
            
        logger.info(f"Searching for batch files with pattern: {pattern}")
       
        # Find all batch files
        files = sorted(
            glob.glob(pattern),
            key=lambda x: int(x.split("_batch_")[1].split(".")[0])
        )
        
        total_files = len(files)

        if total_files == 0:
            logger.warning(f"No batch files found to rename for {filename}")
            return

        logger.info(f"Renaming {total_files} batch files to sequential format")
        
        # Rename files to show sequence
        for i, old_file in enumerate(files, 1):
            new_name = f"{base_name}_{i}_of_{total_files}.fits"
            if directory:
                new_name = os.path.join(directory, new_name)

            try:
                os.rename(old_file, new_name)
                logger.debug(f"Renamed {os.path.basename(old_file)} to {os.path.basename(new_name)}")
            except OSError as e:
                logger.error(f"Error renaming {old_file} to {new_name}: {e}")
                logger.debug("Rename error details", exc_info=True)
                
        logger.info(f"Successfully renamed {total_files} batch files")
