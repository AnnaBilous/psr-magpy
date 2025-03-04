import glob
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import logging
import psutil

from src.line.field_line import FieldLine
from src.line.field_line_set import FieldLineSet
from src.line.tracer import FieldLineTracer
from src.line.io import FieldLineSetFitsIO
logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory management.

    Attributes
    ----------
    max_memory_mb : float
        Maximum memory allowed for processing in megabytes.
    safety_factor : float
        Fraction of available memory to leave as a buffer (default is 0.9).
    min_batch_size : int
        Minimum batch size to process (default is 10).
    check_interval_sec : float
        How often to check memory usage in seconds (default is 1.0).
    n_cores : Optional[int]
        Number of CPU cores to use (default is None, which auto-detects).
    """

    max_memory_mb: float
    safety_factor: float = 0.9
    min_batch_size: int = 10
    check_interval_sec: float = 1.0
    n_cores: Optional[int] = None

    def __post_init__(self) -> None:
        """Setup number of cores after initialization."""
        self.n_cores = self._setup_cores(self.n_cores)

    def _setup_cores(self, n_cores: Optional[int] = None) -> int:
        """Setup and validate the number of CPU cores to use.

        Args:
            n_cores: Requested number of cores, or None for auto-detection.

        Returns:
            Number of cores to use.

        Raises:
            ValueError: If the requested number of cores exceeds available cores.
        """
        if n_cores is None:
            n_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()

        n_cores = max(1, n_cores)
        max_cores = psutil.cpu_count()
        if n_cores > max_cores:
            logger.warning("Requested %d cores, but only %d available. Using %d cores.", n_cores, max_cores, max_cores)
            n_cores = max_cores

        logger.info("Using %d CPU cores", n_cores)
        return n_cores


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
        self._results_in_memory: FieldLineSet = None  # Initialize as an empty FieldLineSet
        self._results_in_memory_size: float = 0.0
        self.tracer = tracer
        logger.info("ResultsManager initialized with memory config: %s", config)

    def __repr__(self) -> str:
        """Return a string representation of the ResultsManager instance."""
        return (f"<ResultsManager(memory_config={self.memory_config}, "
                f"total_memory={self._total_memory:.2f}, "
                f"results_in_memory_size={self._results_in_memory_size:.2f}, "
                f"total_field_lines_in_memory={len(self._results_in_memory.field_lines)})>")

    @property
    def available_memory(self) -> float:
        """Get available memory in MB.

        Returns:
            float: Available memory in megabytes.
        """
        return psutil.virtual_memory().available / (1024 * 1024)

    def find_batch_size(
        self,
        estimated_line_size: float,
    ) -> Tuple[int, float]:
        """Determine optimal batch size based on performance testing.

        Args:
            template: Template for field line calculation.

        Returns:
            Tuple[int, float]: Optimal batch size and estimated line size.

        Raises:
            ValueError: If requested memory exceeds system limits.
        """
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        logger.info("Available memory: %.1f MB", available_memory_mb)
        logger.info("Number of CPU cores to use: %d", self.memory_config.n_cores)

        max_memory_total = self.memory_config.max_memory_mb * self.memory_config.n_cores
        logger.info("Max memory total: %.1f MB", max_memory_total)

        if max_memory_total > self.memory_config.safety_factor * available_memory_mb:
            raise ValueError(
                f"Requested memory ({max_memory_total:.1f} MB) exceeds "
                f"{self.memory_config.safety_factor*100:.1f}% of available system memory "
                f"({available_memory_mb:.1f} MB)"
            )

        max_batch_size = int(
            (self.memory_config.max_memory_mb * 1024 * 1024) / estimated_line_size
        )
        logger.info("Max batch size based on memory limit: %d", max_batch_size)

        return max_batch_size

    def should_write_batch(self, current_size_mb: float) -> bool:
        """Determine if current batch should be written to disk.

        Args:
            current_size_mb: Current size of the batch in megabytes.

        Returns:
            bool: True if the batch should be written to disk, False otherwise.
        """
        return (
            current_size_mb >= self.memory_config.max_memory_mb
            or self.available_memory < self.memory_config.max_memory_mb
        )

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
        if self._results_in_memory is None:
            self._results_in_memory = batch_results
        else:
            self._results_in_memory.append(batch_results)
            

        self._results_in_memory_size = self._results_in_memory.estimate_size()

        if self.should_write_batch(self._results_in_memory_size / (1024 * 1024)):
            logger.info("Writing batch %d to disk", batch_idx)
            
            self.write_results_to_disk(filename, batch_idx)

        return self._results_in_memory

    def write_results_to_disk(self, filename: str, batch_idx: int) -> None:
        """Write results to disk and reset in-memory results.

        Args:
            filename: Base filename for output.
            batch_idx: Current batch index.
        """
        if self._results_in_memory:
            logger.info("Writing results to disk for batch %d", batch_idx)
            line_io = FieldLineSetFitsIO(self._results_in_memory)
            line_io.save_to_fits(f"{filename}_batch_{batch_idx}.fits")
            
            logger.info("Results written to %s_batch_%d.fits", filename, batch_idx)
            logger.debug("Clearing in-memory results after writing to disk.")
            self._results_in_memory_size = 0
            self._results_in_memory = None

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
        
        files = sorted(
            glob.glob(pattern),
            key=lambda x: int(x.split('_batch_')[1].split('.fits')[0])
        )
        total_files = len(files)
        
        if total_files == 0:
            logger.warning("No batch files found to rename for %s", filename)
            return
        
        for i, old_file in enumerate(files, 1):
            new_name = f"{base_name}_{i}_of_{total_files}.fits"
            if directory:
                new_name = os.path.join(directory, new_name)
            
            try:
                os.rename(old_file, new_name)
                logger.info("Renamed %s to %s", old_file, new_name)
            except OSError as e:
                logger.error("Error renaming %s to %s: %s", old_file, new_name, e)
