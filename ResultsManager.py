from dataclasses import dataclass
from typing import Optional, List, Tuple
import psutil
import numpy as np
from FieldLineDefinition import FieldLine, FieldLineSet
from FieldLineTracer import FieldLineTracer


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
            print(
                f"Requested {n_cores} cores, but only {max_cores} available. Using {max_cores} cores."
            )
            n_cores = max_cores

        print(f"Using {n_cores} CPU cores")
        return n_cores


class ResultsManager:
    def __init__(self, config: MemoryConfig, tracer: FieldLineTracer):
        """Initialize results manager.

        Args:
            config: Memory management configuration
        """
        self.config = config
        self._total_memory = psutil.virtual_memory().total
        self._results: List[FieldLine] = []
        self._result_size = 0
        self.tracer = tracer

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
        print(f"Available memory: {available_memory_mb:.1f} MB")
        print(f"Number of CPU cores to use: {self.config.n_cores}")

        # Adjust max memory to account for multiprocessing
        max_memory_total = self.config.max_memory_mb * self.config.n_cores
        print(f"Max memory total: {max_memory_total:.1f} MB")

        if max_memory_total > self.config.safety_factor * available_memory_mb:
            raise ValueError(
                f"Requested memory ({max_memory_total:.1f} MB) exceeds "
                f"{self.config.safety_factor*100:.1f}% of available system memory "
                f"({available_memory_mb:.1f} MB)"
            )

        # Estimate memory usage for a single field line
        n_points = self._calculate_n_points(self.tracer)
        estimated_line_size = self.estimate_single_line_size(template, n_points)
        print(f"Estimated memory per field line: {estimated_line_size / 1024:.2f} KB")

        # Calculate max batch size based on memory constraints
        max_batch_size = int(
            (self.config.max_memory_mb * 1024 * 1024) / estimated_line_size
        )
        print(f"Max batch size based on memory limit: {max_batch_size}")

        return max_batch_size

    def should_write_batch(self, current_size_mb: float) -> bool:
        """Determine if current batch should be written to disk."""
        return (
            current_size_mb >= self.config.max_memory_mb
            or self.available_memory < self.config.max_memory_mb
        )

    def add_results(
        self, batch_results: List[FieldLine], filename: str, batch_idx: int
    ) -> List[FieldLine]:
        """Add batch results and manage memory/output.

        Args:
            batch_results: New batch of field lines to add
            filename: Base filename for output
            batch_idx: Current batch index

        Returns:
            Current results list
        """
        self._results.extend(batch_results)
        current_size = sum(
            self.estimate_single_line_size(
                line, len(line.r_line) if line.r_line is not None else 0
            )
            for line in batch_results
        )
        self._result_size += current_size

        current_result_size_mb = self._result_size / (1024 * 1024)
        print(f"Current size: {current_result_size_mb:.2f} MB")
        if self.should_write_batch(current_result_size_mb):
            print(f"Writing batch {batch_idx} to disk")
            self._write_intermediate_results(filename, batch_idx)
            self._results = []
            self._result_size = 0

        return self._results

    def _write_intermediate_results(self, filename: str, batch_idx: int) -> None:
        """Write intermediate results to disk."""
        if self._results:
            intermediate_set = FieldLineSet(self._results)
            intermediate_set.write_to_fits(f"{filename}_batch_{batch_idx}")

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        vm = psutil.virtual_memory()
        return {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent_used": vm.percent,
            "results_size_mb": self._result_size / (1024 * 1024),
        }
