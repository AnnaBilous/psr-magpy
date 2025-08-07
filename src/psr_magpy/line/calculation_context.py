"""
Calculation context for field line tracing.

Provides a clean interface to all calculation components with explicit
initialization and validation.
"""

from typing import Optional, Tuple

from psr_magpy.field.multipole import MultipoleField
from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.tracing_parameters import TracingParameters
from psr_magpy.line.tracer import FieldLineTracer
from psr_magpy.line.memory_manager import MemoryManager
from psr_magpy.line.progress_tracker import ProgressTracker
from psr_magpy.line.field_line import FieldLine

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__)


class CalculationContext:
    """Runtime context for field line calculations."""

    def __init__(
        self,
        field: MultipoleField,
        grid: FootpointGrid,
        params: Optional[TracingParameters] = None,
    ):
        """Initialize calculation context."""
        self.field = field
        self.grid = grid
        self.params = params or TracingParameters()

        # Validate n_trial against grid size only if memory estimation is enabled
        if (
            not self.params.performance.skip_memory_estimation
            and self.grid.nline is not None
        ):
            self.params.performance.validate_against_grid(self.grid.nline)

        # Initialize tracer
        self._tracer = FieldLineTracer(self.field, self.params)

        # Initialize memory manager (consolidated)
        self._memory_manager = MemoryManager(
            self.params.performance,
            self._tracer,
            max_file_size_mb=self.params.performance.max_memory_mb,  # Use the configured max_memory_mb
            save_results=self.params.output.save_results,
        )
        logger.debug(
            f"Created MemoryManager with max_memory_mb={self.params.performance.max_memory_mb}, save_results={self.params.output.save_results}"
        )

        # Initialize progress tracker with parameters from config
        self._progress_tracker = ProgressTracker(
            verbose=self.params.performance.verbose,
            show_progress=self.params.performance.progress_bar,
        )
        logger.debug(
            f"Created ProgressTracker with verbose={self.params.performance.verbose}, show_progress={self.params.performance.progress_bar}"
        )

    @property
    def tracer(self) -> FieldLineTracer:
        """Get the field line tracer."""
        return self._tracer

    @property
    def memory_manager(self) -> MemoryManager:
        """Get the memory manager."""
        return self._memory_manager

    @property
    def progress_tracker(self) -> ProgressTracker:
        """Get the progress tracker."""
        return self._progress_tracker

    def estimate_memory_requirements(
        self, template: FieldLine, n_trial: int = 10
    ) -> Tuple[float, float]:
        """Estimate memory requirements using simplified heuristics."""
        return self.memory_manager.estimate_memory_requirements(
            template, self.grid, n_trial
        )

    def add_results(self, batch_results, filename: str, batch_idx: int):
        """Add batch results to memory manager."""
        return self.memory_manager.add_results(batch_results, filename, batch_idx)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CalculationContext(field={type(self.field).__name__}, "
            f"grid={type(self.grid).__name__}, "
            f"n_footpoints={self.grid.nline})"
        )
