"""
Calculation context for field line tracing.

Provides a clean interface to all calculation components with explicit
initialization and validation. Replaces CalculatorConfig with better
separation of concerns and immediate initialization.
"""

import logging
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
    """Runtime context for field line calculations.
    
    Provides a clean interface to all calculation components
    with explicit initialization and validation.
    """

    def __init__(
        self,
        field: MultipoleField,
        grid: FootpointGrid,
        params: Optional[TracingParameters] = None,
        max_file_size_mb: float = 100.0
    ):
        """Initialize calculation context.
        
        Parameters
        ----------
        field : MultipoleField
            Magnetic field for calculations
        grid : FootpointGrid
            Grid of footpoints
        params : Optional[TracingParameters], optional
            Calculation parameters, by default None (uses defaults)
        max_file_size_mb : float, optional
            Maximum allowed file size in MB for each results file (default: 100.0)
        """
        self.field = field
        self.grid = grid
        self.params = params or TracingParameters()
        
        # Validate n_trial against grid size (cross-validation)
        if self.grid.nline is not None:
            self.params.performance.validate_against_grid(self.grid.nline)
        
        # Initialize tracer
        self._tracer = FieldLineTracer(self.field, self.params)
        logger.debug(f"Created FieldLineTracer with R_max={self.params.R_max}")
        
        # Initialize memory manager (consolidated)
        self._memory_manager = MemoryManager(self.params.performance, self._tracer, max_file_size_mb=max_file_size_mb)
        logger.debug(f"Created MemoryManager with max_file_size_mb={max_file_size_mb}")
        
        # Initialize progress tracker
        self._progress_tracker = ProgressTracker()
        logger.debug("Created ProgressTracker")

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

    def trace_single_line(self, r_initial, footpoint_id: int, template: FieldLine):
        """Trace a single field line."""
        return self.tracer.trace_single_line(r_initial, footpoint_id, template)

    def estimate_memory_requirements(self, template: FieldLine, n_trial: int = 10) -> Tuple[float, float]:
        """Estimate memory requirements using simplified heuristics."""
        return self.memory_manager.estimate_memory_requirements(template, self.grid, n_trial)

    def calculate_batch_size(self, estimated_line_size: float) -> int:
        """Calculate optimal batch size."""
        return self.memory_manager.calculate_batch_size(estimated_line_size)

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