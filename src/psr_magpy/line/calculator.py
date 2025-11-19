import time
from multiprocessing import Pool
from typing import Optional, Union, Tuple, Any
import numpy as np
from astropy.coordinates.representation import CartesianRepresentation

from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet, LightFieldLineSet
from psr_magpy.line.tracing_parameters import TracingParameters
from psr_magpy.line.calculation_context import CalculationContext
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyGridError

logger = get_logger(__name__)


def _trace_single_line_wrapper(args: Tuple[Any, CartesianRepresentation, int, FieldLine]) -> FieldLine:
    """Wrapper function for multiprocessing field line tracing."""
    from psr_magpy.line.tracer import FieldLineTracer
    tracer = args[0]
    assert isinstance(tracer, FieldLineTracer), "First argument must be FieldLineTracer"
    r_initial, footpoint_id, template = args[1], args[2], args[3]
    return tracer.trace_single_line(r_initial, footpoint_id, template)


class FieldLineCalculator:
    """Calculates field lines using the provided configuration."""

    def __init__(self, context: CalculationContext):
        """Initialize the calculator with the given context."""
        self.context = context
        self.global_max_reversed_error = (
            0.0  # Track maximum reversed error across all batches
        )
        self.global_max_dotprod_error = (
            0.0  # Track maximum dotprod error across all batches
        )
        logger.debug("Initialized FieldLineCalculator")

    def get_field_lines(
        self,
        template: FieldLine,
    ) -> Optional[Union[FieldLineSet, LightFieldLineSet]]:
        with self.context.progress_tracker.track_calculation("Field line calculation"):
            # Check if we can skip memory estimation and batch calculation
            if self._can_skip_memory_estimation():
                logger.info(
                    "Skipping memory estimation and batch calculation - processing all field lines at once"
                )
                optimal_batch_size = self.context.grid.nline or 1
            else:
                trial_size, completion_time = self._estimate_memory(template)
                optimal_batch_size = self._configure_batch(trial_size)

            # Process batches using batch processor
            self.context.batch_processor.process_batches(
                grid=self.context.grid,
                template=template,
                batch_size=optimal_batch_size,
                process_batch_fn=self._create_batch_processor_fn(template),
                params=self.context.params,
            )

            results = self.context.memory_manager.results_in_memory
            return self._finalize_results(results)

    def _can_skip_memory_estimation(self) -> bool:
        """Check if we can skip memory estimation and batch calculation."""
        can_skip = self.context.params.performance.skip_memory_estimation

        if can_skip:
            logger.debug("Skipping memory estimation as requested by user")
        else:
            logger.debug("Not skipping memory estimation")

        return can_skip

    def _estimate_memory(self, template: FieldLine) -> Tuple[float, float]:
        """Estimate memory requirements for field line calculation."""
        trial_size, completion_time = self.context.estimate_memory_requirements(
            template
        )
        trial_size_mb = trial_size / (1024 * 1024)
        self.context.progress_tracker.log_memory_estimation(
            self.context.params.performance.n_trial, trial_size_mb, completion_time
        )
        return trial_size, completion_time

    def _configure_batch(self, trial_size: float) -> int:
        """Configure optimal batch size based on memory estimation."""
        # Get total number of field lines for batch size calculation
        total_field_lines = self.context.grid.nline
        optimal_batch_size = self.context.memory_manager.calculate_batch_size(
            trial_size, total_field_lines
        )
        self.context.progress_tracker.log_batch_size_setting(optimal_batch_size)
        self.context.progress_tracker.log_calculation_start()
        return optimal_batch_size

    def _finalize_results(self, results: Optional[FieldLineSet]) -> Optional[Union[FieldLineSet, LightFieldLineSet]]:
        elapsed = (
            time.time() - self.context.progress_tracker.start_time
            if self.context.progress_tracker.start_time
            else 0
        )

        # Extract necessary attributes before saving
        n_field_lines = results.n_field_lines if results is not None else 0

        # Aggregate errors from all batches in memory
        error_msg = None

        # Check global maximum reversed error from all batches
        if (
            self.global_max_reversed_error
            > self.context.params.integration.reversed_error_threshold
        ):
            error_msg = f"Reversed error ({self.global_max_reversed_error:.3f}) exceeds threshold ({self.context.params.integration.reversed_error_threshold:.3f})"
            logger.warning(error_msg)
        # Check global maximum dotprod error from all batches
        elif (
            self.global_max_dotprod_error
            > self.context.params.integration.dotprod_error_threshold
        ):
            error_msg = f"Dotprod error ({self.global_max_dotprod_error:.3f}) exceeds threshold ({self.context.params.integration.dotprod_error_threshold:.3f})"
            logger.warning(error_msg)
        elif results is not None:
            # Check for dotprod errors in the final results (fallback)
            if hasattr(results, "dotprod_error"):
                error_msg = results.dotprod_error

            # Also check for dotprod errors in memory manager if results are aggregated
            memory_results = self.context.memory_manager.results_in_memory
            if (
                memory_results is not None
                and memory_results != results
                and hasattr(memory_results, "dotprod_error")
                and memory_results.dotprod_error
            ):
                error_msg = memory_results.dotprod_error

        # Results are saved in the final batch handling, not here
        # This prevents duplicate saving

        self.context.progress_tracker.log_calculation_completion(
            elapsed,
            self.context.params.output.save_results_to_disk,
            self.context.params.output.filename or "",
            n_field_lines,
            self.context.memory_manager.batch_files_created,
        )

        # Return results based on configuration
        if (
            self.context.params.output.keep_results_in_memory
            and self.context.memory_manager.results_in_memory is not None
        ):
            # Return full results in memory
            logger.info(
                "Returning full results in memory (keep_results_in_memory=True)"
            )
            return self.context.memory_manager.results_in_memory
        else:
            # Return lightweight result object
            return LightFieldLineSet(
                n_field_lines=n_field_lines,
                reversed_error_warning=(
                    error_msg if error_msg and "Reversed error" in error_msg else None
                ),
                dotprod_error=(
                    error_msg if error_msg and "dotprod_error" in error_msg else None
                ),
            )

    def _create_batch_processor_fn(self, template: FieldLine):
        """Create a function for batch processing that tracks errors.
        
        Returns a function that processes a batch and tracks errors.
        """
        def process_batch_with_error_tracking(batch_grid: FootpointGrid) -> FieldLineSet:
            """Process a batch and track errors."""
            batch_results = self._process_batch(batch_grid, template)

            # Track global maximum reversed error from this batch
            if (
                hasattr(batch_results, "reversed_error")
                and batch_results.reversed_error is not None
            ):
                batch_max_error = float(np.max(batch_results.reversed_error))
                self.global_max_reversed_error = max(
                    self.global_max_reversed_error, batch_max_error
                )
                logger.debug(
                    f"Max reversed error = {batch_max_error:.6f}, global max = {self.global_max_reversed_error:.6f}"
                )

            # Track global maximum dotprod error from this batch
            if (
                hasattr(batch_results, "dotprod_error")
                and batch_results.dotprod_error is not None
            ):
                batch_max_error = float(np.max(batch_results.dotprod_error))
                self.global_max_dotprod_error = max(
                    self.global_max_dotprod_error, batch_max_error
                )
                logger.debug(
                    f"Max dotprod error = {batch_max_error:.6f}, global max = {self.global_max_dotprod_error:.6f}"
                )

            return batch_results

        return process_batch_with_error_tracking

    def _process_batch(
        self,
        subset_grid: FootpointGrid,
        template: FieldLine,
    ) -> FieldLineSet:
        logger.debug(
            f"Processing batch with {subset_grid.nline} field lines using {self.context.params.performance.n_cores} cores"
        )
        if subset_grid.nline is None:
            raise PSRMagPyGridError("Subset grid nline is None, cannot process batch")
        if subset_grid.footpoint_coordinates is None or subset_grid.indices is None:
            raise PSRMagPyGridError(
                "Subset grid coordinates or indices are None, cannot process batch"
            )
        coordinates: CartesianRepresentation = subset_grid.footpoint_coordinates
        indices: np.ndarray = subset_grid.indices
        if (
            self.context.params.performance.n_cores is None
            or self.context.params.performance.n_cores == 1
        ):
            field_lines = []
            for i in range(subset_grid.nline):
                r_initial = coordinates[i]
                assert r_initial is not None  # Type assertion for linter
                footpoint_id = indices[i]
                field_line = self.context.tracer.trace_single_line(
                    r_initial, footpoint_id, template
                )
                field_lines.append(field_line)
            return FieldLineSet(field_lines)
        else:
            with Pool(processes=self.context.params.performance.n_cores) as pool:
                args = [
                    (self.context.tracer, coordinates[i], indices[i], template)
                    for i in range(subset_grid.nline)
                ]
                field_lines = pool.map(_trace_single_line_wrapper, args)
                return FieldLineSet(field_lines)
