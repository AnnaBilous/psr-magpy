import time
import gc
from multiprocessing import Pool
from typing import Optional, Union
import numpy as np
from astropy.coordinates.representation import CartesianRepresentation
from pathlib import Path

from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet, LightFieldLineSet
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.line.tracing_parameters import TracingParameters
from psr_magpy.line.calculation_context import CalculationContext
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.io import get_io_factory

logger = get_logger(__name__)


def _trace_single_line_wrapper(args):
    """Wrapper function for multiprocessing field line tracing."""
    tracer, r_initial, footpoint_id, template = args
    return tracer.trace_single_line(r_initial, footpoint_id, template)


class FieldLineCalculator:
    """Calculator for field lines based on a magnetic field and grid."""

    def __init__(
        self,
        field: MultipoleField,
        grid: FootpointGrid,
        params: Optional[TracingParameters] = None,
    ):
        """Initialize field line calculator."""
        self.context = CalculationContext(field, grid, params)
        logger.info(f"Initialized FieldLineCalculator with {grid.nline} footpoints")
        logger.debug(f"Field: {field}, Grid: {grid}")

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
                self._process_in_batches(template, optimal_batch_size)
            else:
                trial_size, completion_time = self._estimate_memory(template)
                optimal_batch_size = self._configure_batch(trial_size)
                self._process_in_batches(template, optimal_batch_size)

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

    def _estimate_memory(self, template):
        trial_size, completion_time = self.context.estimate_memory_requirements(
            template
        )
        trial_size_mb = trial_size / (1024 * 1024)
        self.context.progress_tracker.log_memory_estimation(
            self.context.params.performance.n_trial, trial_size_mb, completion_time
        )
        return trial_size, completion_time

    def _configure_batch(self, trial_size):
        # Get total number of field lines for batch size calculation
        total_field_lines = self.context.grid.nline
        optimal_batch_size = self.context.memory_manager.calculate_batch_size(
            trial_size, total_field_lines
        )
        self.context.progress_tracker.log_batch_size_setting(optimal_batch_size)
        self.context.progress_tracker.log_calculation_start()
        return optimal_batch_size

    def _finalize_results(self, results):
        elapsed = (
            time.time() - self.context.progress_tracker.start_time
            if self.context.progress_tracker.start_time
            else 0
        )

        # Extract necessary attributes before saving
        n_field_lines = results.n_field_lines if results is not None else 0
        error_msg = None
        if results is not None:
            if hasattr(results, "reversed_error"):
                reversed_errors = results.reversed_error
                max_error = float(np.max(reversed_errors))
                if max_error > self.context.params.integration.reversed_error_threshold:
                    error_msg = f"Reversed error ({max_error:.3f}) exceeds threshold ({self.context.params.integration.reversed_error_threshold:.3f})"
                    logger.warning(error_msg)  # Log as warning since it doesn't affect success
            elif hasattr(results, "dotprod_error"):
                error_msg = results.dotprod_error

        # Save results to disk if requested
        if self.context.params.output.save_results and results is not None:
            filename = self.context.params.output.filename
            if filename:
                # Save results with batch naming even if no intermediate batches were created
                filename_path = Path(filename)
                batch_filename = (
                    filename_path.parent / f"{filename_path.stem}_batch_001_out_of_001.fits"
                )

                # Save results
                get_io_factory().save_data(results, batch_filename, overwrite=True)
                self.context.progress_tracker.log_io_completion(
                    "save", str(batch_filename), elapsed, n_field_lines
                )

                # Mark that batch files were created
                self.context.memory_manager._batch_files_created = True

                # Rename and organize batch files if any were created
                if self.context.memory_manager.batch_files_created:
                    self.context.memory_manager.rename_batch_files(
                        filename, self.context.progress_tracker
                    )

                # Clear memory after saving
                self.context.memory_manager._results_in_memory = None
                self.context.memory_manager._results_in_memory_size = 0.0
                gc.collect()
                logger.info("Cleared memory after saving results")

        self.context.progress_tracker.log_calculation_completion(
            elapsed,
            self.context.params.output.save_results,
            self.context.params.output.filename or "",
            n_field_lines,
        )

        # Return lightweight result object
        return LightFieldLineSet(
            n_field_lines=n_field_lines,
            reversed_error_warning=error_msg if error_msg and "Reversed error" in error_msg else None,
            dotprod_error=error_msg if error_msg and "dotprod_error" in error_msg else None
        )

    def _process_in_batches(self, template: FieldLine, batch_size: int) -> None:
        """Process field lines in batches."""
        nline = self.context.grid.nline
        if nline is None:
            error_msg = "Grid nline is None, cannot process batches"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.context.grid.indices is None:
            error_msg = "Grid indices is None, cannot process batches"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Processing {nline} field lines in batches of {batch_size}")
        batch_idx = 0
        total_processed = 0

        # Create progress bar for total lines
        progress_bar = self.context.progress_tracker.create_progress_bar(
            total=nline, description="Field lines"
        )

        while total_processed < nline:
            start_idx = total_processed
            batch_indices = np.arange(start_idx, min(start_idx + batch_size, nline))

            self.context.progress_tracker.log_batch_start(
                batch_idx, start_idx, start_idx + len(batch_indices), nline
            )

            batch_grid = self.context.grid.subset_grid(batch_indices)
            batch_results = self._process_batch(batch_grid, template)
            self.context.add_results(
                batch_results, self.context.params.output.filename or "", batch_idx
            )

            total_processed += len(batch_indices)
            batch_idx += 1

            # Update progress bar
            progress_bar.n = total_processed
            progress_bar.refresh()

            self.context.progress_tracker.log_batch_completion(
                batch_idx,
                total_processed,
                nline,
                self.context.memory_manager.results_in_memory,
            )

        # Close progress bar
        progress_bar.close()

    def _process_batch(
        self,
        subset_grid: FootpointGrid,
        template: FieldLine,
    ) -> FieldLineSet:
        logger.debug(
            f"Processing batch with {subset_grid.nline} field lines using {self.context.params.performance.n_cores} cores"
        )
        if subset_grid.nline is None:
            error_msg = "Subset grid nline is None, cannot process batch"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if subset_grid.footpoint_coordinates is None or subset_grid.indices is None:
            error_msg = (
                "Subset grid coordinates or indices are None, cannot process batch"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
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
