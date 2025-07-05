import logging
import os
import time
from multiprocessing import Pool
from dataclasses import dataclass
from tqdm import tqdm
from typing import Tuple, Optional

import numpy as np
from astropy.coordinates.representation import CartesianRepresentation

from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.tracer import FieldLineTracer
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.line.memory_manager import MemoryManager
from psr_magpy.line.progress_tracker import ProgressTracker
from psr_magpy.line.tracing_parameters import TracingParameters
from psr_magpy.line.calculation_context import CalculationContext
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.line.batch_processor import BatchProcessor

logger = get_logger(__name__)


class FieldLineCalculator:
    """Calculator for field lines based on a magnetic field and grid."""
    
    def __init__(self, field: MultipoleField, grid: FootpointGrid, params: Optional[TracingParameters] = None):
        """Initialize field line calculator.

        Parameters
        ----------
        field : MultipoleField
            Object that calculates the magnetic field
        grid : FootpointGrid
            Grid of footpoints to trace field lines from
        params : Optional[TracingParameters], optional
            Unified tracing parameters, by default None (uses defaults)
        """
        self.context = CalculationContext(field, grid, params)
        logger.info(f"Initialized FieldLineCalculator with {grid.nline} footpoints")
        logger.debug(f"Field: {field}, Grid: {grid}")
        

    def get_field_lines(
        self,
        template: FieldLine,
    ) -> Optional[FieldLineSet]:
        with self.context.progress_tracker.track_calculation("Field line calculation"):
            trial_size, completion_time = self._estimate_memory(template)
            optimal_batch_size = self._configure_batch(trial_size)
            batch_processor = BatchProcessor(
                self.context.grid,
                optimal_batch_size,
                n_cores=self.context.params.performance.n_cores or 1,
                progress_tracker=self.context.progress_tracker
            )
            batch_processor.process(
                template,
                self._process_batch,
                self.context.add_results,
                self.context.params.output.filename or "",
                self.context.params.output.save_results
            )
            results = self.context.memory_manager.results_in_memory
            self._finalize_results(results)
        return results

    def _estimate_memory(self, template):
        trial_size, completion_time = self.context.estimate_memory_requirements(template)
        trial_size_mb = trial_size / (1024 * 1024)
        self.context.progress_tracker.log_memory_estimation(
            self.context.params.performance.n_trial, trial_size_mb, completion_time
        )
        return trial_size, completion_time

    def _configure_batch(self, trial_size):
        optimal_batch_size = self.context.calculate_batch_size(trial_size)
        self.context.progress_tracker.log_batch_size_setting(optimal_batch_size)
        self.context.progress_tracker.log_calculation_start()
        return optimal_batch_size

    def _finalize_results(self, results):
        elapsed = time.time() - self.context.progress_tracker.start_time if self.context.progress_tracker.start_time else 0
        self.context.progress_tracker.log_calculation_completion(
            elapsed, self.context.params.output.save_results, self.context.params.output.filename or "",
            results.n_field_lines if results else 0
        )

    def _process_batch(
        self,
        subset_grid: FootpointGrid,
        template: FieldLine,
    ) -> FieldLineSet:
        logger.debug(f"Processing batch with {subset_grid.nline} field lines using {self.context.params.performance.n_cores} cores")
        if subset_grid.nline is None:
            error_msg = "Subset grid nline is None, cannot process batch"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if subset_grid.footpoint_coordinates is None or subset_grid.indices is None:
            error_msg = "Subset grid coordinates or indices are None, cannot process batch"
            logger.error(error_msg)
            raise ValueError(error_msg)
        coordinates: CartesianRepresentation = subset_grid.footpoint_coordinates
        indices: np.ndarray = subset_grid.indices
        if self.context.params.performance.n_cores is None or self.context.params.performance.n_cores == 1:
            field_lines = []
            for i in range(subset_grid.nline):
                r_initial = coordinates[i]
                assert r_initial is not None  # Type assertion for linter
                footpoint_id = indices[i]
                field_line = self.context.tracer.trace_single_line(r_initial, footpoint_id, template)
                field_lines.append(field_line)
            return FieldLineSet(field_lines)
        else:
            with Pool(processes=self.context.params.performance.n_cores) as pool:
                args = [
                    (coordinates[i], indices[i], template)
                    for i in range(subset_grid.nline)
                ]
                field_lines = pool.starmap(self.context.tracer.trace_single_line, args)
                return FieldLineSet(field_lines)

    def _subset_by_indices(self, indices: np.ndarray) -> FootpointGrid:
        grid = self.context.grid
        if grid.indices is None or grid.nline is None:
            error_msg = "Grid indices or nline is None, cannot subset grid"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if len(indices) == 0:
            error_msg = "Cannot create subset with empty indices"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not all(0 <= i < grid.nline for i in indices):
            error_msg = f"Invalid indices. Must be between 0 and {grid.nline-1}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Subsetting grid: {len(indices)} points")
        return grid.subset_grid(indices)

    def _get_batch_indices(self, start: int, batch_size: int) -> np.ndarray:
        grid = self.context.grid
        if grid.nline is None or grid.indices is None:
            error_msg = "Grid nline or indices is None, cannot get batch indices"
            logger.error(error_msg)
            raise ValueError(error_msg)
        end = min(start + batch_size, grid.nline)
        return grid.indices[start:end]

