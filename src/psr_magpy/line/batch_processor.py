"""Batch processing strategies for field line calculations."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.memory_manager import MemoryManager
from psr_magpy.line.progress_tracker import ProgressReporter
from psr_magpy.line.tracing_parameters import TracingParameters
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyGridError

logger = get_logger(__name__)


class BatchProcessor(ABC):
    """Base class for batch processing strategies."""

    @abstractmethod
    def process_batches(
        self,
        grid: FootpointGrid,
        template: FieldLine,
        batch_size: int,
        process_batch_fn,
        params: TracingParameters,
    ) -> None:
        """Process field lines in batches.
        
        Parameters
        ----------
        grid : FootpointGrid
            Grid containing footpoints to process
        template : FieldLine
            Template field line for tracing
        batch_size : int
            Size of each batch
        process_batch_fn : callable
            Function to process a single batch: (subset_grid) -> FieldLineSet
        params : TracingParameters
            Tracing parameters including output configuration
        """
        pass


class MemoryAwareBatchProcessor(BatchProcessor):
    """Batch processor using memory-aware strategy (current implementation)."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        progress_reporter: ProgressReporter,
    ):
        """Initialize memory-aware batch processor.
        
        Parameters
        ----------
        memory_manager : MemoryManager
            Memory manager for handling batch results
        progress_reporter : ProgressReporter
            Progress reporter for logging and progress bars
        """
        self.memory_manager = memory_manager
        self.progress_reporter = progress_reporter
        logger.debug("Initialized MemoryAwareBatchProcessor")

    def process_batches(
        self,
        grid: FootpointGrid,
        template: FieldLine,
        batch_size: int,
        process_batch_fn,
        params: TracingParameters,
    ) -> None:
        """Process field lines in batches using memory-aware strategy."""
        nline = grid.nline
        if nline is None:
            error_msg = "Grid nline is None, cannot process batches"
            logger.error(error_msg)
            raise PSRMagPyGridError(error_msg)
        if grid.indices is None:
            error_msg = "Grid indices is None, cannot process batches"
            logger.error(error_msg)
            raise PSRMagPyGridError(error_msg)

        logger.info(f"Processing {nline} field lines in batches of {batch_size}")
        batch_idx = 0
        total_processed = 0

        # Create progress bar for total lines
        progress_bar = self.progress_reporter.create_progress_bar(
            total=nline, description="Field lines"
        )

        try:
            while total_processed < nline:
                start_idx = total_processed
                batch_indices = np.arange(start_idx, min(start_idx + batch_size, nline))

                self.progress_reporter.report_batch_start(
                    batch_idx, start_idx, start_idx + len(batch_indices), nline
                )

                batch_grid = grid.subset_grid(batch_indices)
                batch_results = process_batch_fn(batch_grid)

                # Add results to memory manager
                filename = params.output.filename or ""
                self.memory_manager.add_results(batch_results, filename, batch_idx)

                total_processed += len(batch_indices)
                batch_idx += 1

                # Update progress bar
                progress_bar.n = total_processed
                progress_bar.refresh()

                self.progress_reporter.report_batch_completion(
                    batch_idx,
                    total_processed,
                    nline,
                    self.memory_manager.results_in_memory,
                )

        finally:
            # Close progress bar
            progress_bar.close()

        # Handle final batch and rename batch files after all batches are processed
        if params.output.save_results_to_disk:
            filename = params.output.filename
            if filename:
                self.memory_manager.finalize_batches(filename, batch_idx, self.progress_reporter)

