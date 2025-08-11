"""Progress tracking module for field line calculations.

Provides classes for tracking progress, timing, and logging during
field line calculations with support for progress bars and detailed logging.
"""

import time
import psutil
from contextlib import contextmanager
from typing import Optional, Any, Union
from pathlib import Path

from tqdm import tqdm

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class ProgressTracker:
    """Tracks progress, timing, and logging for field line calculations and I/O operations."""

    def __init__(self, verbose: bool = False, show_progress: bool = True):
        """Initialize progress tracker."""
        self.verbose = verbose
        self.show_progress = show_progress
        self.start_time: Optional[float] = None
        self._progress_bar: Optional[tqdm] = None
        self._io_progress_bar: Optional[tqdm] = None

        logger.debug("Initialized ProgressTracker with %s", self.verbose)

    @contextmanager
    def track_calculation(self, description: str):
        """Context manager for tracking calculation progress."""
        self.start_calculation(description)
        try:
            yield self
        finally:
            self.end_calculation()

    @contextmanager
    def track_io_operation(
        self,
        operation: str,
        filepath: Union[str, Path],
        data_size: Optional[int] = None,
    ):
        """Context manager for tracking I/O operations."""
        self.start_io_operation(operation, filepath, data_size)
        try:
            yield self
        finally:
            self.end_io_operation()

    def start_calculation(self, description: str) -> None:
        """Start tracking a new calculation."""
        self.start_time = time.time()
        logger.info(f"Starting: {description}")
        if self.verbose:
            logger.debug(f"Calculation started at {time.strftime('%H:%M:%S')}")

    def end_calculation(self) -> None:
        """End the current calculation and log completion time."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            logger.info(f"Calculation completed in {elapsed:.1f} seconds")
            if self.verbose:
                logger.debug(f"Calculation ended at {time.strftime('%H:%M:%S')}")
            self.start_time = None

    def start_io_operation(
        self,
        operation: str,
        filepath: Union[str, Path],
        data_size: Optional[int] = None,
    ) -> None:
        """Start tracking an I/O operation."""
        self.start_time = time.time()
        filepath = Path(filepath)
        operation_desc = f"{operation.capitalize()}ing {filepath.name}"

        logger.info(f"Starting I/O: {operation_desc}")
        if self.verbose:
            logger.debug(f"I/O operation started at {time.strftime('%H:%M:%S')}")
            if data_size:
                logger.debug(f"Data size: {data_size} items")

        # Create progress bar for I/O operations
        if self.show_progress and data_size:
            self._io_progress_bar = tqdm(
                total=data_size,
                desc=operation_desc,
                unit="items",
                disable=not self.show_progress,
            )

    def end_io_operation(self) -> None:
        """End the current I/O operation and log completion time."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            logger.info(f"I/O operation completed in {elapsed:.1f} seconds")
            if self.verbose:
                logger.debug(f"I/O operation ended at {time.strftime('%H:%M:%S')}")
            self.start_time = None

        # Close I/O progress bar
        if self._io_progress_bar is not None:
            self._io_progress_bar.close()
            self._io_progress_bar = None

    def update_io_progress(
        self, current: int, total: Optional[int] = None, message: str = ""
    ) -> None:
        """Update I/O operation progress."""
        if self._io_progress_bar is not None:
            self._io_progress_bar.n = current
            if total:
                self._io_progress_bar.total = total
            self._io_progress_bar.refresh()

        if message and self.verbose:
            logger.debug(f"I/O Progress: {current}/{total or 'unknown'} - {message}")

    def log_io_start(
        self,
        operation: str,
        filepath: Union[str, Path],
        data_size: Optional[int] = None,
    ) -> None:
        """Log the start of an I/O operation."""
        filepath = Path(filepath)
        logger.info(f"Starting {operation}: {filepath.name}")
        if self.verbose and data_size:
            logger.debug(f"Data size: {data_size} items")

    def log_io_completion(
        self,
        operation: str,
        filepath: Union[str, Path],
        elapsed: float,
        data_size: Optional[int] = None,
    ) -> None:
        """Log the completion of an I/O operation."""
        filepath = Path(filepath)
        logger.info(f"Completed {operation}: {filepath.name} in {elapsed:.1f} seconds")
        if self.verbose and data_size:
            logger.debug(f"Processed {data_size} items")

    def log_file_size(self, filepath: Union[str, Path]) -> None:
        """Log the size of a file."""
        filepath = Path(filepath)
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {size_mb:.2f} MB")
            if self.verbose:
                logger.debug(f"File: {filepath}")

    def log_completion_time(self) -> None:
        """Log the completion time if calculation is in progress."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            logger.info(f"Elapsed time: {elapsed:.1f} seconds")

    def create_progress_bar(self, total: int, description: str = "Processing") -> Any:
        """Create a progress bar for tracking progress."""
        if not self.show_progress:
            # Return a dummy progress bar that does nothing
            class _DummyProgressBar:
                def __init__(self):
                    self.n = 0
                    self.total = 0

                def refresh(self):
                    pass

                def close(self):
                    pass

            return _DummyProgressBar()
        self._progress_bar = tqdm(
            total=total, desc=description, unit="items", disable=not self.show_progress
        )
        logger.debug(f"Created progress bar: {description} ({total} items)")
        return self._progress_bar

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress and log message."""
        if self._progress_bar is not None:
            self._progress_bar.n = current
            self._progress_bar.refresh()

        if message and self.verbose:
            logger.debug(f"Progress: {current}/{total} - {message}")

    def log_batch_start(
        self, batch_idx: int, start_idx: int, end_idx: int, total: int
    ) -> None:
        """Log the start of a batch processing."""
        batch_size = end_idx - start_idx
        logger.info(
            f"Processing batch {batch_idx + 1}: indices {start_idx}-{end_idx-1} ({batch_size} items)"
        )
        if self.verbose:
            logger.debug(
                f"Batch {batch_idx + 1}: {batch_size}/{total} items ({batch_size/total*100:.1f}%)"
            )

    def log_batch_completion(
        self, batch_idx: int, processed: int, total: int, field_line_set=None
    ) -> None:
        """Log the completion of a batch with memory usage information."""
        logger.info(f"Completed batch {batch_idx + 1}: {processed}/{total} field lines")

        # Get process memory usage information
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

        logger.info(f"Memory usage after batch {batch_idx + 1}: {memory_mb:.1f} MB")

        # Get field line data size if available
        if field_line_set is not None:
            try:
                data_size_bytes = field_line_set.estimate_size()
                data_size_mb = data_size_bytes / (1024 * 1024)
                logger.info(
                    f"Field line data size in memory: {data_size_mb:.1f} MB ({data_size_bytes:,} bytes)"
                )
            except Exception as e:
                logger.warning(f"Could not estimate field line data size: {e}")

        if self.verbose:
            logger.debug(f"Progress: {processed/total*100:.1f}% complete")
            logger.debug(
                f"Memory details: RSS={memory_mb:.1f} MB, VMS={memory_info.vms / (1024 * 1024):.1f} MB"
            )

    def log_memory_estimation(
        self, n_trial: int, trial_size_mb: float, completion_time: float
    ) -> None:
        """Log memory estimation results."""
        logger.info(f"Memory estimation: {trial_size_mb:.4f} MB per field line")
        logger.info(
            f"Estimated completion time: {completion_time:.1f} seconds "
            f"({completion_time/60:.1f} minutes, {completion_time/3600:.2f} hours)"
        )
        if self.verbose:
            logger.debug(f"Estimation based on {n_trial} trial field lines")

    def log_batch_size_setting(self, batch_size: int) -> None:
        """Log the batch size setting."""
        logger.info(f"Setting batch size to {batch_size} field lines")
        if self.verbose:
            logger.debug(f"Batch size configured for optimal memory usage")

    def log_calculation_start(self) -> None:
        """Log the start of field line calculations."""
        logger.info(f"Starting field line calculations at {time.strftime('%H:%M:%S')}")

    def log_calculation_completion(
        self,
        elapsed: float,
        save_results_to_disk: bool,
        filename: str = "",
        n_field_lines: int = 0,
    ) -> None:
        """Log the completion of field line calculations."""
        logger.info(f"Calculations completed in {elapsed:.1f} seconds")

        if save_results_to_disk:
            logger.info(f"Results saved to {filename}")
        else:
            logger.info(f"Returning {n_field_lines} field lines in memory")

    def close_progress_bar(self) -> None:
        """Close the current progress bar."""
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
