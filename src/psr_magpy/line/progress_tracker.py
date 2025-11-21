"""Progress tracking module for field line calculations.

Provides classes for tracking progress, timing, and progress bars during
field line calculations. Logging is handled separately via ProgressReporter.
"""

import time
from contextlib import contextmanager
from typing import Optional, Union, Protocol, Any
from pathlib import Path

from tqdm import tqdm

from psr_magpy.utils.logging_config import get_logger


class ProgressBarProtocol(Protocol):
    """Protocol for progress bar interface."""
    
    n: int
    total: int
    
    def refresh(self) -> None:
        """Refresh the progress bar display."""
        ...
    
    def close(self) -> None:
        """Close the progress bar."""
        ...


class _DummyProgressBar:
    """Dummy progress bar that does nothing."""
    
    def __init__(self) -> None:
        self.n = 0
        self.total = 0

    def refresh(self) -> None:
        """No-op refresh."""
        pass

    def close(self) -> None:
        """No-op close."""
        pass


class ProgressTracker:
    """Tracks progress, timing, and progress bars for field line calculations and I/O operations.
    
    This class manages state only (timing, progress bars). Logging is handled separately
    via ProgressReporter which combines this class with a logger.
    """

    def __init__(self, verbose: bool = False, show_progress: bool = True):
        """Initialize progress tracker."""
        self.verbose = verbose
        self.show_progress = show_progress
        self.start_time: Optional[float] = None
        self._progress_bar: Optional[tqdm] = None
        self._io_progress_bar: Optional[tqdm] = None

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

    def end_calculation(self) -> Optional[float]:
        """End the current calculation and return elapsed time."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.start_time = None
            return elapsed
        return None

    def start_io_operation(
        self,
        operation: str,
        filepath: Union[str, Path],
        data_size: Optional[int] = None,
    ) -> str:
        """Start tracking an I/O operation and return operation description."""
        self.start_time = time.time()
        filepath = Path(filepath)
        operation_desc = f"{operation.capitalize()}ing {filepath.name}"

        # Create progress bar for I/O operations
        if self.show_progress and data_size:
            self._io_progress_bar = tqdm(
                total=data_size,
                desc=operation_desc,
                unit="items",
                disable=not self.show_progress,
            )
        
        return operation_desc

    def end_io_operation(self) -> Optional[float]:
        """End the current I/O operation and return elapsed time."""
        elapsed = None
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.start_time = None

        # Close I/O progress bar
        if self._io_progress_bar is not None:
            self._io_progress_bar.close()
            self._io_progress_bar = None
        
        return elapsed

    def update_io_progress(
        self, current: int, total: Optional[int] = None, message: str = ""
    ) -> None:
        """Update I/O operation progress."""
        if self._io_progress_bar is not None:
            self._io_progress_bar.n = current
            if total:
                self._io_progress_bar.total = total
            self._io_progress_bar.refresh()

    def get_elapsed_time(self) -> Optional[float]:
        """Get elapsed time if calculation is in progress."""
        if self.start_time is not None:
            return time.time() - self.start_time
        return None
    
    def get_file_size_mb(self, filepath: Union[str, Path]) -> Optional[float]:
        """Get file size in MB if file exists."""
        filepath = Path(filepath)
        if filepath.exists():
            return filepath.stat().st_size / (1024 * 1024)
        return None

    def create_progress_bar(self, total: int, description: str = "Processing") -> Union[tqdm, _DummyProgressBar]:
        """Create a progress bar for tracking progress."""
        if not self.show_progress:
            return _DummyProgressBar()
        self._progress_bar = tqdm(
            total=total, desc=description, unit="items", disable=not self.show_progress
        )
        return self._progress_bar

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress bar."""
        if self._progress_bar is not None:
            self._progress_bar.n = current
            self._progress_bar.refresh()

    def get_batch_info(self, batch_idx: int, start_idx: int, end_idx: int, total: int) -> dict:
        """Get batch information for logging."""
        batch_size = end_idx - start_idx
        return {
            "batch_idx": batch_idx,
            "batch_number": batch_idx + 1,
            "start_idx": start_idx,
            "end_idx": end_idx - 1,
            "batch_size": batch_size,
            "total": total,
            "percentage": batch_size / total * 100 if total > 0 else 0.0,
        }
    
    def get_memory_info(self) -> dict:
        """Get current process memory information."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
        }

    def close_progress_bar(self) -> None:
        """Close the current progress bar."""
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None


class ProgressReporter:
    """Unified interface for progress tracking and logging.
    
    Combines ProgressTracker (state/timing/progress bars) with logger (structured logging)
    to provide a single abstraction for progress reporting.
    """

    def __init__(self, verbose: bool = False, show_progress: bool = True, logger_name: Optional[str] = None):
        """Initialize progress reporter.
        
        Parameters
        ----------
        verbose : bool
            Enable verbose logging
        show_progress : bool
            Show progress bars
        logger_name : str, optional
            Logger name (defaults to module name)
        """
        self.tracker = ProgressTracker(verbose=verbose, show_progress=show_progress)
        self.verbose = verbose
        self.logger = get_logger(logger_name or __name__, console_format_brief=True)

    @property
    def start_time(self) -> Optional[float]:
        """Get start time from tracker."""
        return self.tracker.start_time

    @contextmanager
    def track_calculation(self, description: str):
        """Context manager for tracking calculation progress with logging."""
        self.report_calculation_start(description)
        try:
            yield self
        finally:
            self.report_calculation_end()

    @contextmanager
    def track_io_operation(
        self,
        operation: str,
        filepath: Union[str, Path],
        data_size: Optional[int] = None,
    ):
        """Context manager for tracking I/O operations with logging."""
        self.report_io_start(operation, filepath, data_size)
        try:
            yield self
        finally:
            self.report_io_end(operation, filepath)

    def report_calculation_start(self, description: str) -> None:
        """Report the start of a calculation."""
        self.tracker.start_calculation(description)
        self.logger.info(f"Starting: {description}")
        if self.verbose:
            self.logger.debug(f"Calculation started at {time.strftime('%H:%M:%S')}")

    def report_calculation_end(self) -> Optional[float]:
        """Report the end of a calculation and return elapsed time."""
        elapsed = self.tracker.end_calculation()
        if elapsed is not None:
            self.logger.info(f"Calculation completed in {elapsed:.1f} seconds")
            if self.verbose:
                self.logger.debug(f"Calculation ended at {time.strftime('%H:%M:%S')}")
        return elapsed

    def report_io_start(
        self,
        operation: str,
        filepath: Union[str, Path],
        data_size: Optional[int] = None,
    ) -> None:
        """Report the start of an I/O operation."""
        operation_desc = self.tracker.start_io_operation(operation, filepath, data_size)
        self.logger.info(f"Starting I/O: {operation_desc}")
        if self.verbose:
            self.logger.debug(f"I/O operation started at {time.strftime('%H:%M:%S')}")
            if data_size:
                self.logger.debug(f"Data size: {data_size} items")

    def report_io_end(
        self,
        operation: str,
        filepath: Union[str, Path],
    ) -> Optional[float]:
        """Report the end of an I/O operation and return elapsed time."""
        elapsed = self.tracker.end_io_operation()
        if elapsed is not None:
            filepath = Path(filepath)
            self.logger.info(f"I/O operation completed in {elapsed:.1f} seconds")
            if self.verbose:
                self.logger.debug(f"I/O operation ended at {time.strftime('%H:%M:%S')}")
        return elapsed

    def update_io_progress(
        self, current: int, total: Optional[int] = None, message: str = ""
    ) -> None:
        """Update I/O operation progress."""
        self.tracker.update_io_progress(current, total, message)
        if message and self.verbose:
            self.logger.debug(f"I/O Progress: {current}/{total or 'unknown'} - {message}")

    def report_io_start_simple(
        self,
        operation: str,
        filepath: Union[str, Path],
        data_size: Optional[int] = None,
    ) -> None:
        """Report the start of an I/O operation (simple version without progress bar)."""
        filepath = Path(filepath)
        self.logger.info(f"Starting {operation}: {filepath.name}")
        if self.verbose and data_size:
            self.logger.debug(f"Data size: {data_size} items")

    def report_io_completion(
        self,
        operation: str,
        filepath: Union[str, Path],
        elapsed: float,
        data_size: Optional[int] = None,
    ) -> None:
        """Report the completion of an I/O operation."""
        filepath = Path(filepath)
        self.logger.info(f"Completed {operation}: {filepath.name} in {elapsed:.1f} seconds")
        if self.verbose and data_size:
            self.logger.debug(f"Processed {data_size} items")

    def report_file_size(self, filepath: Union[str, Path]) -> None:
        """Report the size of a file."""
        size_mb = self.tracker.get_file_size_mb(filepath)
        if size_mb is not None:
            self.logger.info(f"File size: {size_mb:.2f} MB")
            if self.verbose:
                self.logger.debug(f"File: {filepath}")

    def report_completion_time(self) -> None:
        """Report the completion time if calculation is in progress."""
        elapsed = self.tracker.get_elapsed_time()
        if elapsed is not None:
            self.logger.info(f"Elapsed time: {elapsed:.1f} seconds")

    def create_progress_bar(self, total: int, description: str = "Processing") -> Union[tqdm, _DummyProgressBar]:
        """Create a progress bar for tracking progress."""
        progress_bar = self.tracker.create_progress_bar(total, description)
        if self.verbose:
            self.logger.debug(f"Created progress bar: {description} ({total} items)")
        return progress_bar

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress and optionally log message."""
        self.tracker.update_progress(current, total, message)
        if message and self.verbose:
            self.logger.debug(f"Progress: {current}/{total} - {message}")

    def report_batch_start(
        self, batch_idx: int, start_idx: int, end_idx: int, total: int
    ) -> None:
        """Report the start of a batch processing."""
        batch_info = self.tracker.get_batch_info(batch_idx, start_idx, end_idx, total)
        self.logger.info(
            f"Processing batch {batch_info['batch_number']}: "
            f"indices {batch_info['start_idx']}-{batch_info['end_idx']} "
            f"({batch_info['batch_size']} items)"
        )
        if self.verbose:
            self.logger.debug(
                f"Batch {batch_info['batch_number']}: "
                f"{batch_info['batch_size']}/{batch_info['total']} items "
                f"({batch_info['percentage']:.1f}%)"
            )

    def report_batch_completion(
        self, batch_idx: int, processed: int, total: int, field_line_set: Optional[Any] = None
    ) -> None:
        """Report the completion of a batch with memory usage information."""
        self.logger.info(f"Completed batch {batch_idx + 1}: {processed}/{total} field lines")

        memory_info = self.tracker.get_memory_info()
        self.logger.info(f"Memory usage after batch {batch_idx + 1}: {memory_info['rss_mb']:.1f} MB")

        # Get field line data size if available
        if field_line_set is not None:
            try:
                data_size_bytes = field_line_set.estimate_size()
                data_size_mb = data_size_bytes / (1024 * 1024)
                self.logger.info(
                    f"Field line data size in memory: {data_size_mb:.1f} MB ({data_size_bytes:,} bytes)"
                )
            except Exception as e:
                self.logger.warning(f"Could not estimate field line data size: {e}")

        if self.verbose:
            self.logger.debug(f"Progress: {processed/total*100:.1f}% complete")
            self.logger.debug(
                f"Memory details: RSS={memory_info['rss_mb']:.1f} MB, "
                f"VMS={memory_info['vms_mb']:.1f} MB"
            )

    def report_memory_estimation(
        self, n_trial: int, trial_size_mb: float, completion_time: float
    ) -> None:
        """Report memory estimation results."""
        self.logger.info(f"Memory estimation: {trial_size_mb:.4f} MB per field line")
        self.logger.info(
            f"Estimated completion time: {completion_time:.1f} seconds "
            f"({completion_time/60:.1f} minutes, {completion_time/3600:.2f} hours)"
        )
        if self.verbose:
            self.logger.debug(f"Estimation based on {n_trial} trial field lines")

    def report_batch_size_setting(self, batch_size: int) -> None:
        """Report the batch size setting."""
        self.logger.info(f"Setting batch size to {batch_size} field lines")
        if self.verbose:
            self.logger.debug(f"Batch size configured for optimal memory usage")

    def report_calculation_start_simple(self) -> None:
        """Report the start of field line calculations."""
        self.logger.info(f"Starting field line calculations at {time.strftime('%H:%M:%S')}")

    def report_calculation_completion(
        self,
        elapsed: float,
        save_results_to_disk: bool,
        filename: str = "",
        n_field_lines: int = 0,
        batch_files_created: bool = False,
    ) -> None:
        """Report the completion of field line calculations."""
        self.logger.info(f"Calculations completed in {elapsed:.1f} seconds")

        if save_results_to_disk and batch_files_created:
            self.logger.info(f"Results saved to {filename[:-5]} in batches")
        elif save_results_to_disk and not batch_files_created:
            self.logger.info(f"Results will be saved to {filename[:-5]} in batches")
        else:
            self.logger.info(f"Returning {n_field_lines} field lines in memory")

    def close_progress_bar(self) -> None:
        """Close the current progress bar."""
        self.tracker.close_progress_bar()
