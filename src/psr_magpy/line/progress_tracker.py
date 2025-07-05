"""Progress tracking module for field line calculations.

Provides classes for tracking progress, timing, and logging during
field line calculations with support for progress bars and detailed logging.

Classes:
    ProgressTracker: Tracks progress, timing, and logging for field line calculations

Example:
    >>> from psr_magpy.line.progress_tracker import ProgressTracker
    >>> tracker = ProgressTracker(verbose=True, show_progress=True)
    >>> with tracker.track_calculation("Field line calculation"):
    ...     # Your calculation code here
    ...     tracker.update_progress(50, 100, "Processing batch 1")
    ...     tracker.log_completion_time()
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Any

from tqdm import tqdm

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class ProgressTracker:
    """Tracks progress, timing, and logging for field line calculations.
    
    Provides comprehensive progress tracking including timing, progress bars,
    and detailed logging throughout the calculation process.
    
    Attributes:
        verbose: Whether to enable verbose logging
        show_progress: Whether to show progress bars
        start_time: Start time of the current calculation
        _progress_bar: Current progress bar instance
    """
    
    def __init__(self, verbose: bool = False, show_progress: bool = True):
        """Initialize progress tracker.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to enable verbose logging, by default False
        show_progress : bool, optional
            Whether to show progress bars, by default True
        """
        self.verbose = verbose
        self.show_progress = show_progress
        self.start_time: Optional[float] = None
        self._progress_bar: Optional[tqdm] = None
        
        logger.debug(f"Initialized ProgressTracker: verbose={verbose}, show_progress={show_progress}")
    
    @contextmanager
    def track_calculation(self, description: str):
        """Context manager for tracking calculation progress.
        
        Parameters
        ----------
        description : str
            Description of the calculation being tracked
        """
        self.start_calculation(description)
        try:
            yield self
        finally:
            self.end_calculation()
    
    def start_calculation(self, description: str) -> None:
        """Start tracking a new calculation.
        
        Parameters
        ----------
        description : str
            Description of the calculation being tracked
        """
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
    
    def log_completion_time(self) -> None:
        """Log the completion time if calculation is in progress."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            logger.info(f"Elapsed time: {elapsed:.1f} seconds")
    
    def create_progress_bar(self, total: int, description: str = "Processing") -> Any:
        """Create a progress bar for tracking progress.
        
        Parameters
        ----------
        total : int
            Total number of items to process
        description : str, optional
            Description for the progress bar, by default "Processing"
            
        Returns
        -------
        tqdm or DummyProgressBar
            Progress bar instance
        """
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
            total=total,
            desc=description,
            unit="items",
            disable=not self.show_progress
        )
        logger.debug(f"Created progress bar: {description} ({total} items)")
        return self._progress_bar
    
    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress and log message.
        
        Parameters
        ----------
        current : int
            Current progress count
        total : int
            Total items to process
        message : str, optional
            Optional message to log, by default ""
        """
        if self._progress_bar is not None:
            self._progress_bar.n = current
            self._progress_bar.refresh()
        
        if message and self.verbose:
            logger.debug(f"Progress: {current}/{total} - {message}")
    
    def log_batch_start(self, batch_idx: int, start_idx: int, end_idx: int, total: int) -> None:
        """Log the start of a batch processing.
        
        Parameters
        ----------
        batch_idx : int
            Current batch index
        start_idx : int
            Starting index for this batch
        end_idx : int
            Ending index for this batch
        total : int
            Total number of items
        """
        batch_size = end_idx - start_idx
        logger.info(f"Processing batch {batch_idx + 1}: indices {start_idx}-{end_idx-1} ({batch_size} items)")
        if self.verbose:
            logger.debug(f"Batch {batch_idx + 1}: {batch_size}/{total} items ({batch_size/total*100:.1f}%)")
    
    def log_batch_completion(self, batch_idx: int, processed: int, total: int) -> None:
        """Log the completion of a batch.
        
        Parameters
        ----------
        batch_idx : int
            Completed batch index
        processed : int
            Total items processed so far
        total : int
            Total number of items
        """
        logger.info(f"Completed batch {batch_idx + 1}: {processed}/{total} field lines")
        if self.verbose:
            logger.debug(f"Progress: {processed/total*100:.1f}% complete")
    
    def log_memory_estimation(self, n_trial: int, trial_size_mb: float, completion_time: float) -> None:
        """Log memory estimation results.
        
        Parameters
        ----------
        n_trial : int
            Number of trial field lines used
        trial_size_mb : float
            Estimated size per field line in MB
        completion_time : float
            Estimated completion time in seconds
        """
        logger.info(f"Memory estimation: {trial_size_mb:.4f} MB per field line")
        logger.info(
            f"Estimated completion time: {completion_time:.1f} seconds "
            f"({completion_time/60:.1f} minutes, {completion_time/3600:.2f} hours)"
        )
        if self.verbose:
            logger.debug(f"Estimation based on {n_trial} trial field lines")
    
    def log_batch_size_setting(self, batch_size: int) -> None:
        """Log the batch size setting.
        
        Parameters
        ----------
        batch_size : int
            Set batch size
        """
        logger.info(f"Setting batch size to {batch_size} field lines")
        if self.verbose:
            logger.debug(f"Batch size configured for optimal memory usage")
    
    def log_calculation_start(self) -> None:
        """Log the start of field line calculations."""
        logger.info(f"Starting field line calculations at {time.strftime('%H:%M:%S')}")
    
    def log_calculation_completion(self, elapsed: float, save_results: bool, filename: str = "", 
                                 n_field_lines: int = 0) -> None:
        """Log the completion of field line calculations.
        
        Parameters
        ----------
        elapsed : float
            Total elapsed time in seconds
        save_results : bool
            Whether results were saved to disk
        filename : str, optional
            Output filename if saved, by default ""
        n_field_lines : int, optional
            Number of field lines returned, by default 0
        """
        logger.info(f"Calculations completed in {elapsed:.1f} seconds")
        
        if save_results:
            logger.info(f"Results saved to {filename}")
        else:
            logger.info(f"Returning {n_field_lines} field lines in memory")
    
    def close_progress_bar(self) -> None:
        """Close the current progress bar."""
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None 