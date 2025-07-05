import logging
from multiprocessing import Pool
from typing import Callable, Optional
import numpy as np
from astropy.coordinates.representation import CartesianRepresentation

from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles batch processing for field line calculations."""
    def __init__(self, grid, batch_size, n_cores=1, progress_tracker: Optional[ProgressTracker]=None):
        self.grid = grid
        self.batch_size = batch_size
        self.n_cores = n_cores
        self.progress_tracker = progress_tracker

    def process(self, template: FieldLine, process_batch_fn: Callable[[FootpointGrid, FieldLine], FieldLineSet], add_results_fn: Callable, filename: str, save_results: bool):
        nline = self.grid.nline
        if nline is None:
            raise ValueError("Grid nline is None, cannot process batches")
        if self.grid.indices is None:
            raise ValueError("Grid indices is None, cannot process batches")
        logger.info(f"Processing {nline} field lines in batches of {self.batch_size}")
        batch_idx = 0
        total_processed = 0
        while total_processed < nline:
            start_idx = total_processed
            batch_indices = np.arange(start_idx, min(start_idx + self.batch_size, nline))
            if self.progress_tracker:
                self.progress_tracker.log_batch_start(batch_idx, start_idx, start_idx + len(batch_indices), nline)
            batch_grid = self.grid.subset_grid(batch_indices)
            batch_results = process_batch_fn(batch_grid, template)
            add_results_fn(batch_results, filename, batch_idx)
            total_processed += len(batch_indices)
            batch_idx += 1
            if self.progress_tracker:
                self.progress_tracker.log_batch_completion(batch_idx, total_processed, nline)
        # Optionally handle post-processing (e.g., renaming files) outside this class 