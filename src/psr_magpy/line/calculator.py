import time
import datetime
import numpy as np
import psutil
from typing import Tuple, List, Optional, Dict, Any
import os
import logging

from src.psr_magpy.line.tracer import FieldLineTracer
from src.psr_magpy.grid.grid import FootpointGrid
from src.psr_magpy.line.field_line import FieldLine
from src.psr_magpy.line.field_line_set import FieldLineSet

import matplotlib.pyplot as plt
from src.psr_magpy.field.multipole import MultipoleField

from multiprocessing import Pool
from src.psr_magpy.line.memory_manager import ResultsManager, MemoryConfig
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class TracingConfig:
    """Configuration for field line tracing parameters and output settings."""
    R_max: float = 10.0
    R_min: float = 1.0
    dt: float = 0.01
    max_memory_mb: float = 100.0
    n_cores: Optional[int] = None
    max_batch_size: int = 100
    save_results: bool = True
    filename: Optional[str] = "dummy"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.save_results and not self.filename:
            raise ValueError("filename must be specified when save_results is True")
        if not self.save_results and self.filename:
            raise ValueError("filename should not be specified when save_results is False")
        
        if self.filename:
            directory = os.path.dirname(self.filename)
            if directory and not os.path.exists(directory):
                raise ValueError(f"Output directory does not exist: {directory}")

class FieldLineCalculator:
    def __init__(self, field: MultipoleField, grid: FootpointGrid):
        """Initialize field line calculator.

        Parameters
        ----------
        field : MultipoleField
            Object that calculates the magnetic field
        """
        self.field = field
        self.grid = grid

    def _estimate_memory_requirements_and_time_for_completion(
        self,
        template: FieldLine,
        tracer: FieldLineTracer,
        results_manager: ResultsManager,
        n_trial: int = 100,
    ) -> Tuple[float, float, float]:
        """Estimate memory requirements by processing a small random subset.
        
        Parameters
        ----------
        template : FieldLine
            Template field line containing configuration
        tracer : FieldLineTracer
            Tracer object for field line calculations
        results_manager : ResultsManager
            Manager for handling memory and results
            
        Returns
        -------
        Tuple[float, float, float]
            - batch_size: Calculated batch size
            - random_size: Size of random sample in MB
            - elapsed: Time taken for estimation in seconds
        """
        time_start = time.time()
        n_trial = min(self.grid.nline, n_trial)  
        
        trial_indices = np.random.choice(self.grid.nline, size=n_trial, replace=False)
        trial_grid = self.grid.subset_grid(trial_indices)
        logger.info("Selected %d random footpoints for field line calculation", n_trial)
        
        trial_results = self._process_batch(trial_grid, template, tracer, results_manager.memory_config.n_cores)
        
        trial_size = trial_results.estimate_size() / n_trial
        
        logger.info("Estimated memory size per line: %.2f MB", trial_size/(1024*1024))

        elapsed = time.time() - time_start
        logger.info("Time taken to estimate memory size: %.1f seconds", elapsed)
        completion_time = elapsed * self.grid.nline / n_trial
        logger.info("Estimated time to complete calculation: %.1f seconds or %.1f minutes or %.1f hours", completion_time, completion_time / 60, completion_time / 3600)
        
        return trial_size, completion_time

    def get_field_lines(
        self,
        template: FieldLine,
        config: Optional[TracingConfig] = None,
        max_batch_size: int = 100,
    ) -> Optional[FieldLineSet]:
        """Calculate field lines in batches and save to FITS file."""
        config = config or TracingConfig()
        print(config.save_results)
        
        start_time = time.time()
        
        tracer = FieldLineTracer(self.field, config.R_max, config.R_min, config.dt)
        memory_config = MemoryConfig(max_memory_mb=config.max_memory_mb, max_batch_size=config.max_batch_size, n_cores=config.n_cores)
        results_manager = ResultsManager(memory_config, tracer)
        
        trial_size, completion_time = self._estimate_memory_requirements_and_time_for_completion(
            template, tracer, results_manager
        )
   
        memory_max_batch_size = results_manager.find_batch_size(trial_size)
        batch_size = min(memory_config.max_batch_size, memory_max_batch_size)
        logger.info(f"Setting batch size: {batch_size}")
        
        logger.info(f"Starting field line calculations at {time.strftime('%H:%M:%S')}")
        results = self._process_all_batches(
            template, config.filename or "", tracer, results_manager, config, batch_size
        )
        elapsed = time.time() - start_time
        logger.info(f"\nCalculations completed in {elapsed:.1f} seconds")
        
        return results

    def _process_batch(
        self,
        subset_grid: FootpointGrid,
        template: FieldLine,
        tracer: FieldLineTracer,
        n_cores: Optional[int]
    ) -> FieldLineSet:
        """Process a batch of field lines using multiprocessing.
        
        Parameters
        ----------
        subset_grid : FootpointGrid
            Grid containing the batch of footpoints to process
        template : FieldLine
            Template field line containing configuration
        tracer : FieldLineTracer
            Tracer object for field line calculations
        n_cores : Optional[int]
            Number of cores to use for multiprocessing
            
        Returns
        -------
        FieldLineSet
            Processed batch results
        """
        batch_args = [
            (subset_grid.footpoint_coordinates[idx], subset_grid.indices[idx], template)
            for idx in range(subset_grid.nline)
        ]
        
        with Pool(n_cores) as pool:
            line_list = pool.starmap(tracer.trace_single_line, batch_args)
        result = FieldLineSet(field_lines=line_list, copy_data=True)
        return result

    def _process_all_batches(
        self,
        template: FieldLine,
        filename: str,
        tracer: FieldLineTracer,
        results_manager: ResultsManager,
        config: TracingConfig,
        batch_size: int
    ) -> Optional[FieldLineSet]:


        all_indices = self.grid.indices
        results = None
        total_batches = (self.grid.nline + batch_size - 1) // batch_size

        with tqdm(total=self.grid.nline, desc="Processing field lines") as pbar:
            for start_idx in range(0, self.grid.nline, batch_size):
                batch_indices = all_indices[start_idx : start_idx + batch_size]
                subset_grid = self.grid.subset_grid(batch_indices)
                
                batch_results = self._process_batch(
                    subset_grid,
                    template,
                    tracer,
                    results_manager.memory_config.n_cores
                )
                
                results = results_manager.add_results(batch_results, filename, start_idx)
               
                # Update progress bar
                pbar.update(len(batch_indices))
                pbar.set_postfix({
                    'Batch': f"{(start_idx // batch_size) + 1}/{total_batches}",
                    'Lines': f"{min(start_idx + batch_size, self.grid.nline)}/{self.grid.nline}"
                })

        if config.save_results:
            #logger.info("Writing batch %d to disk", batch_idx)
            results_manager.write_results_to_disk(filename, start_idx)
            results_manager.rename_batch_files(filename)
            return None            
        else:
            return results

    def _print_memory_stats(self, results_manager: ResultsManager) -> None:
        stats = results_manager.get_memory_stats()
        tqdm.write(
            f"Memory usage: {stats['percent_used']}% "
            f"(Available: {stats['available_gb']:.1f} GB)"
        )
