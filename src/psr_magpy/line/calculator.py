import logging
import os
import time
from multiprocessing import Pool
from dataclasses import dataclass
from tqdm import tqdm
from typing import Tuple, Optional

import numpy as np

from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.tracer import FieldLineTracer
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.line.memory_manager import ResultsManager, MemoryConfig
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TracingConfig:
    """Configuration for field line tracing parameters and output settings."""

    R_max: float = 10.0
    min_R_max_line: float = 1.0
    dt: float = 0.01
    max_memory_mb: float = 100.0
    n_cores: Optional[int] = None
    max_batch_size: int = 100
    save_results: bool = True
    filename: Optional[str] = "dummy"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.save_results and not self.filename:
            logger.error("Filename must be specified when save_results is True")
            raise ValueError("filename must be specified when save_results is True")
        if not self.save_results and self.filename:
            logger.warning("Filename specified but save_results is False")
            raise ValueError(
                "filename should not be specified when save_results is False"
            )

        if self.filename:
            directory = os.path.dirname(self.filename)
            if directory and not os.path.exists(directory):
                logger.error(f"Output directory does not exist: {directory}")
                raise ValueError(f"Output directory does not exist: {directory}")
                
        logger.debug(f"TracingConfig initialized: {self}")


class FieldLineCalculator:
    """Calculator for field lines based on a magnetic field and grid."""
    
    def __init__(self, field: MultipoleField, grid: FootpointGrid):
        """Initialize field line calculator.

        Parameters
        ----------
        field : MultipoleField
            Object that calculates the magnetic field
        grid : FootpointGrid
            Grid of footpoints to trace field lines from
        """
        self.field = field
        self.grid = grid
        logger.info(f"Initialized FieldLineCalculator with {grid.nline} footpoints")
        logger.debug(f"Field: {field}, Grid: {grid}")
        

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
        n_trial : int, optional
            Number of trial field lines to calculate, by default 100

        Returns
        -------
        Tuple[float, float, float]
            - batch_size: Calculated batch size
            - random_size: Size of random sample in MB
            - elapsed: Time taken for estimation in seconds
        """
        logger.info(f"Estimating memory requirements with {n_trial} trial field lines")
        time_start = time.time()
        n_trial = min(self.grid.nline, n_trial)

        trial_indices = np.arange(n_trial) #np.random.choice(self.grid.indices, size=n_trial, replace=False)
        trial_grid = self.grid.subset_grid(trial_indices)
        
        logger.debug(f"Selected {n_trial} random footpoints for estimation")

        trial_results = self._process_batch(
            trial_grid, template, tracer, results_manager.memory_config.n_cores
        )

        trial_size = trial_results.estimate_size() / n_trial
        trial_size_mb = trial_size / (1024 * 1024)

        logger.debug(f"Estimated memory size per line: {trial_size_mb:.4f} MB")

        elapsed = time.time() - time_start
        logger.debug(f"Time taken to estimate memory size: {elapsed:.2f} seconds")
        
        completion_time = elapsed * self.grid.nline / n_trial
        logger.info(
            f"Estimated time to complete calculation: {completion_time:.1f} seconds "
            f"({completion_time/60:.1f} minutes, {completion_time/3600:.2f} hours)"
        )

        return trial_size, completion_time

    def get_field_lines(
        self,
        template: FieldLine,
        config: Optional[TracingConfig] = None,
        max_batch_size: int = 100,
    ) -> Optional[FieldLineSet]:
        """Calculate field lines in batches and save to FITS file.
        
        Parameters
        ----------
        template : FieldLine
            Template field line containing configuration
        config : Optional[TracingConfig], optional
            Configuration for tracing, by default None
        max_batch_size : int, optional
            Maximum batch size, by default 100
            
        Returns
        -------
        Optional[FieldLineSet]
            Field line set if save_results is False, otherwise None
        """
        config = config or TracingConfig()
        logger.debug(f"Starting field line calculation with config: {config}")

        start_time = time.time()

        tracer = FieldLineTracer(self.field, config.R_max, config.min_R_max_line, config.dt)
        memory_config = MemoryConfig(
            max_memory_mb=config.max_memory_mb,
            max_batch_size=config.max_batch_size,
            n_cores=config.n_cores,
        )
        results_manager = ResultsManager(memory_config, tracer)
        logger.debug(f"Created ResultsManager with memory config: {memory_config}")

        trial_size, completion_time = (
            self._estimate_memory_requirements_and_time_for_completion(
                template, tracer, results_manager
            )
        )

        memory_max_batch_size = results_manager.find_batch_size(trial_size)
        batch_size = min(memory_config.max_batch_size, memory_max_batch_size)
        logger.info(f"Setting batch size to {batch_size} field lines")

        logger.info(f"Starting field line calculations at {time.strftime('%H:%M:%S')}")
        results = self._process_all_batches(
            template, config.filename or "", tracer, results_manager, config, batch_size
        )
        elapsed = time.time() - start_time

        logger.info(f"Calculations completed in {elapsed:.1f} seconds")
        
        if config.save_results:
            logger.info(f"Results saved to {config.filename}")
        else:
            logger.info(f"Returning {results.n_field_lines} field lines in memory")

        return results

    def _process_batch(
        self,
        subset_grid: FootpointGrid,
        template: FieldLine,
        tracer: FieldLineTracer,
        n_cores: Optional[int],
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
        logger.debug(f"Processing batch of {subset_grid.nline} field lines with {n_cores} cores")
        
        batch_args = [
            (subset_grid.footpoint_coordinates[idx], subset_grid.indices[idx], template)
            for idx in range(subset_grid.nline)
        ]

        with Pool(n_cores) as pool:
            logger.debug(f"Created multiprocessing pool with {n_cores} workers")
            line_list = pool.starmap(tracer.trace_single_line, batch_args)
            
        logger.debug(f"Processed {len(line_list)} field lines")
        result = FieldLineSet(field_lines=line_list, copy_data=True)
        return result

    def _process_all_batches(
        self,
        template: FieldLine,
        filename: str,
        tracer: FieldLineTracer,
        results_manager: ResultsManager,
        config: TracingConfig,
        batch_size: int,
    ) -> Optional[FieldLineSet]:
        """Process all field lines in batches.
        
        Parameters
        ----------
        template : FieldLine
            Template field line containing configuration
        filename : str
            Output filename for results
        tracer : FieldLineTracer
            Tracer object for field line calculations
        results_manager : ResultsManager
            Manager for handling memory and results
        config : TracingConfig
            Configuration for tracing
        batch_size : int
            Batch size for processing
            
        Returns
        -------
        Optional[FieldLineSet]
            Field line set if save_results is False, otherwise None
        """
        
        
        all_indices = self.grid.indices
        results = None
        if self.grid.nline < batch_size:
            total_batches = 1
            batch_size = self.grid.nline
        else:
            total_batches = (self.grid.nline + batch_size - 1) // batch_size
            
        logger.info(f"Processing {self.grid.nline} field lines in batches of {batch_size}")
        logger.debug(f"Total number of batches: {total_batches}")
    

        with tqdm(total=self.grid.nline, desc="Processing field lines") as pbar:
            for start_idx in range(0, self.grid.nline, batch_size):
                batch_indices = all_indices[start_idx : start_idx + batch_size]
                subset_grid = self.grid.subset_grid(batch_indices)
                
                batch_num = (start_idx // batch_size) + 1
                logger.debug(f"Processing batch {batch_num}/{total_batches} with {len(batch_indices)} field lines")

                batch_results = self._process_batch(
                    subset_grid, template, tracer, results_manager.memory_config.n_cores
                )

                results = results_manager.add_results(
                    batch_results, filename, start_idx
                )

                # Update progress bar
                pbar.update(len(batch_indices))
                pbar.set_postfix(
                    {
                        "Batch": f"{batch_num}/{total_batches}",
                        "Lines": f"{min(start_idx + batch_size, self.grid.nline)}/{self.grid.nline}",
                    }
                )
                

        if config.save_results:
            
            logger.info(f"Writing final results to disk: {filename}")
            results_manager.write_results_to_disk(filename, start_idx)
            results_manager.rename_batch_files(filename)
            logger.info("All batch files renamed and finalized")
            return None
        else:
            logger.info(f"Returning complete results with {results.n_field_lines} field lines")
            return results

