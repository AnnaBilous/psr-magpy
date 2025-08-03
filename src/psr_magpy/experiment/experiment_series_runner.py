"""Experiment series runner for executing multiple experiments.

This module provides functionality to run multiple experiments from an experiment series
configuration, with progress tracking and result management.
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

import yaml

from .experiment_series import ExperimentSeriesConfig
from .experiment_config import ExperimentConfig
from .experiment_runner import ExperimentRunner, run_experiment
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    experiment_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    results: Optional[Any] = None


@dataclass
class SeriesResult:
    """Result from running an experiment series."""
    series_name: str
    total_experiments: int
    successful_experiments: int
    failed_experiments: int
    total_duration: float
    experiment_results: List[ExperimentResult]
    output_directory: Optional[Path] = None


class ExperimentSeriesRunner:
    """Runner for executing experiment series."""
    
    def __init__(self, 
                 series_config: ExperimentSeriesConfig,
                 output_directory: Optional[Path] = None,
                 save_individual_configs: bool = True,
                 save_grids: bool = True,
                 continue_on_failure: bool = True,
                 max_log_size: int = 10 * 1024 * 1024,  # 10 MB
                 log_backup_count: int = 5):
        """Initialize the series runner.
        
        Args:
            series_config: The experiment series configuration
            output_directory: Directory to save results and configs
            save_individual_configs: Whether to save individual experiment configs
            save_grids: Whether to save grids for each experiment
            continue_on_failure: Whether to continue if individual experiments fail
            max_log_size: Maximum size of log file before rotation (in bytes)
            log_backup_count: Number of backup log files to keep
        """
        self.series_config = series_config
        self.output_directory = Path(output_directory) if output_directory else Path("experiment_series_output")
        self.save_individual_configs = save_individual_configs
        self.save_grids = save_grids
        self.continue_on_failure = continue_on_failure
        self.max_log_size = max_log_size
        self.log_backup_count = log_backup_count
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Set up file logging for the series
        self._setup_series_logging()
        
        logger.info(f"Initialized series runner for: {series_config.name}")
        logger.info(f"Output directory: {self.output_directory}")
        logger.info(f"Save grids: {self.save_grids}")
        logger.info(f"Log rotation: {self.max_log_size / (1024*1024):.1f} MB max, {self.log_backup_count} backups")
    
    def _setup_series_logging(self) -> None:
        """Set up file logging for the experiment series."""
        # Use the series-specific logs directory
        logs_dir = self.output_directory / self.series_config.name / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Clean up old log files (keep only the 5 most recent)
        self._cleanup_old_logs(logs_dir)
        
        # Create log filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{self.series_config.name}_{timestamp}.log"
        
        # Set up file logging with custom rotation settings
        from psr_magpy.utils.logging_config import setup_logging
        setup_logging(
            log_file=log_file,
            log_level=logging.INFO,
            handler_levels={"console": logging.INFO, "file": logging.DEBUG},
            max_file_size=self.max_log_size,
            backup_count=self.log_backup_count
        )
        
        logger.info(f"Series logging set up: {log_file}")
        logger.info(f"Log rotation: {self.max_log_size / (1024*1024):.1f} MB max, {self.log_backup_count} backups")
    
    def _cleanup_old_logs(self, logs_dir: Path) -> None:
        """Clean up old log files, keeping only the most recent ones."""
        import glob
        
        # Find all log files for this series
        pattern = str(logs_dir / f"{self.series_config.name}_*.log")
        log_files = glob.glob(pattern)
        
        if len(log_files) <= 5:  # Keep up to 5 log files
            return
        
        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        # Remove old files (keep only the 5 most recent)
        files_to_remove = log_files[5:]
        for file_path in files_to_remove:
            try:
                Path(file_path).unlink()
                logger.debug(f"Removed old log file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old log file {file_path}: {e}")
        
        if files_to_remove:
            logger.info(f"Cleaned up {len(files_to_remove)} old log files, keeping 5 most recent")
    
    def run(self) -> SeriesResult:
        """Run the complete experiment series."""
        start_time = time.time()
        logger.info(f"Starting experiment series: {self.series_config.name}")
        
        # Expand all experiments
        experiments = self.series_config.create_experiment_series()
        logger.info(f"Expanded to {len(experiments)} experiments")
        
        # Save series configuration to configs directory
        configs_dir = self.output_directory / self.series_config.name / "configs"
        configs_dir.mkdir(exist_ok=True)
        series_config_path = configs_dir / f"{self.series_config.name}_config.yaml"
        from psr_magpy.io import IOFactory
        io_instance = IOFactory.create_config_io(type(self.series_config))
        io_instance.save(self.series_config, series_config_path, overwrite=True)
        logger.info(f"Saved series config to: {series_config_path}")
        
        # Save individual configs if requested
        if self.save_individual_configs:
            self._save_individual_configs(experiments)
        
        # Run experiments
        experiment_results = self._run_experiments(experiments)
        
        # Calculate summary
        total_duration = time.time() - start_time
        successful = sum(1 for r in experiment_results if r.success)
        failed = len(experiment_results) - successful
        
        # Create series result
        series_result = SeriesResult(
            series_name=self.series_config.name or "unnamed_series",
            total_experiments=len(experiments),
            successful_experiments=successful,
            failed_experiments=failed,
            total_duration=total_duration,
            experiment_results=experiment_results,
            output_directory=self.output_directory
        )
        
        # Log summary
        logger.info(f"Series completed in {total_duration:.2f}s")
        logger.info(f"Successful: {successful}/{len(experiments)}")
        logger.info(f"Failed: {failed}/{len(experiments)}")
        
        # Save results summary
        self._save_results_summary(series_result)
        
        return series_result
    
    def _save_individual_configs(self, experiments: List[ExperimentConfig]) -> None:
        """Save individual experiment configurations."""
        # Use the series-specific configs directory
        configs_dir = self.output_directory / self.series_config.name / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving {len(experiments)} individual configs to {configs_dir}")
        
        from psr_magpy.io import IOFactory
        for experiment in experiments:
            config_path = configs_dir / f"{experiment.name}.yaml"
            io_instance = IOFactory.create_config_io(type(experiment))
            io_instance.save(experiment, config_path, overwrite=True)
        
        logger.info("Individual configs saved")
    

    
    def _run_experiments(self, experiments: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        results = []
        
        # Use the series-specific lines directory
        lines_dir = self.output_directory / self.series_config.name / "lines"
        lines_dir.mkdir(exist_ok=True)
        
        for i, experiment in enumerate(experiments, 1):
            logger.info(f"Running experiment {i}/{len(experiments)}: {experiment.name}")
            
            try:
                # Configure experiment to save results to lines directory BEFORE creating runner
                experiment.tracing.save_results = True
                experiment.tracing.filename = str(lines_dir / f"{experiment.name}.fits")
                experiment.tracing.overwrite = True
                
                # Create field and grid for this experiment
                field = experiment.field.create_field()
                grid = experiment.grid.create_grid()
                
                start_time = time.time()
                experiment_result = run_experiment(experiment)
                duration = time.time() - start_time
                
                result = ExperimentResult(
                    experiment_name=experiment.name,
                    success=True,
                    duration=duration,
                    results=experiment_result
                )
                logger.info(f"✓ Experiment {experiment.name} completed in {duration:.2f}s")
                
            except Exception as e:
                duration = time.time() - start_time
                result = ExperimentResult(
                    experiment_name=experiment.name,
                    success=False,
                    duration=duration,
                    error_message=str(e)
                )
                logger.error(f"✗ Experiment {experiment.name} failed after {duration:.2f}s: {e}")
                
                if not self.continue_on_failure:
                    raise
            
            results.append(result)
        
        return results
    

    
    def _save_results_summary(self, series_result: SeriesResult) -> None:
        """Save a summary of the series results."""
        summary_path = self.output_directory / self.series_config.name / f"{self.series_config.name}_results_summary.yaml"
        
        summary = {
            "series_name": series_result.series_name,
            "total_experiments": series_result.total_experiments,
            "successful_experiments": series_result.successful_experiments,
            "failed_experiments": series_result.failed_experiments,
            "total_duration": series_result.total_duration,
            "success_rate": series_result.successful_experiments / series_result.total_experiments,
            "experiment_results": [
                {
                    "name": r.experiment_name,
                    "success": r.success,
                    "duration": r.duration,
                    "error": r.error_message
                }
                for r in series_result.experiment_results
            ]
        }
        
        # Use I/O factory for saving
        from psr_magpy.io import IOUtils
        IOUtils.ensure_directory(summary_path)
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        logger.info(f"Results summary saved to: {summary_path}")


def run_experiment_series(series_config: ExperimentSeriesConfig,
                         output_directory: Optional[Path] = None,
                         save_individual_configs: bool = True,
                         save_grids: bool = True,
                         continue_on_failure: bool = True,
                         max_log_size: int = 10 * 1024 * 1024,  # 10 MB
                         log_backup_count: int = 5) -> SeriesResult:
    """Convenience function to run an experiment series."""
    runner = ExperimentSeriesRunner(
        series_config=series_config,
        output_directory=output_directory,
        save_individual_configs=save_individual_configs,
        save_grids=save_grids,
        continue_on_failure=continue_on_failure,
        max_log_size=max_log_size,
        log_backup_count=log_backup_count
    )
    return runner.run()


def run_experiment_series_from_file(input_directory: str,
                                   experiment_name: str,
                                   output_directory: Optional[Path] = None,
                                   save_individual_configs: bool = True,
                                   save_grids: bool = True,
                                   continue_on_failure: bool = True,
                                   max_log_size: int = 10 * 1024 * 1024,  # 10 MB
                                   log_backup_count: int = 5) -> SeriesResult:
    """Load and run experiment series from config file."""
    series_config = ExperimentSeriesConfig.load(input_directory, experiment_name)
    return run_experiment_series(
        series_config=series_config,
        output_directory=output_directory,
        save_individual_configs=save_individual_configs,
        save_grids=save_grids,
        continue_on_failure=continue_on_failure,
        max_log_size=max_log_size,
        log_backup_count=log_backup_count
    ) 