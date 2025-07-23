"""Experiment series runner for executing multiple experiments.

This module provides functionality to run multiple experiments from an experiment series
configuration, with progress tracking and result management.
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

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
                 continue_on_failure: bool = True):
        """Initialize the series runner.
        
        Args:
            series_config: The experiment series configuration
            output_directory: Directory to save results and configs
            save_individual_configs: Whether to save individual experiment configs
            continue_on_failure: Whether to continue if individual experiments fail
        """
        self.series_config = series_config
        self.output_directory = Path(output_directory) if output_directory else Path("experiment_series_output")
        self.save_individual_configs = save_individual_configs
        self.continue_on_failure = continue_on_failure
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized series runner for: {series_config.name}")
        logger.info(f"Output directory: {self.output_directory}")
    
    def run(self) -> SeriesResult:
        """Run the complete experiment series."""
        start_time = time.time()
        logger.info(f"Starting experiment series: {self.series_config.name}")
        
        # Expand all experiments
        experiments = self.series_config.create_experiment_series()
        logger.info(f"Expanded to {len(experiments)} experiments")
        
        # Save series configuration
        series_config_path = self.output_directory / f"{self.series_config.name}_series_config.yaml"
        self.series_config.save(series_config_path)
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
        configs_dir = self.output_directory / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving {len(experiments)} individual configs to {configs_dir}")
        
        for experiment in experiments:
            config_path = configs_dir / f"{experiment.name}.yaml"
            experiment.save(config_path)
        
        logger.info("Individual configs saved")
    
    def _run_experiments(self, experiments: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run all experiments sequentially (field line integration uses parallel processing internally)."""
        return self._run_experiments_sequential(experiments)
    
    def _run_experiments_sequential(self, experiments: List[ExperimentConfig]) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        results = []
        
        for i, experiment in enumerate(experiments, 1):
            logger.info(f"Running experiment {i}/{len(experiments)}: {experiment.name}")
            
            try:
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
        summary_path = self.output_directory / f"{self.series_config.name}_results_summary.yaml"
        
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
        
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        logger.info(f"Results summary saved to: {summary_path}")


def run_experiment_series(series_config: ExperimentSeriesConfig,
                         output_directory: Optional[Path] = None,
                         save_individual_configs: bool = True,
                         continue_on_failure: bool = True) -> SeriesResult:
    """Convenience function to run an experiment series."""
    runner = ExperimentSeriesRunner(
        series_config=series_config,
        output_directory=output_directory,
        save_individual_configs=save_individual_configs,
        continue_on_failure=continue_on_failure
    )
    return runner.run()


def run_experiment_series_from_file(config_path: str,
                                   output_directory: Optional[Path] = None,
                                   save_individual_configs: bool = True,
                                   continue_on_failure: bool = True) -> SeriesResult:
    """Load and run experiment series from config file."""
    series_config = ExperimentSeriesConfig.load(config_path)
    return run_experiment_series(
        series_config=series_config,
        output_directory=output_directory,
        save_individual_configs=save_individual_configs,
        continue_on_failure=continue_on_failure
    ) 