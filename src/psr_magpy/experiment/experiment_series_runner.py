"""Experiment series runner module."""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, cast
from collections import OrderedDict
import shutil
import yaml
import glob

from .experiment_config import ExperimentConfig
from .experiment_series import ExperimentSeriesConfig
from .experiment_runner import run_experiment
from .experiment_result import ExperimentResult, SeriesResult

from psr_magpy.utils.logging_config import get_logger, setup_logging
from psr_magpy.io import get_io_factory

logger = get_logger(__name__, console_format_brief=True)


class ExperimentSeriesRunner:
    """Runs a series of experiments with progress tracking."""

    def __init__(
        self,
        series_config: ExperimentSeriesConfig,
        save_grids: bool = True,
        save_individual_configs: bool = False,
        continue_on_failure: bool = True,
        experiment_numbers: Optional[List[int]] = None,
        overwrite_everything: bool = False,
        overwrite_results: bool = False,
        run_remaining: bool = False,
        log_rotation_max_bytes: float = 10 * 1024 * 1024,  # 10 MB
        log_rotation_backup_count: int = 5,
    ) -> None:
        """Initialize runner with configuration."""
        self.series_config = series_config
        self.output_directory = (
            Path(series_config.output_dir) if series_config.output_dir else Path(".")
        )
        self.save_grids = save_grids
        self.save_individual_configs = save_individual_configs
        self.continue_on_failure = continue_on_failure
        self.experiment_numbers = experiment_numbers
        self.overwrite_everything = overwrite_everything
        self.overwrite_results = overwrite_results
        self.run_remaining = run_remaining
        self.log_rotation_max_bytes = log_rotation_max_bytes
        self.log_rotation_backup_count = log_rotation_backup_count

        logger.info("Initialized series runner for: %s", series_config.name)
        logger.info("Output directory: %s", self.output_directory)
        logger.info("Save grids: %s", save_grids)
        logger.info(
            "Log rotation: %.1f MB max, %d backups",
            log_rotation_max_bytes / 1024 / 1024,
            log_rotation_backup_count,
        )

    def _run_experiments(
        self, experiments: List[ExperimentConfig]
    ) -> List[ExperimentResult]:
        """Run a list of experiments."""
        results: List[ExperimentResult] = []
        lines_dir = self.output_directory / self.series_config.name / "lines"
        lines_dir.mkdir(exist_ok=True)

        for i, experiment in enumerate(experiments, 1):
            logger.info(
                "Running experiment %d/%d: %s", i, len(experiments), experiment.name
            )

            # Check if already completed
            result_file = lines_dir / f"{experiment.name}.fits"
            if result_file.exists() and not self.overwrite_results:
                logger.info(
                    "Experiment %s already completed. Skipping (use overwrite_results=True to rerun).",
                    experiment.name,
                )
                result = ExperimentResult(
                    experiment_name=experiment.name,
                    success=True,
                    duration=0.0,
                    results=None,
                )
                results.append(result)
                self._save_results_summary([result])
                continue

            # Configure experiment to save results
            experiment.tracing.save_results = True
            experiment.tracing.filename = str(lines_dir / f"{experiment.name}.fits")
            experiment.tracing.overwrite = True

            # Save grid if requested
            if self.save_grids:
                grids_dir = self.output_directory / self.series_config.name / "grids"
                grids_dir.mkdir(exist_ok=True)
                grid = experiment.grid.create_grid()
                if grid is not None:
                    grid_path = grids_dir / f"{experiment.name}.fits"
                    grid.save_to_fits(str(grid_path))
                    logger.info("Saved grid to: %s", grid_path)

            # Run the experiment
            exp_start_time = time.time()
            try:
                result = cast(ExperimentResult, run_experiment(experiment))
                duration = time.time() - exp_start_time
                if result.success:
                    logger.info(
                        "✓ Experiment %s completed in %.2fs", experiment.name, duration
                    )
                else:
                    error_msg = result.error_message or "Unknown error"
                    logger.error(
                        "✗ Experiment %s failed after %.2fs: %s",
                        experiment.name,
                        duration,
                        error_msg,
                    )
                    if not self.continue_on_failure:
                        raise RuntimeError(error_msg)
            except Exception as e:
                duration = time.time() - exp_start_time
                result = ExperimentResult(
                    experiment_name=experiment.name,
                    success=False,
                    duration=duration,
                    error_message=str(e),
                )
                logger.error(
                    "✗ Experiment %s failed after %.2fs: %s",
                    experiment.name,
                    duration,
                    str(e),
                )
                if not self.continue_on_failure:
                    raise

            # Update summary after each experiment
            self._save_results_summary([result])

            # Clear results from memory after saving
            if result.results is not None:
                result.results = None

            # Keep only the latest result
            results = [result]

        return results

    def run(self) -> SeriesResult:
        """Run the experiment series."""
        # Create output directories
        series_dir = self.output_directory / self.series_config.name

        # Clean up if overwrite_everything is True
        if self.overwrite_everything and series_dir.exists():
            # Delete all files and subdirectories except configs
            for item in series_dir.iterdir():
                if item.name != "configs":
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

        # Create subdirectories
        for subdir in ["lines", "grids", "configs", "logs"]:
            subdir_path = series_dir / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)

        # Set up logging
        log_file = series_dir / "logs" / f"{self.series_config.name}.log"
        setup_logging(
            log_file=log_file,
            log_level=logging.INFO,
            handler_levels={"console": logging.INFO, "file": logging.DEBUG},
            max_file_size=int(self.log_rotation_max_bytes),
            backup_count=self.log_rotation_backup_count,
        )

        logger.info("Series logging set up: %s", log_file)
        logger.info(
            "Log rotation: %.1f MB max, %d backups",
            self.log_rotation_max_bytes / 1024 / 1024,
            self.log_rotation_backup_count,
        )

        # Load or create experiments
        if self.run_remaining:
            experiments = self._get_remaining_experiments()
            if not experiments:
                logger.info("No remaining experiments to run")
                return SeriesResult(
                    series_name=self.series_config.name,
                    total_experiments=0,
                    successful_experiments=0,
                    failed_experiments=0,
                    total_duration=0.0,
                    experiment_results=[],
                    output_directory=series_dir,
                )
        else:
            experiments = self.series_config.load_experiments()

        # Filter experiments by number if specified
        if self.experiment_numbers is not None:
            filtered_experiments = []
            for i in self.experiment_numbers:
                if i < 0 or i >= len(experiments):
                    logger.warning(f"Experiment number {i} out of range, skipping")
                    continue
                filtered_experiments.append(experiments[i])
            experiments = filtered_experiments
            logger.info(f"Running {len(experiments)} selected experiments")

        # Save individual configs if requested
        if self.save_individual_configs:
            self._save_individual_configs(experiments)

        # Run experiments
        start_time = time.time()
        results = self._run_experiments(experiments)
        total_duration = time.time() - start_time

        return SeriesResult(
            series_name=self.series_config.name,
            total_experiments=len(results),
            successful_experiments=sum(1 for r in results if r.success),
            failed_experiments=sum(1 for r in results if not r.success),
            total_duration=total_duration,
            experiment_results=results,
            output_directory=self.output_directory,
        )

    def _save_results(self, results: List[ExperimentResult]) -> None:
        """Save experiment results."""
        logger.info("Saving results for %d experiments", len(results))
        # Save results logic here

    def _get_remaining_experiments(self) -> List[ExperimentConfig]:
        """Load existing summary to find completed experiments."""
        summary_path = (
            self.output_directory
            / self.series_config.name
            / f"{self.series_config.name}_results_summary.yaml"
        )
        completed_experiments = set()
        if summary_path.exists():
            with open(summary_path) as f:
                summary = yaml.safe_load(f)
                if summary and "experiment_results" in summary:
                    completed_experiments = {
                        r["name"] for r in summary["experiment_results"]
                    }

        remaining_experiments = []
        for i, exp in enumerate(self.series_config.create_experiment_series()):
            if exp.name not in completed_experiments or self.overwrite_results:
                remaining_experiments.append(exp)

        return remaining_experiments

    def _format_duration(self, seconds: float) -> str:
        """Format duration in appropriate units.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string like "10.5s", "2.5min", "1.2h", or "3.5d"
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}min"
        hours = minutes / 60
        if hours < 24:
            return f"{hours:.1f}h"
        days = hours / 24
        return f"{days:.1f}d"

    def _save_results_summary(self, experiment_results: List[ExperimentResult]) -> None:
        """Save results summary to YAML file."""
        summary_path = (
            self.output_directory
            / self.series_config.name
            / f"{self.series_config.name}_results_summary.yaml"
        )

        # Load existing summary if it exists
        existing_summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                existing_summary = yaml.safe_load(f) or {}

        # Create new summary
        summary = {
            "series_name": self.series_config.name,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_experiments": len(experiment_results),
            "successful_experiments": sum(1 for r in experiment_results if r.success),
            "failed_experiments": sum(1 for r in experiment_results if not r.success),
            "total_duration": sum(r.duration for r in experiment_results),
            "success_rate": (
                sum(1 for r in experiment_results if r.success)
                / len(experiment_results)
                * 100
                if experiment_results
                else 0.0
            ),
            "experiment_results": [],
        }

        # Merge with existing results
        if "experiment_results" in existing_summary:
            existing_results = {
                r["name"]: r for r in existing_summary["experiment_results"]
            }
        else:
            existing_results = {}

        # Add new results
        for result in experiment_results:
            # Count batch files
            batch_pattern = str(self.output_directory / self.series_config.name / "lines" / f"{result.experiment_name}*batch_*_out_of_*.fits")
            batch_files = glob.glob(batch_pattern)
            logger.debug(f"Batch file pattern: {batch_pattern}")
            logger.debug(f"Found batch files: {batch_files}")

            result_dict = {
                "name": result.experiment_name,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": self._format_duration(result.duration),
                "success": result.success,
                "error": result.error_message if not result.success else None,
                "batch_files": len(batch_files)
            }
            result_dict = OrderedDict(result_dict.items())

            # Update or add result
            if result.experiment_name in existing_results:
                existing_results[result.experiment_name].update(result_dict)
            else:
                existing_results[result.experiment_name] = result_dict

        # Sort results by name and add to summary
        summary["experiment_results"] = sorted(
            existing_results.values(), key=lambda x: x["name"]
        )

        # Create custom dumper to preserve dictionary order
        class OrderedDumper(yaml.SafeDumper):
            pass

        def _dict_representer(
            dumper: yaml.SafeDumper, data: OrderedDict
        ) -> yaml.nodes.MappingNode:
            return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

        OrderedDumper.add_representer(OrderedDict, _dict_representer)

        # Save summary
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, Dumper=OrderedDumper, sort_keys=False, width=1000)

        logger.info(f"Results summary saved to: {summary_path}")

    def _save_individual_configs(self, experiments: List[ExperimentConfig]) -> None:
        """Save individual experiment configurations."""
        # Use the series-specific configs directory
        configs_dir = self.output_directory / self.series_config.name / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(experiments)} individual configs to {configs_dir}")

        for experiment in experiments:
            config_path = configs_dir / f"{experiment.name}.yaml"
            io_instance = get_io_factory().create_config_io(type(experiment))
            io_instance.save(experiment, config_path, overwrite=True)

        logger.info("Individual configs saved")


def run_experiment_series(
    series_config: ExperimentSeriesConfig,
    experiment_numbers: Optional[List[int]] = None,
    run_remaining: bool = False,
    overwrite_results: bool = False,
    overwrite_everything: bool = False,
    save_individual_configs: bool = True,
    save_grids: bool = True,
    continue_on_failure: bool = True,
) -> SeriesResult:
    """Convenience function to run an experiment series."""
    runner = ExperimentSeriesRunner(
        series_config=series_config,
        save_individual_configs=save_individual_configs,
        save_grids=save_grids,
        continue_on_failure=continue_on_failure,
        experiment_numbers=experiment_numbers,
        overwrite_everything=overwrite_everything,
        run_remaining=run_remaining,
        overwrite_results=overwrite_results,
    )

    return runner.run()


def run_experiment_series_from_file(
    config_file: Union[str, Path],
    input_directory: Optional[Union[str, Path]] = None,
    experiment_numbers: Optional[List[int]] = None,
    run_remaining: bool = False,
    overwrite_results: bool = False,
    overwrite_everything: bool = False,
    save_individual_configs: bool = True,
    save_grids: bool = True,
    continue_on_failure: bool = True,
) -> SeriesResult:
    """Run experiment series from a configuration file."""
    # Load series config
    config_file = Path(config_file)
    io_instance = get_io_factory().create_config_io(ExperimentSeriesConfig)
    series_config = io_instance.load(config_file)

    # Set input directory if provided
    if input_directory is not None:
        series_config.output_dir = input_directory

    # Run series
    return run_experiment_series(
        series_config=series_config,
        experiment_numbers=experiment_numbers,
        run_remaining=run_remaining,
        overwrite_results=overwrite_results,
        overwrite_everything=overwrite_everything,
        save_individual_configs=save_individual_configs,
        save_grids=save_grids,
        continue_on_failure=continue_on_failure,
    )
