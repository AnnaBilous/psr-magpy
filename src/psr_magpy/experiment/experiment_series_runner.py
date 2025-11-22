"""Experiment series runner module."""

import logging
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, cast, Set
from collections import OrderedDict
import shutil
import yaml
import glob

from .experiment_config import ExperimentConfig
from .experiment_series import ExperimentSeriesConfig
from .experiment_runner import run_experiment
from .experiment_result import ExperimentResult, SeriesResult

from psr_magpy.utils.logging_config import get_logger, setup_logging
from psr_magpy.io import save_config, load_config
from psr_magpy.io.config_io import ExperimentSeriesConfigIO
from psr_magpy.io.validator import ValidationLevel, RecoveryStrategy, ConfigIOValidator
from psr_magpy.io.memory_monitor import MemoryMonitor, MemoryThreshold

logger = get_logger(__name__, console_format_brief=True)


class DeletionStrategy(Enum):
    """Strategy for deleting existing data before running experiments."""
    NONE = "none"
    ALL = "all"
    EXCEPT_CONFIGS = "except_configs"
    EXCEPT_GRIDS = "except_grids"
    EXCEPT_CONFIGS_AND_GRIDS = "except_configs_and_grids"
    LINES_AND_SUMMARY = "lines_and_summary"


class OverwriteStrategy(Enum):
    """Strategy for overwriting existing files during experiment execution."""
    NEVER = "never"
    ALL = "all"
    PRESERVE_CONFIGS = "preserve_configs"
    PRESERVE_CONFIGS_AND_GRIDS = "preserve_configs_and_grids"


class ExperimentSeriesRunner:
    """Runs a series of experiments with progress tracking."""

    def __init__(
        self,
        series_config: ExperimentSeriesConfig,
        save_grids: bool = True,
        save_individual_configs: bool = False,
        continue_on_failure: bool = True,
        experiment_numbers: Optional[List[int]] = None,
        series_deletion_strategy: DeletionStrategy = DeletionStrategy.NONE,
        experiment_deletion_strategy: DeletionStrategy = DeletionStrategy.NONE,
        overwrite_strategy: OverwriteStrategy = OverwriteStrategy.NEVER,
        run_remaining: bool = False,
        create_directories: bool = True,
        log_rotation_max_bytes: float = 10 * 1024 * 1024,  # 10 MB
        log_rotation_backup_count: int = 5,
    ) -> None:
        """Initialize runner with configuration.
        
        Parameters
        ----------
        series_deletion_strategy : DeletionStrategy
            Strategy for deleting data at series level before running.
        experiment_deletion_strategy : DeletionStrategy
            Strategy for deleting data at experiment level before running.
        overwrite_strategy : OverwriteStrategy
            Strategy for overwriting existing files during execution.
        """
        self.series_config = series_config
        self.output_directory = (
            Path(series_config.output_dir) if series_config.output_dir else Path(".")
        )
        self.save_grids = save_grids
        self.save_individual_configs = save_individual_configs
        self.continue_on_failure = continue_on_failure
        self.experiment_numbers = experiment_numbers
        self.run_remaining = run_remaining
        self.create_directories = create_directories
        self.log_rotation_max_bytes = log_rotation_max_bytes
        self.log_rotation_backup_count = log_rotation_backup_count
        self.series_deletion_strategy = series_deletion_strategy
        self.experiment_deletion_strategy = experiment_deletion_strategy
        self.overwrite_strategy = overwrite_strategy

        logger.info("Initialized series runner for: %s", series_config.name)
        logger.info("Output directory: %s", self.output_directory)
        logger.info("Save grids: %s", save_grids)
        logger.info("Series deletion strategy: %s", series_deletion_strategy.value)
        logger.info("Experiment deletion strategy: %s", self.experiment_deletion_strategy.value)
        logger.info("Overwrite strategy: %s", overwrite_strategy.value)
        logger.info(
            "Log rotation: %.1f MB max, %d backups",
            log_rotation_max_bytes / 1024 / 1024,
            log_rotation_backup_count,
        )

    def _delete_directory_contents(
        self, directory: Path, preserve_directories: Set[str]
    ) -> None:
        """Delete contents of directory except preserved directories.
        
        Parameters
        ----------
        directory : Path
            Directory to clean.
        preserve_directories : Set[str]
            Set of directory names to preserve.
        """
        if not directory.exists():
            return
        
        for item in directory.iterdir():
            if item.name not in preserve_directories:
                if item.is_dir():
                    shutil.rmtree(item)
                    logger.info("Deleted directory: %s", item)
                else:
                    item.unlink()
                    logger.info("Deleted file: %s", item)

    def _delete_experiment_files(
        self, experiment_name: str, strategy: DeletionStrategy
    ) -> None:
        """Delete files for a specific experiment.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        strategy : DeletionStrategy
            Deletion strategy to apply.
        """
        if strategy == DeletionStrategy.NONE:
            return
        
        series_dir = self.output_directory / self.series_config.name
        if not series_dir.exists():
            return
        
        if strategy == DeletionStrategy.ALL:
            # Delete all files for this experiment
            for subdir in [self.series_config.LINES_DIR, self.series_config.GRIDS_DIR, self.series_config.CONFIGS_DIR]:
                file_path = series_dir / subdir / f"{experiment_name}.fits"
                if subdir == self.series_config.CONFIGS_DIR:
                    file_path = series_dir / subdir / f"{experiment_name}.yaml"
                if file_path.exists():
                    file_path.unlink()
                    logger.info("Deleted %s file for experiment %s", subdir, experiment_name)
        
        elif strategy == DeletionStrategy.EXCEPT_CONFIGS:
            # Delete lines and grids, keep configs
            for subdir in [self.series_config.LINES_DIR, self.series_config.GRIDS_DIR]:
                file_path = series_dir / subdir / f"{experiment_name}.fits"
                if file_path.exists():
                    file_path.unlink()
                    logger.info("Deleted %s file for experiment %s", subdir, experiment_name)
        
        elif strategy == DeletionStrategy.EXCEPT_GRIDS:
            # Delete lines and configs, keep grids
            for subdir in [self.series_config.LINES_DIR]:
                file_path = series_dir / subdir / f"{experiment_name}.fits"
                if file_path.exists():
                    file_path.unlink()
                    logger.info("Deleted %s file for experiment %s", subdir, experiment_name)
            config_path = series_dir / self.series_config.CONFIGS_DIR / f"{experiment_name}.yaml"
            if config_path.exists():
                config_path.unlink()
                logger.info("Deleted config file for experiment %s", experiment_name)
        
        elif strategy == DeletionStrategy.EXCEPT_CONFIGS_AND_GRIDS:
            # Delete only lines
            file_path = series_dir / self.series_config.LINES_DIR / f"{experiment_name}.fits"
            if file_path.exists():
                file_path.unlink()
                logger.info("Deleted lines file for experiment %s", experiment_name)
        
        elif strategy == DeletionStrategy.LINES_AND_SUMMARY:
            # Delete only lines file
            file_path = series_dir / self.series_config.LINES_DIR / f"{experiment_name}.fits"
            if file_path.exists():
                file_path.unlink()
                logger.info("Deleted lines file for experiment %s", experiment_name)

    def _apply_deletion_strategy(
        self, strategy: DeletionStrategy, target_dir: Path, scope: str = "series"
    ) -> None:
        """Apply deletion strategy to target directory.
        
        Parameters
        ----------
        strategy : DeletionStrategy
            Deletion strategy to apply.
        target_dir : Path
            Target directory.
        scope : str
            Scope of deletion ("series" or "experiment").
        """
        if strategy == DeletionStrategy.NONE:
            return
        
        if not target_dir.exists():
            return
        
        if strategy == DeletionStrategy.ALL:
            # Delete everything except logs
            preserve = {self.series_config.LOGS_DIR}
            self._delete_directory_contents(target_dir, preserve)
            logger.info("Deleted all data (except logs) at %s level", scope)
        
        elif strategy == DeletionStrategy.EXCEPT_CONFIGS:
            preserve = {self.series_config.CONFIGS_DIR, self.series_config.LOGS_DIR}
            self._delete_directory_contents(target_dir, preserve)
            logger.info("Deleted all data (except configs and logs) at %s level", scope)
        
        elif strategy == DeletionStrategy.EXCEPT_GRIDS:
            preserve = {self.series_config.GRIDS_DIR, self.series_config.LOGS_DIR}
            self._delete_directory_contents(target_dir, preserve)
            logger.info("Deleted all data (except grids and logs) at %s level", scope)
        
        elif strategy == DeletionStrategy.EXCEPT_CONFIGS_AND_GRIDS:
            preserve = {
                self.series_config.CONFIGS_DIR,
                self.series_config.GRIDS_DIR,
                self.series_config.LOGS_DIR,
            }
            self._delete_directory_contents(target_dir, preserve)
            logger.info(
                "Deleted all data (except configs, grids, and logs) at %s level", scope
            )
        
        elif strategy == DeletionStrategy.LINES_AND_SUMMARY:
            # Delete only lines directory and summary files
            lines_dir = target_dir / self.series_config.LINES_DIR
            if lines_dir.exists():
                shutil.rmtree(lines_dir)
                logger.info("Deleted lines directory at %s level", scope)
            
            summary_files = list(target_dir.glob("*_results_summary.yaml"))
            for summary_file in summary_files:
                summary_file.unlink()
                logger.info("Deleted summary file: %s", summary_file)

    def _should_overwrite_file(
        self, file_path: Path, file_type: str = "lines"
    ) -> bool:
        """Determine if a file should be overwritten based on strategy.
        
        Parameters
        ----------
        file_path : Path
            Path to the file.
        file_type : str
            Type of file ("lines", "grids", "configs").
        
        Returns
        -------
        bool
            True if file should be overwritten.
        """
        if self.overwrite_strategy == OverwriteStrategy.NEVER:
            return False
        
        if self.overwrite_strategy == OverwriteStrategy.ALL:
            return True
        
        if self.overwrite_strategy == OverwriteStrategy.PRESERVE_CONFIGS:
            return file_type != "configs"
        
        if self.overwrite_strategy == OverwriteStrategy.PRESERVE_CONFIGS_AND_GRIDS:
            return file_type not in ("configs", "grids")
        
        return False

    def _run_experiments(
        self, experiments: List[ExperimentConfig]
    ) -> List[ExperimentResult]:
        """Run a list of experiments."""
        results: List[ExperimentResult] = []
        lines_dir = (
            self.output_directory
            / self.series_config.name
            / self.series_config.LINES_DIR
        )

        for i, experiment in enumerate(experiments, 1):
            logger.info(
                "Running experiment %d/%d: %s", i, len(experiments), experiment.name
            )

            # Apply experiment-level deletion if specified
            if self.experiment_deletion_strategy != DeletionStrategy.NONE:
                self._delete_experiment_files(
                    experiment.name, self.experiment_deletion_strategy
                )

            # Check if already completed
            result_file = lines_dir / f"{experiment.name}.fits"
            if result_file.exists() and not self._should_overwrite_file(
                result_file, file_type="lines"
            ):
                logger.info(
                    "Experiment %s already completed. Skipping (use overwrite_strategy to rerun).",
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
            experiment.tracing.save_results_to_disk = True
            experiment.tracing.filename = str(lines_dir / f"{experiment.name}.fits")
            experiment.tracing.overwrite = True

            # Create lines directory if needed
            if not lines_dir.exists():
                lines_dir.mkdir(parents=True, exist_ok=True)

            # Save grid if requested
            if self.save_grids:
                grids_dir = (
                    self.output_directory
                    / self.series_config.name
                    / self.series_config.GRIDS_DIR
                )
                grid = experiment.grid.create_grid()
                if grid is not None:
                    grid_path = grids_dir / f"{experiment.name}.fits"
                    if not grid_path.exists() or self._should_overwrite_file(
                        grid_path, file_type="grids"
                    ):
                        if not grids_dir.exists():
                            grids_dir.mkdir(parents=True, exist_ok=True)
                        grid.save_to_fits(str(grid_path))
                        logger.info("Saved grid to: %s", grid_path)
                    else:
                        logger.info("Grid file already exists, skipping: %s", grid_path)

            # Run the experiment
            exp_start_time = time.time()
            try:
                result = cast(
                    ExperimentResult,
                    run_experiment(experiment, save_grids=self.save_grids),
                )
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

            # Add result to the list
            results.append(result)

        return results

    def run(self) -> SeriesResult:
        """Run the experiment series."""
        # Create output directories
        series_dir = self.output_directory / self.series_config.name

        # Apply series-level deletion strategy
        if self.series_deletion_strategy != DeletionStrategy.NONE:
            self._apply_deletion_strategy(
                self.series_deletion_strategy, series_dir, scope="series"
            )

        # Create directories if requested and they don't exist
        if self.create_directories:
            if not series_dir.exists():
                series_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            for subdir in self.series_config.get_subdirectories():
                subdir_path = series_dir / subdir
                subdir_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info("Skipping directory creation (create_directories=False)")

        # Set up logging
        log_file = (
            series_dir / self.series_config.LOGS_DIR / f"{self.series_config.name}.log"
        )
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
            # Get total experiments first to know the series size
            all_experiments = self.series_config.load_experiments()
            total_experiments_in_series = len(all_experiments)

            experiments = self._get_remaining_experiments()
            if not experiments:
                logger.info("No remaining experiments to run")
                return SeriesResult(
                    series_name=self.series_config.name,
                    total_experiments=total_experiments_in_series,
                    experiments_in_this_session=0,
                    successful_experiments=0,
                    failed_experiments=0,
                    total_duration=0.0,
                    experiment_results=[],
                    output_directory=series_dir,
                )
        else:
            try:
                experiments = self.series_config.load_experiments()
                total_experiments_in_series = len(experiments)
            except ValueError as e:
                # If no experiments found and we want to save configs, create them
                if self.save_individual_configs:
                    logger.info(
                        "No existing experiments found, creating new experiment series"
                    )
                    experiments = self.series_config.create_experiment_series(
                        save_configs=True, create_directories=self.create_directories
                    )
                    total_experiments_in_series = len(experiments)
                else:
                    # Re-raise the error if we don't want to save configs
                    raise e

        # Filter experiments by number if specified
        if self.experiment_numbers is not None:
            filtered_experiments = []
            for i in self.experiment_numbers:
                if i < 0 or i >= len(experiments):
                    logger.warning(f"Experiment number {i} out of range, skipping")
                    continue
                filtered_experiments.append(experiments[i])
            experiments = filtered_experiments
            logger.info(
                f"Running {len(experiments)} selected experiments out of {total_experiments_in_series} total"
            )

        # Save individual configs if requested
        if self.save_individual_configs:
            self._save_individual_configs(experiments)

        # Run experiments
        start_time = time.time()
        results = self._run_experiments(experiments)
        total_duration = time.time() - start_time

        return SeriesResult(
            series_name=self.series_config.name,
            total_experiments=total_experiments_in_series,
            experiments_in_this_session=len(results),
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
        for i, exp in enumerate(
            self.series_config.create_experiment_series(create_directories=False)
        ):
            if exp.name not in completed_experiments or self.overwrite_strategy != OverwriteStrategy.NEVER:
                remaining_experiments.append(exp)

        return remaining_experiments

    def _format_duration(self, seconds: float) -> str:
        """Format duration in appropriate units.

        Parameters
        ----------
        seconds : float
            Duration in seconds.

        Returns
        -------
        str
            Formatted string like "10.5s", "2.5min", "1.2h", or "3.5d".
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
            batch_pattern = str(
                self.output_directory
                / self.series_config.name
                / self.series_config.LINES_DIR
                / f"{result.experiment_name}*batch_*_out_of_*.fits"
            )
            batch_files = glob.glob(batch_pattern)
            logger.debug(f"Batch file pattern: {batch_pattern}")
            logger.debug(f"Found batch files: {batch_files}")

            # Create result dictionary with all fields
            result_dict = {
                "name": result.experiment_name,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": self._format_duration(result.duration),
                "success": result.success,
                "batch_files": len(batch_files),
            }

            # Add error message if present
            if result.error_message:
                result_dict["error"] = result.error_message

            # Convert to OrderedDict
            ordered_items = []
            ordered_items.append(("name", result_dict["name"]))
            ordered_items.append(("start_time", result_dict["start_time"]))
            ordered_items.append(("duration", result_dict["duration"]))
            ordered_items.append(("success", result_dict["success"]))
            if "error" in result_dict:
                ordered_items.append(("error", result_dict["error"]))
            ordered_items.append(("batch_files", result_dict["batch_files"]))
            result_dict = OrderedDict(ordered_items)

            # Update or add result
            if result.experiment_name in existing_results:
                existing_results[result.experiment_name].update(result_dict)
            else:
                existing_results[result.experiment_name] = result_dict

        # Sort results by name
        sorted_results = sorted(existing_results.values(), key=lambda x: x["name"])

        # Create new summary with updated statistics
        summary = {
            "series_name": self.series_config.name,
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_experiments": len(sorted_results),
            "successful_experiments": sum(1 for r in sorted_results if r["success"]),
            "failed_experiments": sum(1 for r in sorted_results if not r["success"]),
            "total_duration": self._format_duration(
                sum(
                    (
                        float(r["duration"].rstrip("s"))
                        if r["duration"].endswith("s")
                        else (
                            float(r["duration"].rstrip("min")) * 60
                            if r["duration"].endswith("min")
                            else (
                                float(r["duration"].rstrip("h")) * 3600
                                if r["duration"].endswith("h")
                                else float(r["duration"].rstrip("d")) * 86400
                            )
                        )
                    )
                    for r in sorted_results
                )
            ),
            "success_rate": (
                sum(1 for r in sorted_results if r["success"])
                / len(sorted_results)
                * 100
                if sorted_results
                else 0.0
            ),
            "experiment_results": sorted_results,
        }

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
        configs_dir = (
            self.output_directory
            / self.series_config.name
            / self.series_config.CONFIGS_DIR
        )

        logger.info(f"Saving {len(experiments)} individual configs to {configs_dir}")

        for experiment in experiments:
            config_path = configs_dir / f"{experiment.name}.yaml"
            if not config_path.exists() or self._should_overwrite_file(
                config_path, file_type="configs"
            ):
                if not configs_dir.exists():
                    configs_dir.mkdir(parents=True, exist_ok=True)
                save_config(experiment, config_path, overwrite=True)
                logger.info(f"Saved config to: {config_path}")
            else:
                logger.info(f"Config file already exists, skipping: {config_path}")

        logger.info("Individual configs saved")


class InMemoryExperimentSeriesRunner:
    """Lightweight runner for experiment series that keeps results in memory."""

    def __init__(
        self,
        series_config: ExperimentSeriesConfig,
        keep_results: bool = True,
        max_memory_mb: float = 1000.0,
        continue_on_failure: bool = True,
        experiment_numbers: Optional[List[int]] = None,
        save_individual_configs: bool = False,
        save_grids: bool = False,
    ) -> None:
        """Initialize in-memory runner with configuration."""
        self.series_config = series_config
        self.output_directory = (
            Path(series_config.output_dir) if series_config.output_dir else Path(".")
        )
        self.keep_results = keep_results
        self.max_memory_mb = max_memory_mb
        self.continue_on_failure = continue_on_failure
        self.experiment_numbers = experiment_numbers
        self.save_individual_configs = save_individual_configs
        self.save_grids = save_grids

        logger.info("Initialized in-memory series runner for: %s", series_config.name)
        logger.info("Keep results in memory: %s", keep_results)
        logger.info("Max memory limit: %.1f MB", max_memory_mb)
        logger.info("Save grids: %s", save_grids)

    def _run_experiments(
        self, experiments: List[ExperimentConfig]
    ) -> List[ExperimentResult]:
        """Run a list of experiments with results kept in memory."""
        results: List[ExperimentResult] = []
        total_memory_used = 0.0

        for i, experiment in enumerate(experiments, 1):
            logger.info(
                "Running experiment %d/%d: %s", i, len(experiments), experiment.name
            )

            # Configure experiment for in-memory results
            experiment.tracing.save_results_to_disk = False
            experiment.tracing.keep_results_in_memory = True
            experiment.tracing.filename = None

            # Run the experiment
            exp_start_time = time.time()
            try:
                result = cast(
                    ExperimentResult,
                    run_experiment(experiment, save_grids=self.save_grids),
                )
                duration = time.time() - exp_start_time

                if result.success and result.results is not None:
                    # Estimate memory usage
                    result_memory_mb = result.results.estimate_size() / (1024 * 1024)
                    total_memory_used += result_memory_mb

                    logger.info(
                        "✓ Experiment %s completed in %.2fs (%.1f MB)",
                        experiment.name,
                        duration,
                        result_memory_mb,
                    )

                    # Check memory limit
                    if total_memory_used > self.max_memory_mb:
                        warning_msg = f"Memory usage ({total_memory_used:.1f} MB) exceeds limit ({self.max_memory_mb:.1f} MB)"
                        logger.warning(warning_msg)
                        if not self.keep_results:
                            logger.info("Clearing results due to memory limit")
                            result.results = None
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

            # Add result to the list (keep results in memory if requested)
            if not self.keep_results and result.results is not None:
                result.results = None

            results.append(result)

        logger.info("Total memory used: %.1f MB", total_memory_used)
        return results

    def run(self) -> SeriesResult:
        """Run the experiment series with results in memory."""
        # Create experiments
        experiments = self.series_config.create_experiment_series(
            save_configs=self.save_individual_configs,
            create_directories=False,  # In-memory runner doesn't need directories
        )
        total_experiments_in_series = len(experiments)

        # Filter experiments by number if specified
        if self.experiment_numbers is not None:
            filtered_experiments = []
            for i in self.experiment_numbers:
                if i < 0 or i >= len(experiments):
                    logger.warning(f"Experiment number {i} out of range, skipping")
                    continue
                filtered_experiments.append(experiments[i])
            experiments = filtered_experiments
            logger.info(
                f"Running {len(experiments)} selected experiments out of {total_experiments_in_series} total"
            )

        # Run experiments
        start_time = time.time()
        results = self._run_experiments(experiments)
        total_duration = time.time() - start_time

        return SeriesResult(
            series_name=self.series_config.name,
            total_experiments=total_experiments_in_series,
            experiments_in_this_session=len(results),
            successful_experiments=sum(1 for r in results if r.success),
            failed_experiments=sum(1 for r in results if not r.success),
            total_duration=total_duration,
            experiment_results=results,
            output_directory=self.output_directory,
        )


def run_experiment_series(
    series_config: ExperimentSeriesConfig,
    experiment_numbers: Optional[List[int]] = None,
    run_remaining: bool = False,
    series_deletion_strategy: DeletionStrategy = DeletionStrategy.NONE,
    experiment_deletion_strategy: DeletionStrategy = DeletionStrategy.NONE,
    overwrite_strategy: OverwriteStrategy = OverwriteStrategy.NEVER,
    save_individual_configs: bool = True,
    save_grids: bool = True,
    continue_on_failure: bool = True,
    create_directories: bool = True,
) -> SeriesResult:
    """Run an experiment series.
    
    Parameters
    ----------
    series_config : ExperimentSeriesConfig
        Configuration for the experiment series.
    series_deletion_strategy : DeletionStrategy
        Strategy for deleting data at series level before running.
    experiment_deletion_strategy : DeletionStrategy
        Strategy for deleting data at experiment level before running.
    overwrite_strategy : OverwriteStrategy
        Strategy for overwriting existing files during execution.
    continue_on_failure : bool, optional
        Whether to continue if an experiment fails. Default is True.
    create_directories : bool, optional
        Whether to create output directories. Default is True.
    
    Returns
    -------
    SeriesResult
        Results from running the experiment series.
    """
    # Check if any experiments have save_results_to_disk=False and warn user
    experiments = series_config.create_experiment_series(
        save_configs=False, create_directories=False
    )
    for experiment in experiments:
        if (
            hasattr(experiment.tracing, "save_results_to_disk")
            and not experiment.tracing.save_results_to_disk
        ):
            logger.warning(
                "WARNING: run_experiment_series() will override save_results_to_disk=False "
                "and force results to be saved to disk. Use run_experiment_series_in_memory() "
                "if you want to keep results in memory only."
            )
            break

    runner = ExperimentSeriesRunner(
        series_config=series_config,
        save_individual_configs=save_individual_configs,
        save_grids=save_grids,
        continue_on_failure=continue_on_failure,
        experiment_numbers=experiment_numbers,
        series_deletion_strategy=series_deletion_strategy,
        experiment_deletion_strategy=experiment_deletion_strategy,
        overwrite_strategy=overwrite_strategy,
        run_remaining=run_remaining,
        create_directories=create_directories,
    )

    return runner.run()


def run_experiment_series_from_file(
    config_file: Union[str, Path],
    input_directory: Optional[Union[str, Path]] = None,
    experiment_numbers: Optional[List[int]] = None,
    run_remaining: bool = False,
    series_deletion_strategy: DeletionStrategy = DeletionStrategy.NONE,
    experiment_deletion_strategy: DeletionStrategy = DeletionStrategy.NONE,
    overwrite_strategy: OverwriteStrategy = OverwriteStrategy.NEVER,
    save_individual_configs: bool = True,
    save_grids: bool = True,
    continue_on_failure: bool = True,
) -> SeriesResult:
    """Run experiment series from a configuration file."""
    # Load series config
    config_file = Path(config_file)
    series_config = load_config(ExperimentSeriesConfig, config_file)

    # Set input directory if provided
    if input_directory is not None:
        series_config.output_dir = input_directory

    # Run series
    return run_experiment_series(
        series_config=series_config,
        experiment_numbers=experiment_numbers,
        run_remaining=run_remaining,
        series_deletion_strategy=series_deletion_strategy,
        experiment_deletion_strategy=experiment_deletion_strategy,
        overwrite_strategy=overwrite_strategy,
        save_individual_configs=save_individual_configs,
        save_grids=save_grids,
        continue_on_failure=continue_on_failure,
    )


def run_experiment_series_in_memory(
    series_config: ExperimentSeriesConfig,
    keep_results: bool = True,
    max_memory_mb: float = 1000.0,
    experiment_numbers: Optional[List[int]] = None,
    save_individual_configs: bool = False,
    save_grids: bool = False,
    continue_on_failure: bool = True,
) -> SeriesResult:
    """Run an experiment series with results kept in memory.
    
    Parameters
    ----------
    series_config : ExperimentSeriesConfig
        Configuration for the experiment series.
    keep_results : bool, optional
        Whether to keep results in memory. Default is True.
    max_memory_mb : Optional[int], optional
        Maximum memory in MB. Default is None.
    save_grids : bool, optional
        Whether to save grids. Default is False.
    continue_on_failure : bool, optional
        Whether to continue if an experiment fails. Default is True.
    
    Returns
    -------
    SeriesResult
        Results from running the experiment series.
    """
    runner = InMemoryExperimentSeriesRunner(
        series_config=series_config,
        keep_results=keep_results,
        max_memory_mb=max_memory_mb,
        continue_on_failure=continue_on_failure,
        experiment_numbers=experiment_numbers,
        save_individual_configs=save_individual_configs,
        save_grids=save_grids,
    )

    return runner.run()


def run_experiment_series_in_memory_from_file(
    config_file: Union[str, Path],
    input_directory: Optional[Union[str, Path]] = None,
    keep_results: bool = True,
    max_memory_mb: float = 1000.0,
    experiment_numbers: Optional[List[int]] = None,
    save_individual_configs: bool = False,
    save_grids: bool = False,
    continue_on_failure: bool = True,
) -> SeriesResult:
    """Run experiment series from a configuration file with results kept in memory."""
    # Load series config
    config_file = Path(config_file)
    series_config = load_config(ExperimentSeriesConfig, config_file)

    # Set input directory if provided
    if input_directory is not None:
        series_config.output_dir = input_directory

    # Run series in memory
    return run_experiment_series_in_memory(
        series_config=series_config,
        keep_results=keep_results,
        max_memory_mb=max_memory_mb,
        experiment_numbers=experiment_numbers,
        save_individual_configs=save_individual_configs,
        save_grids=save_grids,
        continue_on_failure=continue_on_failure,
    )
