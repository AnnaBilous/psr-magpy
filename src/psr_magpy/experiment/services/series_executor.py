"""Service for series-level orchestration and result aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_result import SeriesResult
from psr_magpy.utils.logging_config import get_logger

if TYPE_CHECKING:
    from psr_magpy.experiment.experiment_series_runner import (
        DeletionStrategy,
        OverwriteStrategy,
    )

logger = get_logger(__name__, console_format_brief=True)


class SeriesExecutor:
    """Run experiment series through runner adapters."""

    def run_disk_series(
        self,
        *,
        series_config: ExperimentSeriesConfig,
        experiment_numbers: Optional[List[int]] = None,
        run_remaining: bool = False,
        series_deletion_strategy: "DeletionStrategy",
        experiment_deletion_strategy: "DeletionStrategy",
        overwrite_strategy: "OverwriteStrategy",
        save_individual_configs: bool = True,
        save_grids: bool = True,
        continue_on_failure: bool = True,
        create_directories: bool = True,
    ) -> SeriesResult:
        """Run disk-backed series via ``ExperimentSeriesRunner``."""
        # Keep legacy warning behavior before runner execution.
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

        from psr_magpy.experiment.experiment_series_runner import ExperimentSeriesRunner

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

    def run_in_memory_series(
        self,
        *,
        series_config: ExperimentSeriesConfig,
        experiment_numbers: Optional[List[int]] = None,
        keep_results: bool = True,
        max_memory_mb: float = 1000.0,
        continue_on_failure: bool = True,
        save_individual_configs: bool = False,
        save_grids: bool = False,
    ) -> SeriesResult:
        """Run in-memory series via ``InMemoryExperimentSeriesRunner``."""
        from psr_magpy.experiment.experiment_series_runner import (
            InMemoryExperimentSeriesRunner,
        )

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

    def run_from_file(
        self,
        *,
        config_file: Path,
        input_directory: Optional[Path] = None,
        experiment_numbers: Optional[List[int]] = None,
        run_remaining: bool = False,
        series_deletion_strategy: "DeletionStrategy",
        experiment_deletion_strategy: "DeletionStrategy",
        overwrite_strategy: "OverwriteStrategy",
        save_individual_configs: bool = True,
        save_grids: bool = True,
        continue_on_failure: bool = True,
    ) -> SeriesResult:
        """Load series config from file and run disk-backed series."""
        from psr_magpy.io import load_config

        series_config = load_config(ExperimentSeriesConfig, config_file)
        if input_directory is not None:
            series_config.output_dir = str(input_directory)

        return self.run_disk_series(
            series_config=series_config,
            experiment_numbers=experiment_numbers,
            run_remaining=run_remaining,
            series_deletion_strategy=series_deletion_strategy,
            experiment_deletion_strategy=experiment_deletion_strategy,
            overwrite_strategy=overwrite_strategy,
            save_individual_configs=save_individual_configs,
            save_grids=save_grids,
            continue_on_failure=continue_on_failure,
            create_directories=True,
        )
