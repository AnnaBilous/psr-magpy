"""Subset workflow orchestration for coarse-to-fine experiment series."""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from psr_magpy.utils.exceptions import PSRMagPyExperimentError
from psr_magpy.utils.logging_config import get_logger

if TYPE_CHECKING:
    from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig

logger = get_logger(__name__, console_format_brief=True)


class SubsetWorkflowService:
    """Create fine subset grids from coarse OFL results."""

    def create_subset_series_from_ofl(
        self,
        *,
        fine_series_config: "ExperimentSeriesConfig",
        coarse_series_config: "ExperimentSeriesConfig",
        R_LC: float,
        h_LC_coeff: float = 3**1.5 / 2.0,
        experiment_numbers: Optional[List[int]] = None,
        save_configs: bool = True,
    ) -> "ExperimentSeriesConfig":
        """Run coarse-to-fine OFL subset workflow."""
        from psr_magpy.grid.grid import create_fine_resolution_subset_from_coarse_OFL

        coarse_status = coarse_series_config.verify_directories(create_if_missing=False)
        if not coarse_status["configs"]["exists"] or not coarse_status["lines"]["exists"]:
            raise PSRMagPyExperimentError(
                "Coarse series directories (configs, lines) must exist"
            )

        fine_series_config.verify_directories(create_if_missing=True)
        fine_experiments = fine_series_config.create_experiment_series(save_configs=False)

        if experiment_numbers is None:
            coarse_experiments = coarse_series_config.create_experiment_series(
                save_configs=False
            )
            experiment_numbers = list(range(len(coarse_experiments)))

        logger.info(
            f"Processing {len(experiment_numbers)} experiments for OFL subset creation"
        )

        successful_count = 0
        failed_count = 0

        for exp_num in experiment_numbers:
            logger.info(f"Processing experiment {exp_num}...")
            try:
                coarse_experiment = coarse_series_config.load_experiment_config(exp_num)
                logger.debug(f"Loaded coarse experiment: {coarse_experiment.name}")

                fine_experiment = fine_series_config.get_experiment_by_index(
                    exp_num, fine_experiments
                )
                if fine_experiment is None:
                    logger.warning(
                        f"No corresponding fine experiment found for {exp_num}, skipping..."
                    )
                    failed_count += 1
                    continue

                logger.debug(f"Using fine experiment: {fine_experiment.name}")
                coarse_lines_dir = coarse_series_config.get_lines_dir()
                fine_grids_dir = fine_series_config.get_grids_dir()

                grid_filepath, updated_grid_config = (
                    create_fine_resolution_subset_from_coarse_OFL(
                        coarse_experiment=coarse_experiment,
                        fine_experiment=fine_experiment,
                        R_LC=R_LC,
                        coarse_lines_dir=coarse_lines_dir,
                        fine_grids_dir=fine_grids_dir,
                        h_LC_coeff=h_LC_coeff,
                    )
                )

                fine_experiment.grid = updated_grid_config
                subset_metadata = {
                    "coarse_experiment": coarse_experiment.name,
                    "subset_grid_type": "healpix",
                    "is_subset_grid": True,
                    "subset_grid_file": grid_filepath.name,
                    "R_LC": R_LC,
                    "h_LC_coeff": h_LC_coeff,
                }
                if fine_experiment.metadata is None:
                    fine_experiment.metadata = {}
                fine_experiment.metadata.update(subset_metadata)

                if save_configs:
                    fine_config_file = (
                        fine_series_config.get_configs_dir()
                        / f"{fine_experiment.name}_subset.yaml"
                    )
                    fine_experiment.save(str(fine_config_file), overwrite=True)
                    logger.info(f"Saved subset experiment config: {fine_config_file}")

                successful_count += 1
            except Exception as exc:
                error_msg = f"Failed to process experiment {exp_num}: {exc}"
                logger.error(error_msg)
                failed_count += 1
                continue

        logger.info(
            f"OFL subset creation completed: {successful_count} successful, "
            f"{failed_count} failed"
        )

        if fine_series_config.metadata is None:
            fine_series_config.metadata = {}
        fine_series_config.metadata.update(
            {
                "original_series": fine_series_config.name,
                "subset_experiments": experiment_numbers,
                "subset_grid_type": "healpix",
                "has_subset_grids": True,
                "coarse_series": coarse_series_config.name,
                "R_LC": R_LC,
                "h_LC_coeff": h_LC_coeff,
            }
        )
        return fine_series_config
