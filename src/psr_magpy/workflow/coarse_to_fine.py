"""Workflow for creating fine resolution subset grids from coarse experiments."""

from typing import Optional, List
from pathlib import Path

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_series_runner import run_experiment_series
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyExperimentError

logger = get_logger(__name__, console_format_brief=True)


def create_fine_subset_workflow(
    coarse_series_config: ExperimentSeriesConfig,
    fine_series_config: ExperimentSeriesConfig,
    R_LC: float,
    h_LC_coeff: float = 3**1.5 / 2.0,
    experiment_numbers: Optional[List[int]] = None,
    run_fine_series: bool = True,
    **run_kwargs,
) -> ExperimentSeriesConfig:
    """Complete workflow for creating and running fine resolution subset series.

    This workflow:
    1. Verifies coarse series directories exist
    2. Creates fine resolution subset grids from coarse OFL footpoints
    3. Optionally runs the fine resolution experiment series
    4. Returns updated fine series config

    Args:
        coarse_series_config: Coarse resolution series config
        fine_series_config: Fine resolution series config
        R_LC: Light cylinder radius for OFL calculation
        h_LC_coeff: Height coefficient for OFL calculation
        experiment_numbers: Optional list of experiment numbers to process
        run_fine_series: If True, run the fine series after creating subsets
        **run_kwargs: Additional arguments passed to run_experiment_series

    Returns:
        Updated fine series config with subset grids

    Example:
        >>> fine_config = create_fine_subset_workflow(
        ...     coarse_series_config=coarse_config,
        ...     fine_series_config=fine_config,
        ...     R_LC=10.0,
        ...     run_fine_series=True
        ... )
    """
    logger.info("Starting fine subset workflow")
    logger.info(f"Coarse series: {coarse_series_config.name}")
    logger.info(f"Fine series: {fine_series_config.name}")

    # Verify coarse directories
    coarse_status = coarse_series_config.verify_directories(create_if_missing=False)
    if not coarse_status["configs"]["exists"] or not coarse_status["lines"]["exists"]:
        raise PSRMagPyExperimentError("Coarse series directories (configs, lines) must exist")

    # Create subset series from OFL
    logger.info("Creating fine resolution subset grids from coarse OFL footpoints...")
    fine_series_config = fine_series_config.create_subset_series_from_OFL(
        coarse_series_config=coarse_series_config,
        R_LC=R_LC,
        h_LC_coeff=h_LC_coeff,
        experiment_numbers=experiment_numbers,
        save_configs=True,
    )

    # Optionally run the fine series
    if run_fine_series:
        logger.info("Running fine resolution experiment series...")
        series_result = run_experiment_series(
            series_config=fine_series_config,
            save_individual_configs=False,
            save_grids=False,
            continue_on_failure=True,
            delete_existing_lines_and_summary=True,
            **run_kwargs,
        )
        series_result.print_summary()

    logger.info("Fine subset workflow completed")
    return fine_series_config

