"""Workflow for creating fine resolution subset grids from coarse experiments."""

from typing import Optional, List
from pathlib import Path

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


def create_fine_subset_workflow(
    coarse_series_config: ExperimentSeriesConfig,
    fine_series_config: ExperimentSeriesConfig,
    R_LC: float,
    r_max: Optional[float] = None,
    h_LC_coeff: float = 3**1.5 / 2.0,
    experiment_numbers: Optional[List[int]] = None,
) -> ExperimentSeriesConfig:
    """Complete workflow for creating fine resolution subset series.

    This workflow:
    1. Verifies coarse series directories exist
    2. Creates fine resolution subset grids from coarse OFL footpoints
    3. Returns updated fine series config

    Parameters
    ----------
    coarse_series_config : ExperimentSeriesConfig
        Coarse resolution series config.
    fine_series_config : ExperimentSeriesConfig
        Fine resolution series config.
    R_LC : float
        Light cylinder radius for OFL calculation.
    r_max : Optional[float], optional
        Optional override for ``fine_series_config.tracing.max_integration_radius``.
        If None, keep the value already present in fine config.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    experiment_numbers : Optional[List[int]], optional
        Optional list of experiment numbers to process. Default is None.

    Returns
    -------
    ExperimentSeriesConfig
        Updated fine series config with subset grids.

    Example:
        >>> fine_config = create_fine_subset_workflow(
        ...     coarse_series_config=coarse_config,
        ...     fine_series_config=fine_config,
        ...     R_LC=10.0
        ... )
    """
    logger.info("Starting fine subset workflow")
    logger.info(f"Coarse series: {coarse_series_config.name}")
    logger.info(f"Fine series: {fine_series_config.name}")
    if r_max is not None:
        fine_series_config.tracing.max_integration_radius = float(r_max)
        logger.info(
            "Overriding fine tracing max_integration_radius to %.3f", float(r_max)
        )

    # Create subset series from OFL
    logger.info("Creating fine resolution subset grids from coarse OFL footpoints...")
    fine_series_config = fine_series_config.create_subset_series_from_OFL(
        coarse_series_config=coarse_series_config,
        R_LC=R_LC,
        h_LC_coeff=h_LC_coeff,
        experiment_numbers=experiment_numbers,
        save_configs=True,
    )

    logger.info("Fine subset workflow completed")
    return fine_series_config

