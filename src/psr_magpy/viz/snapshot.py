"""Data container and loader for PA snapshot figure rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from psr_magpy.experiment import create_full_healpix_grid_for_experiment
from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.grid_config import HealpixGridConfig
from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.io.data_products import (
    DataProductSpec,
    load_extracted_pa_product,
    load_footpoint_map_product,
)
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.physical_conversions import get_P_ms
from psr_magpy.viz.healpix import project_array_to_equirectangular, project_skymap_fits
from psr_magpy.workflow.line_batch_workflow_prep import experiment_number_from_config
from psr_magpy.workflow.make_skymap import format_skymap_output_filename
from psr_magpy.workflow.workflow_utils import create_healpix_grid

logger = get_logger(__name__)

ImageSize = Tuple[int, int]


@dataclass
class SnapshotData:
    """Static inputs for one PA snapshot (experiment × ``R_LC``).

    Loaded once per case; :math:`\\theta_\\mathrm{obs}`-dependent panels are built
    at render time from :attr:`field_line_set` and :attr:`grid`.
    """

    experiment_number: int
    R_LC: float
    series_config: ExperimentSeriesConfig

    skymap: np.ndarray
    skymap_longitude_deg: np.ndarray
    skymap_latitude_deg: np.ndarray

    oflr_map: np.ndarray
    grid_indices: np.ndarray
    oflr_image: np.ndarray
    oflr_longitude_deg: np.ndarray
    oflr_latitude_deg: np.ndarray

    field_line_set: FieldLineSet
    grid: HealpixGrid

    alpha_deg: float
    BQ: float
    period_ms: float

    nside_skymap: int
    nside_oflr: int
    image_size: ImageSize


def dipole_latitude_deg(experiment_config: ExperimentConfig) -> float:
    """Magnetic dipole inclination (degrees) from experiment field components."""
    for comp in experiment_config.field.components:
        if comp.moment_type == "dipole":
            return float(comp.moment_orientation_latitude_degree)
    error_msg = f"No dipole component in experiment {experiment_config.name!r}"
    logger.error(error_msg)
    raise ValueError(error_msg)


def quadrupole_B_magnitude(experiment_config: ExperimentConfig) -> float:
    """Quadrupole ``B_magnitude`` (f_Q) from experiment field components."""
    for comp in experiment_config.field.components:
        if comp.moment_type == "quadrupole":
            return float(comp.B_magnitude)
    error_msg = f"No quadrupole component in experiment {experiment_config.name!r}"
    logger.error(error_msg)
    raise ValueError(error_msg)


def categorize_oflr_map_for_r_lc(oflr_map: np.ndarray, r_lc: float) -> np.ndarray:
    """Threshold OFLR map values for snapshot surface plot at current ``r_lc``."""
    categorical_map = np.asarray(oflr_map).copy()
    categorical_map[oflr_map > r_lc] = r_lc
    categorical_map[oflr_map < r_lc] = np.nan
    categorical_map[oflr_map == r_lc] = 1
    return categorical_map


def resolve_skymap_product_path(
    data_dir: Union[str, Path],
    series_config: ExperimentSeriesConfig,
    experiment_number: int,
    r_lc: float,
    *,
    nside: int,
    skymap_filename_pattern: str,
) -> Path:
    """Resolve skymap FITS path (same naming as :func:`~psr_magpy.workflow.make_skymap`)."""
    data_dir = Path(data_dir)
    exp_config = series_config.load_experiment_config(experiment_number)
    exp_no = experiment_number_from_config(exp_config)
    grid = create_healpix_grid(HealpixGridConfig(nside=int(nside)))
    name = format_skymap_output_filename(
        skymap_filename_pattern,
        experiment_config=exp_config,
        series_name=series_config.name,
        experiment_number=exp_no,
        grid=grid,
        r_lc=float(r_lc),
    )
    return data_dir / series_config.name / "data_products" / "skymaps" / name


def load_snapshot_data(
    experiment_number: int,
    r_lc: float,
    series_config: ExperimentSeriesConfig,
    data_dir: Union[str, Path],
    products: Mapping[str, DataProductSpec],
    *,
    nside_skymap: int = 16,
    nside_oflr: int = 1024,
    image_size: ImageSize = (256, 256),
    skymap_filename_pattern: str,
) -> SnapshotData:
    """Load static data for one PA snapshot figure.

    Parameters
    ----------
    experiment_number, r_lc
        Case identifiers.
    series_config
        Experiment series configuration.
    data_dir
        Root data directory (heavy-computing parent).
    products
        Data-product registry (e.g. Multipole ``PA_SNAPSHOT_DATA_PRODUCTS``).
    nside_skymap, nside_oflr, image_size
        Projection resolution.
    skymap_filename_pattern
        Skymap :meth:`str.format` pattern for
        :func:`~psr_magpy.workflow.make_skymap.format_skymap_output_filename`.

    Returns
    -------
    SnapshotData
        Bundle for :func:`render_pa_snapshot_panel` (Phase 6).
    """
    data_dir = Path(data_dir)
    r_lc = float(r_lc)
    image_size = (int(image_size[0]), int(image_size[1]))

    logger.info(
        "Loading snapshot data for experiment %03d, R_LC=%.2f",
        experiment_number,
        r_lc,
    )

    skymap_path = resolve_skymap_product_path(
        data_dir,
        series_config,
        experiment_number,
        r_lc,
        nside=nside_skymap,
        skymap_filename_pattern=skymap_filename_pattern,
    )
    if not skymap_path.is_file():
        error_msg = f"Skymap file not found: {skymap_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    skymap, skymap_lon, skymap_lat = project_skymap_fits(
        skymap_path, image_size=image_size
    )

    oflr_map, grid_indices = load_footpoint_map_product(
        data_dir,
        series_config.name,
        "OFLR_map",
        experiment_number,
        products,
    )
    oflr_proj, oflr_lon, oflr_lat = project_array_to_equirectangular(
        oflr_map,
        grid_indices,
        int(nside_oflr),
        image_size=image_size,
    )
    oflr_image = categorize_oflr_map_for_r_lc(oflr_proj, r_lc)

    field_line_set = load_extracted_pa_product(
        data_dir,
        series_config.name,
        experiment_number,
        products,
        r_lc=r_lc,
    )

    exp_config = series_config.load_experiment_config(experiment_number)
    grid = create_full_healpix_grid_for_experiment(exp_config)
    alpha_deg = dipole_latitude_deg(exp_config)
    bq = quadrupole_B_magnitude(exp_config)
    period_ms = float(get_P_ms(r_lc))

    return SnapshotData(
        experiment_number=int(experiment_number),
        R_LC=r_lc,
        series_config=series_config,
        skymap=skymap,
        skymap_longitude_deg=skymap_lon,
        skymap_latitude_deg=skymap_lat,
        oflr_map=oflr_map,
        grid_indices=grid_indices,
        oflr_image=oflr_image,
        oflr_longitude_deg=oflr_lon,
        oflr_latitude_deg=oflr_lat,
        field_line_set=field_line_set,
        grid=grid,
        alpha_deg=alpha_deg,
        BQ=bq,
        period_ms=period_ms,
        nside_skymap=int(nside_skymap),
        nside_oflr=int(nside_oflr),
        image_size=image_size,
    )
