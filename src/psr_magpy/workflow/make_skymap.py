"""Workflow for creating skymaps from experiment data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import healpy as hp
import numpy as np

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.grid_config import HealpixGridConfig
from psr_magpy.experiment.services.batch_data_reader import BatchDataReader
from psr_magpy.grid.grid import FootpointGrid, HealpixGrid
from psr_magpy.io import load_data
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.exceptions import PSRMagPyIOError
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.workflow.line_batch_workflow_prep import (
    LineDataSource,
    experiment_number_from_config,
    get_workflow_batch_load_template,
    prepare_line_batch_after_load,
    resolve_lines_directory,
    resolve_oflr_lines_file_path,
)
from psr_magpy.workflow.workflow_utils import (
    create_healpix_grid,
    extract_line_properties,
    normalize_line_arrays,
)

logger = get_logger(__name__, console_format_brief=True)


def skymap_pixel_counts_from_prepared_batch(
    batch_subset: Optional[FieldLineSet],
    skymap_grid: FootpointGrid,
) -> np.ndarray:
    """Count field-line points in HEALPix bins for one prepared :class:`FieldLineSet`.

    Parameters
    ----------
    batch_subset
        Output of :func:`~psr_magpy.workflow.line_batch_workflow_prep.prepare_line_batch_after_load`
        (or ``None`` / empty).
    skymap_grid
        HEALPix (or compatible) grid with ``nside`` and ``nline`` set.

    Returns
    -------
    np.ndarray
        Integer counts per pixel, length ``skymap_grid.nline``.
    """
    if skymap_grid.nline is None:
        raise PSRMagPyIOError("Skymap grid nline is None")

    grid_size = skymap_grid.nline
    if batch_subset is None or batch_subset.n_field_lines == 0:
        return np.zeros(grid_size, dtype=np.int64)

    try:
        tangent_theta_lines, tangent_phi_lines, r_line_data, _ = extract_line_properties(
            batch_subset, require_pa=False
        )

        pixel_counts = np.zeros(grid_size, dtype=np.int64)

        for theta_line, phi_line, r_line in zip(
            tangent_theta_lines, tangent_phi_lines, r_line_data
        ):
            if theta_line is None or phi_line is None or r_line is None:
                continue

            result = normalize_line_arrays(theta_line, phi_line, r_line)
            if result is None:
                continue
            theta_line, phi_line, r_line, _point_radii = result

            theta_line_filtered = np.asarray(theta_line, dtype=np.float64).flatten()
            phi_line_filtered = np.asarray(phi_line, dtype=np.float64).flatten()

            if theta_line_filtered.ndim == 0:
                theta_line_filtered = np.array([theta_line_filtered])
            if phi_line_filtered.ndim == 0:
                phi_line_filtered = np.array([phi_line_filtered])

            theta_rad = np.deg2rad(theta_line_filtered)
            phi_rad = np.deg2rad(phi_line_filtered)

            valid_mask = (
                np.isfinite(theta_rad)
                & np.isfinite(phi_rad)
                & (theta_rad >= 0)
                & (theta_rad <= np.pi)
            )
            if not np.any(valid_mask):
                continue

            theta_rad_valid = theta_rad[valid_mask]
            phi_rad_valid = phi_rad[valid_mask]

            pixel_indices = hp.ang2pix(skymap_grid.nside, theta_rad_valid, phi_rad_valid)

            unique_pixels, counts = np.unique(pixel_indices, return_counts=True)
            valid_pixel_mask = (unique_pixels >= 0) & (unique_pixels < grid_size)
            valid_pixels = unique_pixels[valid_pixel_mask]
            valid_counts = counts[valid_pixel_mask]
            pixel_counts[valid_pixels] += valid_counts

        return pixel_counts
    except Exception as e:
        logger.warning("Error counting skymap pixels from batch: %s", e)
        return np.zeros(grid_size, dtype=np.int64)


def _accumulate_counts(acc: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
    if acc is None:
        return new.copy()
    return acc + new


def _skymap_for_r_lc_lines(
    experiment_config: ExperimentConfig,
    lines_dir: str,
    load_template: FieldLine,
    grid: HealpixGrid,
    r_lc: float,
    *,
    trim_type: str,
    max_radius: Optional[float],
    r_min: Optional[float],
    reversed_error_threshold: float,
    target_template: Optional[FieldLineTemplate],
    threshold_line_distance: float,
    field: Any,
) -> np.ndarray:
    """Aggregate pixel counts over all batch FITS under ``lines_dir``."""
    grid_size = grid.nline
    if grid_size is None:
        raise PSRMagPyIOError("Skymap grid nline is None")

    reader = BatchDataReader(experiment_config.name)

    def process_batch(batch_data: FieldLineSet) -> np.ndarray:
        prepared = prepare_line_batch_after_load(
            batch_data,
            source="lines",
            r_lc=r_lc,
            trim_type=trim_type,
            max_radius=max_radius,
            r_min=r_min,
            reversed_error_threshold=reversed_error_threshold,
            target_template=target_template,
            threshold_line_distance=threshold_line_distance,
            field=field,
        )
        return skymap_pixel_counts_from_prepared_batch(prepared, grid)

    counts = reader.process_batches_incremental(
        template=load_template,
        lines_dir=lines_dir,
        processor_fn=process_batch,
        accumulator_fn=_accumulate_counts,
        initial_value=None,
    )

    if counts is None:
        logger.warning("No batch data processed for R_LC = %.2f", r_lc)
        return np.zeros(grid_size, dtype=np.float64)

    total_points = float(np.sum(counts))
    logger.info("Counted %.0f points in HEALPix bins for R_LC = %.1f", total_points, r_lc)
    return counts.astype(np.float64)


def _skymap_for_r_lc_oflr(
    oflr_path: Path,
    load_template: FieldLine,
    grid: HealpixGrid,
    r_lc: float,
    *,
    trim_type: str,
    max_radius: Optional[float],
    r_min: Optional[float],
    reversed_error_threshold: float,
    target_template: Optional[FieldLineTemplate],
    threshold_line_distance: float,
    field: Any,
) -> np.ndarray:
    """Pixel counts from a single OFLR reintegrated lines FITS (one ``R_LC``)."""
    grid_size = grid.nline
    if grid_size is None:
        raise PSRMagPyIOError("Skymap grid nline is None")

    if not oflr_path.is_file():
        logger.warning("OFLR lines file missing: %s", oflr_path)
        return np.zeros(grid_size, dtype=np.float64)

    batch = load_data(FieldLineSet, oflr_path, template=load_template)
    if batch is None or batch.n_field_lines == 0:
        logger.warning("Empty or missing batch in %s", oflr_path)
        return np.zeros(grid_size, dtype=np.float64)

    prepared = prepare_line_batch_after_load(
        batch,
        source="data_products_oflr",
        r_lc=r_lc,
        trim_type=trim_type,
        max_radius=max_radius,
        r_min=r_min,
        reversed_error_threshold=reversed_error_threshold,
        target_template=target_template,
        threshold_line_distance=threshold_line_distance,
        field=field,
    )
    counts = skymap_pixel_counts_from_prepared_batch(prepared, grid)
    total_points = float(np.sum(counts))
    logger.info("Counted %.0f points for R_LC = %.1f from %s", total_points, r_lc, oflr_path.name)
    return counts.astype(np.float64)


def make_skymap(
    experiment_config: ExperimentConfig,
    parent_dir: Union[str, Path],
    series_name: str,
    line_source: LineDataSource,
    healpix_grid_config: HealpixGridConfig,
    R_LC_values: np.ndarray,
    *,
    save: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    trim_radius_coeff: float = 1.0,
    r_min: Optional[float] = None,
    trim_type: str = "cylinder",
    reversed_error_threshold: float = 0.01,
    target_template: Optional[FieldLineTemplate] = None,
    threshold_line_distance: float = 0.0,
    field: Any = None,
) -> Union[np.ndarray, Dict[float, np.ndarray]]:
    """Create HEALPix skymaps from line batches under ``parent_dir`` / ``series_name``.

    Parameters
    ----------
    experiment_config
        Experiment metadata (name for batch glob; number for OFLR filenames).
    parent_dir
        Series root, e.g. the heavy-data directory containing ``<series_name>/lines``.
    series_name
        Subdirectory name for the series (folder that contains ``lines`` or
        ``data_products/OFLR``).
    line_source
        ``\"lines\"``: many ``*_batch_*.fits`` under ``<parent>/<series>/lines``.
        ``\"data_products_oflr\"``: one ``OFLR_lines_*.fits`` per ``R_LC``.
    healpix_grid_config, R_LC_values
        Grid config and light-cylinder radii to iterate (one skymap per value).
    save
        Write one FITS per ``R_LC`` under ``output_dir`` (or default skymaps folder).
    output_dir
        If ``None`` and ``save`` is True: ``<parent_dir>/<series_name>/data_products/skymaps``.
    trim_radius_coeff
        Outer trim radius is ``trim_radius_coeff * r_lc`` (passed as ``max_radius`` to batch prep).
    r_min
        Minimum spherical radius ``||r||``; leading points below this are dropped
        in :func:`prepare_line_batch_after_load`. None skips inner trim.
    trim_type
        ``\"cylinder\"`` or ``\"sphere\"`` for outer :meth:`FieldLineSet.trim_to_light_cylinder`.
    reversed_error_threshold
        Passed to :func:`prepare_line_batch_after_load` (reversed-error filter when present).
    target_template, threshold_line_distance, field
        Forwarded to :func:`prepare_line_batch_after_load` / enrichment.

    Returns
    -------
    np.ndarray or dict
        Single array if ``len(R_LC_values)==1``, else ``{float(r_lc): array}``.
    """
    grid = create_healpix_grid(healpix_grid_config)
    grid_size = grid.nline
    if grid_size is None:
        raise PSRMagPyIOError("Skymap grid nline is None")

    load_template = FieldLine.from_template(get_workflow_batch_load_template())
    parent = Path(parent_dir).expanduser().resolve()

    all_skymaps: Dict[float, np.ndarray] = {}

    for r_lc in R_LC_values:
        r_lc_f = float(r_lc)
        logger.info("Processing R_LC = %.2f", r_lc_f)
        max_radius = trim_radius_coeff * r_lc_f
        logger.info("  trim_radius = %.2f", max_radius)

        if line_source == "lines":
            lines_path = str(resolve_lines_directory(parent, series_name))
            result = _skymap_for_r_lc_lines(
                experiment_config,
                lines_path,
                load_template,
                grid,
                r_lc_f,
                trim_type=trim_type,
                max_radius=max_radius,
                r_min=r_min,
                reversed_error_threshold=reversed_error_threshold,
                target_template=target_template,
                threshold_line_distance=threshold_line_distance,
                field=field,
            )
        elif line_source == "data_products_oflr":
            exp_num = experiment_number_from_config(experiment_config)
            oflr_path = resolve_oflr_lines_file_path(parent, series_name, exp_num, r_lc_f)
            result = _skymap_for_r_lc_oflr(
                oflr_path,
                load_template,
                grid,
                r_lc_f,
                trim_type=trim_type,
                max_radius=max_radius,
                r_min=r_min,
                reversed_error_threshold=reversed_error_threshold,
                target_template=target_template,
                threshold_line_distance=threshold_line_distance,
                field=field,
            )
        else:
            raise ValueError(f"Unknown line_source: {line_source!r}")

        if result.shape[0] != grid_size:
            raise ValueError(
                f"Skymap size {result.shape[0]} does not match grid size {grid_size}"
            )
        all_skymaps[r_lc_f] = result

    if save:
        out = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else parent / series_name / "data_products" / "skymaps"
        )
        out.mkdir(parents=True, exist_ok=True)

        for r_lc_f, skymap in all_skymaps.items():
            filename = (
                f"skymap_{experiment_config.name}_nside{grid.nside:03d}_RLC_{r_lc_f:05.2f}.fits"
            )
            filepath = out / filename
            _save_skymap_fits(
                skymap,
                grid,
                filepath,
                experiment_config.name,
                np.array([r_lc_f]),
                trim_radius_coeff,
            )
            logger.info("Saved skymap for R_LC=%.2f to %s", r_lc_f, filepath)

    if len(R_LC_values) == 1:
        return all_skymaps[float(R_LC_values[0])]
    return all_skymaps


def _save_skymap_fits(
    skymap: np.ndarray,
    grid: HealpixGrid,
    filepath: Path,
    experiment_name: str,
    R_LC_values: np.ndarray,
    trim_radius_coeff: float,
) -> None:
    """Save skymap to FITS file."""
    from astropy.io import fits

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["EXPERIMENT"] = experiment_name
    primary_hdu.header["NSIDE"] = grid.nside
    primary_hdu.header["NPIX"] = len(skymap)
    primary_hdu.header["NRLC"] = len(R_LC_values)
    primary_hdu.header["TRIM_COEFF"] = (trim_radius_coeff, "Trim radius coefficient")
    primary_hdu.header["COMMENT"] = "HEALPix skymap"

    for i, r_lc in enumerate(R_LC_values):
        primary_hdu.header[f"RLC_{i}"] = (r_lc, f"Light cylinder radius {i}")

    columns = [fits.Column(name="SKYMAP", format="D", array=skymap)]
    data_hdu = fits.BinTableHDU.from_columns(columns)
    hdul = fits.HDUList([primary_hdu, data_hdu])
    hdul.writeto(filepath, overwrite=True)
    hdul.close()
