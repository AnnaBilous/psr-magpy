"""Workflow for creating skymaps from experiment data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import healpy as hp
import numpy as np

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.grid_config import HealpixGridConfig, create_full_healpix_grid
from psr_magpy.line.tracing_parameters import TracingDefaults
from psr_magpy.experiment.services.batch_data_reader import BatchDataReader
from psr_magpy.grid.grid import FootpointGrid, HealpixGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.exceptions import PSRMagPyIOError
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.workflow.line_batch_workflow_prep import (
    LineDataSource,
    experiment_number_from_config,
    get_oflr_data_product_load_template,
    get_workflow_batch_load_template,
    prepare_line_batch_after_load,
    resolve_data_products_oflr_directory,
    resolve_lines_directory,
)
from psr_magpy.workflow.workflow_utils import (
    extract_line_properties,
    normalize_line_arrays,
)

logger = get_logger(__name__, console_format_brief=True)

DEFAULT_SKYMAP_OUTPUT_FILENAME_PATTERN = (
    "skymap_{experiment_number:03d}_RLC_{r_lc:05.2f}.fits"
)

DEFAULT_OFLR_LINES_BATCH_STEM_PATTERN = (
    "OFLR_lines_{experiment_number:03d}_RLC_{r_lc:05.2f}"
)


def format_skymap_output_filename(
    pattern: str,
    *,
    experiment_config: ExperimentConfig,
    series_name: str,
    experiment_number: int,
    r_lc: float,
    grid: Optional[HealpixGrid] = None,
) -> str:
    """Build a skymap FITS filename from a :meth:`str.format` pattern.

    Available fields: ``experiment_number``, ``experiment_name``, ``series_name``,
    ``r_lc``, and optionally ``nside`` if the pattern references it.
    """
    fields = {
        "experiment_number": int(experiment_number),
        "experiment_name": experiment_config.name,
        "series_name": series_name,
        "r_lc": float(r_lc),
    }
    if grid is not None:
        fields["nside"] = int(grid.nside)
    name = pattern.format(**fields)
    lower = name.lower()
    if not (lower.endswith(".fits") or lower.endswith(".fit")):
        name = f"{name}.fits"
    return name


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
    except Exception as exc:
        error_msg = f"Error counting skymap pixels from batch: {exc}"
        logger.error(error_msg)
        raise


def _accumulate_counts(acc: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
    if acc is None:
        return new.copy()
    return acc + new


def _skymap_for_r_lc_batches(
    experiment_config: ExperimentConfig,
    batches_dir: str,
    load_template: FieldLine,
    grid: HealpixGrid,
    r_lc: float,
    *,
    prepare_source: LineDataSource,
    trim_type: str,
    max_radius: Optional[float],
    r_min: Optional[float],
    reversed_error_threshold: float,
    target_template: Optional[FieldLineTemplate],
    threshold_line_distance: float,
    field: Any,
    batch_filename_prefix: Optional[str] = None,
) -> np.ndarray:
    """Aggregate pixel counts over batch FITS under ``batches_dir``.

    Batch stem is ``{batch_filename_prefix}_batch_*.fits`` when ``batch_filename_prefix``
    is set; otherwise ``{experiment_config.name}_batch_*.fits``. If no batch
    chunks exist, loads a single ``{prefix}.fits`` when present.
    """
    grid_size = grid.nline
    if grid_size is None:
        raise PSRMagPyIOError("Skymap grid nline is None")

    reader = BatchDataReader(experiment_config.name)

    def process_batch(batch_data: FieldLineSet) -> np.ndarray:
        prepared = prepare_line_batch_after_load(
            batch_data,
            source=prepare_source,
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
        lines_dir=batches_dir,
        processor_fn=process_batch,
        accumulator_fn=_accumulate_counts,
        initial_value=None,
        batch_filename_prefix=batch_filename_prefix,
    )

    if counts is None:
        logger.warning("No batch data processed for R_LC = %.2f", r_lc)
        return np.zeros(grid_size, dtype=np.float64)

    total_points = float(np.sum(counts))
    logger.info("Counted %.0f points in HEALPix bins for R_LC = %.1f", total_points, r_lc)
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
    threshold_line_distance: float = TracingDefaults.THRESHOLD_LINE_DISTANCE,
    field: Any = None,
    skymap_output_filename_pattern: Optional[str] = None,
    oflr_batch_stem_pattern: Optional[str] = None,
) -> Union[np.ndarray, Dict[float, np.ndarray]]:
    """Create HEALPix skymaps from line batches under ``parent_dir`` / ``series_name``.

    Parameters
    ----------
    experiment_config
        Experiment metadata (``name`` for batch FITS glob under ``lines`` or OFLR dir).
    parent_dir
        Series root, e.g. the heavy-data directory containing ``<series_name>/lines``.
    series_name
        Subdirectory name for the series (folder that contains ``lines`` or
        ``data_products/OFLR``).
    line_source
        ``\"lines\"``: many ``{experiment}_batch_*.fits`` under ``<parent>/<series>/lines``
        (typically ``*_batch_*_out_of_*.fits``).
        ``\"data_products_oflr\"``: reintegration spill files under ``data_products/OFLR``,
        stem ``OFLR_lines_{experiment_number:03d}_RLC_{r:.2f}_batch_*.fits`` (same stem as
        trace ``filename`` before ``BatchFileManager`` adds ``_out_of_``); falls back to
        ``OFLR_lines_…_RLC_….fits`` if no chunks exist. FITS are loaded with
        :func:`~psr_magpy.workflow.line_batch_workflow_prep.get_oflr_data_product_load_template`
        (``r_line`` on disk); :func:`prepare_line_batch_after_load` enriches to the workflow template.
    healpix_grid_config
        HEALPix settings. Only ``nside`` is used. ``subset`` and ``grid_file_path``
        are ignored, so the map is a full sphere (``12 * nside**2`` pixels).
    R_LC_values
        Light-cylinder radii to iterate (one skymap per value).
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
    target_template
        Forwarded to :func:`prepare_line_batch_after_load`.
    threshold_line_distance
        Minimum ``line_R_max`` before missing tangents are recomputed. Default is
        the tracing default, ``1`` stellar radius. Pass ``0`` to recompute for any
        non-empty line.
    field
        Forwarded to :func:`prepare_line_batch_after_load` when ``b_line`` must be filled.
    skymap_output_filename_pattern
        :meth:`str.format` pattern for each saved FITS basename under ``output_dir``.
        Fields: ``experiment_number``, ``experiment_name``, ``series_name``, ``r_lc``,
        ``nside``. Default matches historical names
        (``skymap_{experiment_number:03d}_RLC_{r_lc:05.2f}.fits`` by default).
    oflr_batch_stem_pattern
        Stem for OFLR batch glob (no ``.fits``); default ``OFLR_lines_{nnn}_RLC_{r}``.

    Returns
    -------
    np.ndarray or dict
        Single array if ``len(R_LC_values)==1``, else ``{float(r_lc): array}``.
    """
    grid = create_full_healpix_grid(healpix_grid_config)
    grid_size = grid.nline
    if grid_size is None:
        raise PSRMagPyIOError("Skymap grid nline is None")

    if line_source == "lines":
        load_template = FieldLine.from_template(get_workflow_batch_load_template())
    elif line_source == "data_products_oflr":
        load_template = FieldLine.from_template(get_oflr_data_product_load_template())
    else:
        raise ValueError(f"Unknown line_source: {line_source!r}")

    parent = Path(parent_dir).expanduser().resolve()
    experiment_number = experiment_number_from_config(experiment_config)
    filename_pattern = (
        skymap_output_filename_pattern or DEFAULT_SKYMAP_OUTPUT_FILENAME_PATTERN
    )
    oflr_stem_pattern = (
        oflr_batch_stem_pattern or DEFAULT_OFLR_LINES_BATCH_STEM_PATTERN
    )

    all_skymaps: Dict[float, np.ndarray] = {}

    for r_lc in R_LC_values:
        r_lc_f = float(r_lc)
        logger.info("Processing R_LC = %.2f", r_lc_f)
        max_radius = trim_radius_coeff * r_lc_f
        logger.info("  trim_radius = %.2f", max_radius)

        if line_source == "lines":
            batches_dir = str(resolve_lines_directory(parent, series_name))
            result = _skymap_for_r_lc_batches(
                experiment_config,
                batches_dir,
                load_template,
                grid,
                r_lc_f,
                prepare_source="lines",
                trim_type=trim_type,
                max_radius=max_radius,
                r_min=r_min,
                reversed_error_threshold=reversed_error_threshold,
                target_template=target_template,
                threshold_line_distance=threshold_line_distance,
                field=field,
            )
        elif line_source == "data_products_oflr":
            batches_dir = str(resolve_data_products_oflr_directory(parent, series_name))
            oflr_prefix = oflr_stem_pattern.format(
                experiment_number=experiment_number,
                experiment_name=experiment_config.name,
                series_name=series_name,
                r_lc=r_lc_f,
            )
            result = _skymap_for_r_lc_batches(
                experiment_config,
                batches_dir,
                load_template,
                grid,
                r_lc_f,
                prepare_source="data_products_oflr",
                trim_type=trim_type,
                max_radius=max_radius,
                r_min=r_min,
                reversed_error_threshold=reversed_error_threshold,
                target_template=target_template,
                threshold_line_distance=threshold_line_distance,
                field=field,
                batch_filename_prefix=oflr_prefix,
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
            filename = format_skymap_output_filename(
                filename_pattern,
                experiment_config=experiment_config,
                series_name=series_name,
                experiment_number=experiment_number,
                grid=grid,
                r_lc=r_lc_f,
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
