"""Workflow for extracting PA data filtered by theta tolerance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.services.batch_data_reader import BatchDataReader
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.io import save_data
from psr_magpy.utils.exceptions import PSRMagPyIOError, PSRMagPyPropertyError
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
from psr_magpy.line.pa_theta_filter import filter_field_line_set_by_theta

logger = get_logger(__name__, console_format_brief=True)

DEFAULT_EXTRACT_PA_OUTPUT_FILENAME_PATTERN = (
    "extract_PA_{experiment_number:03d}_RLC_{r_lc:05.2f}.fits"
)


# Template for loading batch data with required properties
_PA_EXTRACTION_TEMPLATE = FieldLineTemplate(
    save_footpoint_id=True,
    save_r_line=True,
    save_pa_line=True,
    save_tangent_theta=True,
    save_tangent_phi=True,
    save_reversed_error=True,
    save_r_cyl_max=True,
    save_z_abs_max=True,
)

# Template for saving filtered PA data (only properties that are actually present)
_PA_SAVE_TEMPLATE = FieldLineTemplate(
    save_footpoint_id=True,
    save_r_line=True,
    save_pa_line=True,
    save_tangent_theta=True,
    save_tangent_phi=True,
)



def _extract_pa_from_prepared_batch(
    prepared: Optional[FieldLineSet],
    *,
    grid_theta: np.ndarray,
    theta_tolerance: float,
) -> FieldLineSet:
    """Theta-filter prepared lines and return a compact :class:`FieldLineSet`.

    Expects ``prepare_line_batch_after_load`` to have filled ``pa_line`` via
    :func:`~psr_magpy.line.property_calculator.calculate_pa_line` (same as tracing).
    """
    if prepared is None or prepared.n_field_lines == 0:
        return FieldLineSet(_PA_SAVE_TEMPLATE, copy_data=False)
    try:
        filtered_lines = filter_field_line_set_by_theta(
            prepared,
            grid_theta,
            theta_tolerance,
            _PA_SAVE_TEMPLATE,
            require_pa=True,
        )
        if filtered_lines:
            return FieldLineSet(filtered_lines, copy_data=True)
        return FieldLineSet(_PA_SAVE_TEMPLATE, copy_data=False)
    except Exception as exc:
        logger.warning("Error extracting PA from prepared batch: %s", exc)
        return FieldLineSet(_PA_SAVE_TEMPLATE, copy_data=False)


def _field_line_from_load_template(
    load_template: Optional[Union[FieldLineTemplate, FieldLine]],
    line_source: LineDataSource,
) -> FieldLine:
    """Resolve FITS load template; defaults depend on ``line_source``."""
    if load_template is not None:
        if isinstance(load_template, FieldLineTemplate):
            return FieldLine.from_template(load_template)
        return load_template
    if line_source == "lines":
        return FieldLine.from_template(get_workflow_batch_load_template())
    if line_source == "data_products_oflr":
        return FieldLine.from_template(get_oflr_data_product_load_template())
    error_msg = f"Unknown line_source: {line_source!r}"
    logger.error(error_msg)
    raise ValueError(error_msg)


def _accumulate_pa_field_lines(
    acc: Optional[FieldLineSet], new: FieldLineSet
) -> FieldLineSet:
    if acc is None:
        return new
    if new.n_field_lines == 0:
        return acc
    acc.append(new)
    return acc


def _extract_pa_for_r_lc_batches(
    experiment_config: ExperimentConfig,
    batches_dir: str,
    load_template: FieldLine,
    r_lc: float,
    *,
    prepare_source: LineDataSource,
    trim_type: str,
    max_radius: Optional[float],
    r_min: Optional[float],
    reversed_error_threshold: float,
    grid_theta: np.ndarray,
    theta_tolerance: float,
    batch_filename_prefix: Optional[str] = None,
    target_template: Optional[FieldLineTemplate] = None,
) -> FieldLineSet:
    """Aggregate theta-filtered PA lines over batch FITS under ``batches_dir``."""
    reader = BatchDataReader(experiment_config.name)
    enrich_template = target_template or _PA_EXTRACTION_TEMPLATE

    def process_batch(batch_data: FieldLineSet) -> FieldLineSet:
        prepared = prepare_line_batch_after_load(
            batch_data,
            source=prepare_source,
            r_lc=float(r_lc),
            trim_type=trim_type,
            max_radius=max_radius,
            r_min=r_min,
            reversed_error_threshold=reversed_error_threshold,
            target_template=enrich_template,
        )
        return _extract_pa_from_prepared_batch(
            prepared,
            grid_theta=grid_theta,
            theta_tolerance=theta_tolerance,
        )

    filtered_result = reader.process_batches_incremental(
        template=load_template,
        lines_dir=batches_dir,
        processor_fn=process_batch,
        accumulator_fn=_accumulate_pa_field_lines,
        initial_value=None,
        batch_filename_prefix=batch_filename_prefix,
    )

    if filtered_result is None or filtered_result.n_field_lines == 0:
        logger.warning("No filtered PA data for R_LC = %.2f", r_lc)
        return FieldLineSet(_PA_SAVE_TEMPLATE, copy_data=False)

    logger.info(
        "Extracted %d filtered field lines for R_LC = %.2f",
        filtered_result.n_field_lines,
        r_lc,
    )
    return filtered_result


def _save_extracted_pa_field_line_set(
    field_line_set: FieldLineSet,
    filepath: Path,
) -> None:
    """Write one extracted-PA FITS (``field_line_set`` must use ``_PA_SAVE_TEMPLATE``)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    save_data(field_line_set, filepath, overwrite=True)


def make_extract_PA(
    experiment_config: ExperimentConfig,
    R_LC_values: np.ndarray,
    theta_tolerance: float,
    grid_theta: np.ndarray,
    *,
    parent_dir: Optional[Union[str, Path]] = None,
    series_name: Optional[str] = None,
    line_source: Optional[LineDataSource] = None,
    lines_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    trim_radius_coeff: float = 1.0,
    r_min: Optional[float] = None,
    trim_type: str = "cylinder",
    reversed_error_threshold: float = 0.01,
    output_filename_pattern: Optional[str] = None,
    load_template: Optional[Union[FieldLineTemplate, FieldLine]] = None,
    target_template: Optional[FieldLineTemplate] = None,
) -> Union[FieldLineSet, Dict[float, FieldLineSet]]:
    """Extract PA data filtered by theta tolerance from line batches.

    Batch layout and preparation match :func:`~psr_magpy.workflow.make_skymap.make_skymap`
    (``line_source``, ``parent_dir`` / ``series_name``, ``prepare_line_batch_after_load``).

    Parameters
    ----------
    experiment_config, R_LC_values, theta_tolerance, grid_theta
        Experiment metadata, light-cylinder radii, and observer colatitude grid.
    parent_dir, series_name, line_source
        Series root and batch source (``\"lines\"`` or ``\"data_products_oflr\"``).
        Preferred over legacy ``lines_dir``.
    lines_dir
        Legacy: raw ``lines/`` directory (equivalent to ``line_source=\"lines\"``).
    save, output_dir
        When ``save`` is True, write one FITS per ``R_LC`` immediately after that
        radius is processed (not deferred until all radii finish). Paths:
        ``extract_PA_{experiment_number:03d}_RLC_{r:.2f}.fits`` under
        ``<parent>/<series>/data_products/extracted_PA`` when ``output_dir`` is None.
    output_filename_pattern
        :meth:`str.format` pattern for saved FITS; default uses ``experiment_number``
        and ``r_lc``.
    trim_radius_coeff, r_min, trim_type, reversed_error_threshold
        Forwarded to :func:`prepare_line_batch_after_load`.
    load_template
        FITS columns to read per batch. If ``None``, ``\"lines\"`` uses
        :func:`~psr_magpy.workflow.line_batch_workflow_prep.get_workflow_batch_load_template`
        and ``\"data_products_oflr\"`` uses
        :func:`~psr_magpy.workflow.line_batch_workflow_prep.get_oflr_data_product_load_template`.
        Pass a template matching on-disk columns (e.g.
        :meth:`~psr_magpy.line.field_line_template.FieldLineTemplate.template_OFL_and_line_coordinates`
        for ``r_line`` without tangents); missing along-line fields are filled in
        ``prepare_line_batch_after_load`` via ``target_template``.
    target_template
        Properties to enrich before trim/filter. Default is ``_PA_EXTRACTION_TEMPLATE``.
    """
    if line_source is not None:
        if parent_dir is None or series_name is None:
            error_msg = "parent_dir and series_name are required when line_source is set"
            logger.error(error_msg)
            raise PSRMagPyIOError(error_msg)
    elif lines_dir is not None:
        line_source = "lines"
        batches_root = Path(lines_dir).expanduser().resolve()
        parent_dir = batches_root.parent.parent
        series_name = batches_root.parent.name
    else:
        error_msg = "Specify line_source with parent_dir/series_name, or lines_dir"
        logger.error(error_msg)
        raise PSRMagPyIOError(error_msg)

    parent = Path(parent_dir).expanduser().resolve()
    experiment_number = experiment_number_from_config(experiment_config)
    pa_pattern = (
        output_filename_pattern or DEFAULT_EXTRACT_PA_OUTPUT_FILENAME_PATTERN
    )

    load_field_line = _field_line_from_load_template(load_template, line_source)

    all_results: Dict[float, FieldLineSet] = {}
    out: Optional[Path] = None
    if save:
        out = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else parent / series_name / "data_products" / "extracted_PA"
        )
        out.mkdir(parents=True, exist_ok=True)

    for r_lc in R_LC_values:
        r_lc_f = float(r_lc)
        max_radius = float(trim_radius_coeff) * r_lc_f
        logger.info("Processing R_LC = %.2f", r_lc_f)
        logger.info(
            "  r_min=%s, trim_radius=%.2f",
            r_min if r_min is not None else "None",
            max_radius,
        )

        if line_source == "lines":
            batches_dir = str(resolve_lines_directory(parent, series_name))
            result = _extract_pa_for_r_lc_batches(
                experiment_config,
                batches_dir,
                load_field_line,
                r_lc_f,
                prepare_source="lines",
                trim_type=trim_type,
                max_radius=max_radius,
                r_min=r_min,
                reversed_error_threshold=reversed_error_threshold,
                grid_theta=grid_theta,
                theta_tolerance=theta_tolerance,
                target_template=target_template,
            )
        else:
            batches_dir = str(
                resolve_data_products_oflr_directory(parent, series_name)
            )
            oflr_prefix = (
                f"OFLR_lines_{experiment_number:03d}_RLC_{r_lc_f:05.2f}"
            )
            result = _extract_pa_for_r_lc_batches(
                experiment_config,
                batches_dir,
                load_field_line,
                r_lc_f,
                prepare_source="data_products_oflr",
                trim_type=trim_type,
                max_radius=max_radius,
                r_min=r_min,
                reversed_error_threshold=reversed_error_threshold,
                grid_theta=grid_theta,
                theta_tolerance=theta_tolerance,
                batch_filename_prefix=oflr_prefix,
                target_template=target_template,
            )

        all_results[r_lc_f] = result

        if save and out is not None:
            filepath = out / pa_pattern.format(
                experiment_number=experiment_number,
                experiment_name=experiment_config.name,
                r_lc=r_lc_f,
            )
            _save_extracted_pa_field_line_set(result, filepath)
            logger.info(
                "Saved filtered FieldLineSet for R_LC=%.2f to %s", r_lc_f, filepath
            )

    if len(R_LC_values) == 1:
        return all_results[float(R_LC_values[0])]
    return all_results


def extract_PA_processor(
    experiment_config: ExperimentConfig,
    r_lc: float,
    lines_dir: str,
    theta_tolerance: float,
    grid_theta: np.ndarray,
    save: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    radius_range: Optional[tuple] = None,
    trim_type: str = "cylinder",
    reversed_error_threshold: float = 0.01,
    load_template: Optional[Union[FieldLineTemplate, FieldLine]] = None,
    target_template: Optional[FieldLineTemplate] = None,
) -> FieldLineSet:
    """Extract PA from raw ``lines/`` batches (legacy single-directory API).

    Prefer :func:`make_extract_PA` with ``parent_dir``, ``series_name``, and
    ``line_source``.
    """
    min_radius = radius_range[0] if radius_range is not None else None
    max_radius = radius_range[1] if radius_range is not None else None

    load_field_line = _field_line_from_load_template(load_template, "lines")

    filtered_result = _extract_pa_for_r_lc_batches(
        experiment_config,
        lines_dir,
        load_field_line,
        float(r_lc),
        prepare_source="lines",
        trim_type=trim_type,
        max_radius=max_radius,
        r_min=min_radius,
        reversed_error_threshold=reversed_error_threshold,
        grid_theta=grid_theta,
        theta_tolerance=theta_tolerance,
        target_template=target_template,
    )

    if save:
        if output_path is None:
            output_dir = Path(lines_dir).parent / "data_products" / "extracted_PA"
            output_dir.mkdir(parents=True, exist_ok=True)
            pa_pattern = DEFAULT_EXTRACT_PA_OUTPUT_FILENAME_PATTERN
            exp_no = experiment_number_from_config(experiment_config)
            output_path = output_dir / pa_pattern.format(
                experiment_number=exp_no,
                experiment_name=experiment_config.name,
                r_lc=float(r_lc),
            )
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        _save_extracted_pa_field_line_set(filtered_result, Path(output_path))
        logger.info("Saved filtered FieldLineSet to %s", output_path)

    return filtered_result


