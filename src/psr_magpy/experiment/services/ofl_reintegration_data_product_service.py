"""Build OFL-based line data products from existing experiment line batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Literal, Union

import numpy as np

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.line.calculation_context import CalculationContext
from psr_magpy.line.calculator import FieldLineCalculator
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.open_field_line_edges import (
    OFLEdgeCalculator,
    OFLEdgeResult,
    _load_experiment_setup,
)
from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.utils.exceptions import (
    PSRMagPyExperimentError,
    PSRMagPyGridError,
    PSRMagPyIOError,
    PSRMagPyPropertyError,
)
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)

SelectionMode = Literal["edge", "region"]


@dataclass
class ReintegrationDataProductResult:
    """Outcome for one experiment data-product build."""

    experiment_number: int
    selection_mode: SelectionMode
    output_dir: Path
    r_lc_values: np.ndarray
    selected_footpoint_counts: List[int]


class OFLReintegrationDataProductService:
    """Create OFL edge/region data products under per-experiment data_products folders."""

    def _resolve_r_max_values(
        self,
        *,
        r_max: Optional[Union[float, np.ndarray, Dict[float, float]]],
        r_lc_values: np.ndarray,
    ) -> List[Optional[float]]:
        """Resolve per-R_LC r_max values from scalar/array/dict input."""
        n_values = len(r_lc_values)
        if r_max is None:
            return [None] * n_values
        if isinstance(r_max, (int, float)):
            return [float(r_max)] * n_values
        if isinstance(r_max, np.ndarray):
            if len(r_max) != n_values:
                error_msg = (
                    f"r_max array length ({len(r_max)}) must match r_lc_values "
                    f"length ({n_values})"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            return [float(x) for x in r_max.tolist()]
        if isinstance(r_max, dict):
            resolved: List[Optional[float]] = []
            for r_lc in r_lc_values:
                key = float(r_lc)
                if key not in r_max:
                    error_msg = f"r_max dictionary missing key for R_LC = {key:.2f}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                resolved.append(float(r_max[key]))
            return resolved
        error_msg = (
            "r_max must be float, np.ndarray, dict[float, float], or None; "
            f"got {type(r_max)}"
        )
        logger.error(error_msg)
        raise TypeError(error_msg)

    def _calculate_footpoints_only(
        self,
        *,
        experiment_config_path: Path,
        lines_dir: Path,
        r_lc_values: np.ndarray,
        h_lc_coeff: float,
        selection_mode: SelectionMode,
        reversed_error_threshold: Optional[float] = None,
    ) -> OFLEdgeResult:
        """Compute OFL edge/region footpoint IDs only (no extraction/saving)."""
        batch_load_template = None
        if selection_mode == "region" and reversed_error_threshold is not None:
            batch_load_template = FieldLine(
                save_footpoint_id=True,
                save_r_cyl_max=True,
                save_z_abs_max=True,
                save_reversed_error=True,
            )
        try:
            _, grid, ofl_data, _ = _load_experiment_setup(
                experiment_config_path=experiment_config_path,
                lines_dir=lines_dir,
                load_template=None,
                batch_load_template=batch_load_template,
            )
        except PSRMagPyIOError as exc:
            # Fallback: allow region selection even when legacy files do not
            # include reversed_error column in batch data.
            if (
                selection_mode == "region"
                and reversed_error_threshold is not None
                and "reversed_error" in str(exc)
            ):
                logger.warning(
                    "reversed_error_threshold=%.3e requested but reversed_error "
                    "column is unavailable for %s; continuing without filtering.",
                    reversed_error_threshold,
                    experiment_config_path,
                )
                _, grid, ofl_data, _ = _load_experiment_setup(
                    experiment_config_path=experiment_config_path,
                    lines_dir=lines_dir,
                    load_template=None,
                    batch_load_template=None,
                )
                reversed_error_threshold = None
            else:
                raise
        try:
            if selection_mode == "edge":
                if not isinstance(grid, HealpixGrid):
                    error_msg = (
                        "Precise edge calculation requires HealpixGrid got "
                        f"{type(grid).__name__}."
                    )
                    logger.error(error_msg)
                    raise PSRMagPyGridError(error_msg)

                calculator = OFLEdgeCalculator(
                    grid, h_LC_coeff=h_lc_coeff, precise_edges=True
                )
                return calculator.calculate_edges(
                    ofl_data,
                    R_LC=r_lc_values,
                )

            if selection_mode == "region":
                working = ofl_data
                if reversed_error_threshold is not None:
                    try:
                        working = working.filter_by_reversed_error(
                            reversed_error_threshold
                        )
                    except PSRMagPyPropertyError:
                        logger.warning(
                            "reversed_error_threshold=%.3e requested but "
                            "reversed_error property is missing for %s; "
                            "continuing without filtering.",
                            reversed_error_threshold,
                            experiment_config_path,
                        )
                if reversed_error_threshold is not None:
                    region_footpoints_unfiltered = [
                        ofl_data.get_OFL_footpoints(r_lc, h_LC_coeff=h_lc_coeff)
                        for r_lc in r_lc_values
                    ]
                    region_footpoints = [
                        working.get_OFL_footpoints(r_lc, h_LC_coeff=h_lc_coeff)
                        for r_lc in r_lc_values
                    ]
                    for r_lc, unfiltered_ids, filtered_ids in zip(
                        r_lc_values, region_footpoints_unfiltered, region_footpoints
                    ):
                        excluded_count = len(unfiltered_ids) - len(filtered_ids)
                        logger.info(
                            "Region selection R_LC=%.2f: excluded %d footpoints "
                            "by reversed_error threshold %.3e",
                            r_lc,
                            excluded_count,
                            reversed_error_threshold,
                        )
                else:
                    region_footpoints = [
                        working.get_OFL_footpoints(r_lc, h_LC_coeff=h_lc_coeff)
                        for r_lc in r_lc_values
                    ]
                return OFLEdgeResult(
                    R_LC=np.asarray(r_lc_values, dtype=float),
                    edge_footpoints=region_footpoints,
                    precise_edges=False,
                )

            error_msg = f"Unsupported selection_mode: {selection_mode}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as exc:
            error_msg = (
                f"Failed to calculate OFL {selection_mode} footpoints: {exc}"
            )
            logger.error(error_msg)
            raise PSRMagPyExperimentError(error_msg) from exc

    def _integrate_selected_footpoints(
        self,
        *,
        experiment_config_path: Path,
        selection_mode: SelectionMode,
        output_dir: Path,
        r_lc_values: np.ndarray,
        selected_footpoints: List[np.ndarray],
        template: FieldLineTemplate,
        r_max: Optional[Union[float, np.ndarray, Dict[float, float]]] = None,
        skip_verify_by_reversing: bool = True,
        n_cores: Optional[int] = None,
        dt_reint: Optional[float] = None,
    ) -> None:
        """Integrate selected footpoints and save line data products."""
        experiment_config = ExperimentConfig.load(experiment_config_path)
        base_grid = experiment_config.grid.create_grid()
        if base_grid is None:
            error_msg = f"Failed to create grid for {experiment_config.name}"
            logger.error(error_msg)
            raise PSRMagPyGridError(error_msg)
        if base_grid.indices is None:
            error_msg = f"Grid indices are missing for {experiment_config.name}"
            logger.error(error_msg)
            raise PSRMagPyGridError(error_msg)

        footpoint_id_to_position = {
            int(footpoint_id): i
            for i, footpoint_id in enumerate(base_grid.indices.tolist())
        }

        field = experiment_config.field.create_field()
        # Build base tracing params with output disabled to avoid validation
        # failures when source experiment config has save_results_to_disk=True
        # but no filename (common for series-level configs).
        tracing_update: Dict[str, Any] = {
            "save_results_to_disk": False,
            "filename": None,
            # Reintegration is data-product generation; reverse check can
            # be disabled to avoid effectively tracing each line twice.
            "verify_by_reversing": not skip_verify_by_reversing,
        }
        if n_cores is not None:
            tracing_update["n_cores"] = int(n_cores)
        if dt_reint is not None:
            tracing_update["dt"] = float(dt_reint)
        tracing_config_base = experiment_config.tracing.model_copy(
            update=tracing_update
        )
        base_params = tracing_config_base.create_tracing_parameters()
        r_max_values = self._resolve_r_max_values(r_max=r_max, r_lc_values=r_lc_values)

        prefix = "OFLR_edge_lines" if selection_mode == "edge" else "OFLR_lines"
        experiment_number = None
        if experiment_config.metadata and "experiment_number" in experiment_config.metadata:
            experiment_number = int(experiment_config.metadata["experiment_number"])
        else:
            experiment_number = int(experiment_config.name.rsplit("_", 1)[1])

        for r_lc, footpoint_ids, r_max_cur in zip(
            r_lc_values, selected_footpoints, r_max_values
        ):
            if len(footpoint_ids) == 0:
                logger.warning(
                    "Skipping integration for R_LC=%.2f: no selected footpoints", r_lc
                )
                continue

            subset_positions = np.asarray(
                [
                    footpoint_id_to_position[int(footpoint_id)]
                    for footpoint_id in footpoint_ids
                    if int(footpoint_id) in footpoint_id_to_position
                ],
                dtype=int,
            )
            if subset_positions.size == 0:
                logger.warning(
                    "Skipping integration for R_LC=%.2f: selected footpoints not in grid",
                    r_lc,
                )
                continue

            subset_grid = base_grid.subset_grid(subset_positions)
            output_filename = (
                output_dir
                / f"{prefix}_{experiment_number:03d}_RLC_{float(r_lc):05.2f}.fits"
            )
            tracing_params = base_params
            if r_max_cur is not None:
                if base_params.integration.threshold_line_distance >= r_max_cur:
                    error_msg = (
                        "r_max must be greater than threshold_line_distance "
                        f"({base_params.integration.threshold_line_distance}), got {r_max_cur}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                integration_params = base_params.integration.model_copy(
                    update={"max_integration_radius": float(r_max_cur)}
                )
                tracing_params = base_params.model_copy(
                    update={"integration": integration_params}
                )

            output_params = tracing_params.output.model_copy(
                update={
                    "save_results_to_disk": True,
                    "keep_results_in_memory": False,
                    "filename": str(output_filename),
                    "overwrite": True,
                }
            )
            tracing_params = tracing_params.model_copy(update={"output": output_params})

            context = CalculationContext(field, subset_grid, tracing_params)
            calculator = FieldLineCalculator(context)
            calculator.get_field_lines(template)

            logger.info(
                "Saved reintegrated %s lines for R_LC=%.2f to %s",
                selection_mode,
                r_lc,
                output_filename,
            )

    def run(
        self,
        *,
        series_config: ExperimentSeriesConfig,
        experiment_numbers: Sequence[int],
        r_lc_values: Sequence[float],
        r_max: Optional[Union[float, np.ndarray, Dict[float, float]]] = None,
        selection_mode: SelectionMode = "edge",
        template: Optional[FieldLineTemplate] = None,
        h_lc_coeff: float = 3**1.5 / 2.0,
        reversed_error_threshold: Optional[float] = None,
        skip_verify_by_reversing: bool = True,
        n_cores: Optional[int] = None,
        dt_reint: Optional[float] = None,
    ) -> Dict[int, ReintegrationDataProductResult]:
        """Build OFL data products for selected experiments.

        Notes
        -----
        - This service does not persist any target experiment config.
        - Outputs are always written to data-products folders.
        - ``n_cores``, if set, overrides each experiment YAML for the integration step.
        - For ``selection_mode="region"``, ``reversed_error_threshold`` keeps lines with
          ``reversed_error <= threshold`` (and NaNs); requires batches saved with
          ``reversed_error``.
        - If ``reversed_error_threshold`` is requested but ``reversed_error`` is
          unavailable in stored files, the service falls back to unfiltered region
          selection with a warning.
        - ``skip_verify_by_reversing=True`` disables reverse verification during
          reintegration to avoid double tracing overhead.
        """
        if series_config.output_dir is None:
            raise ValueError("series_config.output_dir must be set")

        if reversed_error_threshold is not None and selection_mode != "region":
            error_msg = (
                "reversed_error_threshold is only supported for selection_mode='region'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        r_lc = np.asarray(r_lc_values, dtype=float)
        if r_lc.size == 0:
            raise ValueError("r_lc_values must contain at least one value")

        if template is None:
            # Default minimal template for reintegration data products.
            template = FieldLineTemplate.template_line_coordinates_only()

        lines_dir = Path(series_config.output_dir) / series_config.name / "lines"
        if not lines_dir.exists():
            raise FileNotFoundError(f"Lines directory not found: {lines_dir}")

        results: Dict[int, ReintegrationDataProductResult] = {}
        for experiment_number in experiment_numbers:
            exp_config_path = series_config.get_experiment_config_path(experiment_number)
            if not exp_config_path.exists():
                logger.warning(
                    "Skipping missing experiment config %03d: %s",
                    experiment_number,
                    exp_config_path,
                )
                continue

            data_products_dir = (
                Path(series_config.output_dir)
                / series_config.name
                / "data_products"
            )
            output_subdir = "OFLR_edges" if selection_mode == "edge" else "OFLR"
            output_dir = data_products_dir / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

            calc_result = self._calculate_footpoints_only(
                experiment_config_path=exp_config_path,
                lines_dir=lines_dir,
                r_lc_values=r_lc,
                h_lc_coeff=h_lc_coeff,
                selection_mode=selection_mode,
                reversed_error_threshold=reversed_error_threshold,
            )
            self._integrate_selected_footpoints(
                experiment_config_path=exp_config_path,
                selection_mode=selection_mode,
                output_dir=output_dir,
                r_lc_values=np.asarray(calc_result.R_LC, dtype=float),
                selected_footpoints=calc_result.edge_footpoints,
                template=template,
                r_max=r_max,
                skip_verify_by_reversing=skip_verify_by_reversing,
                n_cores=n_cores,
                dt_reint=dt_reint,
            )

            counts = [len(x) for x in calc_result.edge_footpoints]
            results[experiment_number] = ReintegrationDataProductResult(
                experiment_number=experiment_number,
                selection_mode=selection_mode,
                output_dir=output_dir,
                r_lc_values=np.asarray(calc_result.R_LC, dtype=float),
                selected_footpoint_counts=counts,
            )

        return results
