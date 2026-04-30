"""Build OFL-based line data products from existing experiment line batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Literal

import numpy as np

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.line.calculation_context import CalculationContext
from psr_magpy.line.calculator import FieldLineCalculator
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.open_field_line_edges import (
    OFLEdgeCalculator,
    OFLEdgeResult,
    _load_experiment_setup,
)
from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.utils.exceptions import PSRMagPyGridError, PSRMagPyExperimentError
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

    def _calculate_footpoints_only(
        self,
        *,
        experiment_config_path: Path,
        lines_dir: Path,
        r_lc_values: np.ndarray,
        h_lc_coeff: float,
        selection_mode: SelectionMode,
    ) -> OFLEdgeResult:
        """Compute OFL edge/region footpoint IDs only (no extraction/saving)."""
        _, grid, ofl_data, _ = _load_experiment_setup(
            experiment_config_path=experiment_config_path,
            lines_dir=lines_dir,
            load_template=None,
        )
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
                region_footpoints = [
                    ofl_data.get_OFL_footpoints(r_lc, h_LC_coeff=h_lc_coeff)
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
        r_max: Optional[float] = None,
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
        base_params = experiment_config.tracing.create_tracing_parameters()
        if r_max is not None:
            if base_params.integration.threshold_line_distance >= r_max:
                error_msg = (
                    "r_max must be greater than threshold_line_distance "
                    f"({base_params.integration.threshold_line_distance}), got {r_max}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            integration_params = base_params.integration.model_copy(
                update={"max_integration_radius": float(r_max)}
            )
            base_params = base_params.model_copy(update={"integration": integration_params})

        prefix = "OFLR_edge_lines" if selection_mode == "edge" else "OFLR_lines"
        experiment_number = None
        if experiment_config.metadata and "experiment_number" in experiment_config.metadata:
            experiment_number = int(experiment_config.metadata["experiment_number"])
        else:
            experiment_number = int(experiment_config.name.rsplit("_", 1)[1])

        for r_lc, footpoint_ids in zip(r_lc_values, selected_footpoints):
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
            output_params = base_params.output.model_copy(
                update={
                    "save_results_to_disk": True,
                    "keep_results_in_memory": False,
                    "filename": str(output_filename),
                    "overwrite": True,
                }
            )
            tracing_params = base_params.model_copy(update={"output": output_params})

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
        r_max: Optional[float] = None,
        selection_mode: SelectionMode = "edge",
        template: Optional[FieldLineTemplate] = None,
        h_lc_coeff: float = 3**1.5 / 2.0,
        trim_lines: bool = True,
        trim_type: str = "cylinder",
        trim_radius: float = 5.0,
    ) -> Dict[int, ReintegrationDataProductResult]:
        """Build OFL data products for selected experiments.

        Notes
        -----
        - This service does not persist any target experiment config.
        - Outputs are always written to data-products folders.
        - Reverse-error filtering is intentionally disabled (`reversed_error_threshold=None`).
        """
        if series_config.output_dir is None:
            raise ValueError("series_config.output_dir must be set")

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

            output_dir = (
                Path(series_config.output_dir)
                / series_config.name
                / "data_products"
                / exp_config_path.stem
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            calc_result = self._calculate_footpoints_only(
                experiment_config_path=exp_config_path,
                lines_dir=lines_dir,
                r_lc_values=r_lc,
                h_lc_coeff=h_lc_coeff,
                selection_mode=selection_mode,
            )
            self._integrate_selected_footpoints(
                experiment_config_path=exp_config_path,
                selection_mode=selection_mode,
                output_dir=output_dir,
                r_lc_values=np.asarray(calc_result.R_LC, dtype=float),
                selected_footpoints=calc_result.edge_footpoints,
                template=template,
                r_max=r_max,
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
