"""Build OFL-based line data products from existing experiment line batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Literal

import numpy as np

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.open_field_line_edges import (
    OFLEdgeResult,
    calculate_and_save_OFL_edges,
    calculate_and_save_OFL_regions,
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

    def run(
        self,
        *,
        series_config: ExperimentSeriesConfig,
        experiment_numbers: Sequence[int],
        r_lc_values: Sequence[float],
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

            if selection_mode == "edge":
                calc_result: OFLEdgeResult = calculate_and_save_OFL_edges(
                    experiment_config_path=exp_config_path,
                    lines_dir=lines_dir,
                    template=template,
                    R_LC_values=r_lc,
                    output_dir=output_dir,
                    h_LC_coeff=h_lc_coeff,
                    trim_lines=trim_lines,
                    trim_type=trim_type,
                    trim_radius=trim_radius,
                    precise_edges=True,
                    reversed_error_threshold=None,
                )
            elif selection_mode == "region":
                calc_result = calculate_and_save_OFL_regions(
                    experiment_config_path=exp_config_path,
                    lines_dir=lines_dir,
                    template=template,
                    R_LC_values=r_lc,
                    output_dir=output_dir,
                    h_LC_coeff=h_lc_coeff,
                    trim_lines=trim_lines,
                    trim_type=trim_type,
                    trim_radius=trim_radius,
                    reversed_error_threshold=None,
                )
            else:
                raise ValueError(f"Unsupported selection_mode: {selection_mode}")

            counts = [len(x) for x in calc_result.edge_footpoints]
            results[experiment_number] = ReintegrationDataProductResult(
                experiment_number=experiment_number,
                selection_mode=selection_mode,
                output_dir=output_dir,
                r_lc_values=np.asarray(calc_result.R_LC, dtype=float),
                selected_footpoint_counts=counts,
            )

        return results
