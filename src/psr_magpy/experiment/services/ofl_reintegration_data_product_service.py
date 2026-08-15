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
    downsample_edge_footpoint_vectors,
    jitter_footpoint_vectors as apply_footpoint_vector_jitter,
    _load_experiment_setup,
)
from psr_magpy.grid.grid import CustomGrid, HealpixGrid
from astropy.coordinates.representation import CartesianRepresentation
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

    def _build_r_max_lookup(
        self,
        r_max: Optional[Union[float, np.ndarray, Dict[float, float]]],
        r_lc_values: np.ndarray,
    ) -> Dict[float, Optional[float]]:
        """Map each ``R_LC`` to its integration cap (caller input order)."""
        resolved = self._resolve_r_max_values(
            r_max=r_max, r_lc_values=r_lc_values
        )
        return {float(r_lc): resolved[i] for i, r_lc in enumerate(r_lc_values)}

    def _align_r_max_to_r_lc(
        self,
        r_lc_values: np.ndarray,
        r_max_lookup: Dict[float, Optional[float]],
    ) -> List[Optional[float]]:
        """Return ``r_max`` values in the same order as ``r_lc_values``."""
        aligned: List[Optional[float]] = []
        for r_lc in r_lc_values:
            key = float(r_lc)
            if key in r_max_lookup:
                aligned.append(r_max_lookup[key])
                continue
            match = None
            for lookup_key, value in r_max_lookup.items():
                if np.isclose(lookup_key, key, rtol=0.0, atol=1e-6):
                    match = value
                    break
            if match is None:
                error_msg = f"r_max missing for R_LC={key:.2f}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            aligned.append(match)
        return aligned

    def _calculate_footpoints_only(
        self,
        *,
        experiment_config_path: Path,
        lines_dir: Path,
        r_lc_values: np.ndarray,
        h_lc_coeff: float,
        selection_mode: SelectionMode,
        reversed_error_threshold: Optional[float] = None,
        use_compound_grid: bool = False,
        iterative_edges: bool = True,
    ) -> OFLEdgeResult:
        """Compute OFL edge/region footpoint IDs only (no extraction/saving)."""
        batch_load_template = None
        if reversed_error_threshold is not None:
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
                selection_mode in ("region", "edge")
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
                    grid,
                    h_LC_coeff=h_lc_coeff,
                    precise_edges=True,
                    reversed_error_threshold=reversed_error_threshold,
                    use_compound_grid=use_compound_grid,
                    iterative_edges=iterative_edges,
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
        downsample_edges: bool = False,
        downsample_factor: int = 8,
        jitter_footpoint_vectors: bool = False,
        jitter_fraction: float = 1e-3,
    ) -> None:
        """Integrate selected footpoints and save line data products.

        When ``downsample_edges`` is True, each R_LC's selected (edge) footpoints
        are downsampled to one representative direction per coarse HEALPix pixel
        (``nside // downsample_factor``) and reintegrated from a :class:`CustomGrid`
        whose footpoint IDs are cardinal numbers (0..M-1).  Optional jitter may be
        applied after downsampling.
        """
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

        if downsample_edges and not isinstance(base_grid, HealpixGrid):
            error_msg = (
                "downsample_edges requires a HealpixGrid base grid, got "
                f"{type(base_grid).__name__} for {experiment_config.name}"
            )
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

            if downsample_edges:
                vectors = downsample_edge_footpoint_vectors(
                    np.asarray(footpoint_ids, dtype=int),
                    base_grid.nside,
                    downsample_factor=downsample_factor,
                )
                if jitter_footpoint_vectors:
                    vectors = apply_footpoint_vector_jitter(
                        vectors, jitter_fraction=jitter_fraction
                    )
                    logger.info(
                        "Reintegration R_LC=%.2f: jittered %d downsampled directions "
                        "(jitter_fraction=%.3e)",
                        r_lc,
                        vectors.shape[0],
                        jitter_fraction,
                    )
                coordinates = CartesianRepresentation(
                    vectors[:, 0], vectors[:, 1], vectors[:, 2]
                )
                subset_grid = CustomGrid(
                    nline_ori=int(vectors.shape[0]), coordinates=coordinates
                )
                logger.info(
                    "Reintegration R_LC=%.2f: downsampled %d edge footpoints to %d "
                    "CustomGrid directions",
                    r_lc,
                    len(footpoint_ids),
                    vectors.shape[0],
                )
            else:
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
                logger.info(
                    "Reintegration R_LC=%.2f: max_integration_radius=%.4f",
                    r_lc,
                    r_max_cur,
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
        use_compound_grid: bool = False,
        iterative_edges: bool = True,
        downsample_edges: bool = False,
        downsample_factor: int = 8,
        jitter_footpoint_vectors: bool = False,
        jitter_fraction: float = 1e-3,
        skip_verify_by_reversing: bool = True,
        n_cores: Optional[int] = None,
        dt_reint: Optional[float] = None,
        output_subdir: Optional[str] = None,
    ) -> Dict[int, ReintegrationDataProductResult]:
        """Build OFL data products for selected experiments.

        Notes
        -----
        - This service does not persist any target experiment config.
        - Outputs are always written to data-products folders.
        - ``output_subdir``, if set, overrides the default under ``data_products``
          (``OFLR_edges`` for edge mode, ``OFLR`` for region mode).
        - ``n_cores``, if set, overrides each experiment YAML for the integration step.
        - For ``selection_mode="region"``, ``reversed_error_threshold`` keeps lines with
          ``reversed_error <= threshold`` (NaNs dropped); requires batches saved with
          ``reversed_error``.
        - For ``selection_mode="edge"``, ``reversed_error_threshold`` makes a footpoint's
          neighbour count valid only for neighbours with ``reversed_error <= threshold``
          (NaN/untraced neighbours dropped); requires batches saved with ``reversed_error``.
        - If ``reversed_error_threshold`` is requested but ``reversed_error`` is
          unavailable in stored files, the service falls back to unfiltered
          selection with a warning.
        - For ``selection_mode="edge"``, ``use_compound_grid=True`` detects edges
          against a full-sphere grid built from ``grid.nside``, evaluating OFL
          membership over a compound (zero-filled outside traced) map.
        - For ``selection_mode="edge"``, ``iterative_edges`` (default True) controls
          whether edges are nested shells (each smaller R_LC excludes the previous
          larger region) or computed independently per R_LC.
        - For ``selection_mode="edge"``, ``downsample_edges=True`` reintegrates not the
          raw edge pixels but one representative direction per coarse HEALPix pixel
          (``nside // downsample_factor``), built as a :class:`CustomGrid` with
          cardinal footpoint IDs. Requires a HEALPix base grid.
        - For ``selection_mode="edge"`` with ``downsample_edges=True``,
          ``jitter_footpoint_vectors`` may apply small random offsets to downsampled
          directions. ``jitter_fraction`` scales the jitter limit relative to the
          median consecutive-vector spacing (default ``1e-3``).
        - ``skip_verify_by_reversing=True`` disables reverse verification during
          reintegration to avoid double tracing overhead.
        """
        if series_config.output_dir is None:
            raise ValueError("series_config.output_dir must be set")

        if reversed_error_threshold is not None and selection_mode not in (
            "region",
            "edge",
        ):
            error_msg = (
                "reversed_error_threshold is only supported for "
                "selection_mode='region' or 'edge'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if downsample_edges and selection_mode != "edge":
            error_msg = "downsample_edges is only supported for selection_mode='edge'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if jitter_footpoint_vectors and not downsample_edges:
            error_msg = (
                "jitter_footpoint_vectors requires downsample_edges=True"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        r_lc = np.asarray(r_lc_values, dtype=float)
        if r_lc.size == 0:
            raise ValueError("r_lc_values must contain at least one value")

        r_max_lookup = self._build_r_max_lookup(r_max=r_max, r_lc_values=r_lc)

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
            if output_subdir is None:
                resolved_subdir = "OFLR_edges" if selection_mode == "edge" else "OFLR"
            else:
                resolved_subdir = str(output_subdir)
            output_dir = data_products_dir / resolved_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

            calc_result = self._calculate_footpoints_only(
                experiment_config_path=exp_config_path,
                lines_dir=lines_dir,
                r_lc_values=r_lc,
                h_lc_coeff=h_lc_coeff,
                selection_mode=selection_mode,
                reversed_error_threshold=reversed_error_threshold,
                use_compound_grid=use_compound_grid,
                iterative_edges=iterative_edges,
            )
            calc_r_lc = np.asarray(calc_result.R_LC, dtype=float)
            r_max_aligned = self._align_r_max_to_r_lc(calc_r_lc, r_max_lookup)
            self._integrate_selected_footpoints(
                experiment_config_path=exp_config_path,
                selection_mode=selection_mode,
                output_dir=output_dir,
                r_lc_values=calc_r_lc,
                selected_footpoints=calc_result.edge_footpoints,
                template=template,
                r_max=np.asarray(r_max_aligned, dtype=float),
                skip_verify_by_reversing=skip_verify_by_reversing,
                n_cores=n_cores,
                dt_reint=dt_reint,
                downsample_edges=downsample_edges,
                downsample_factor=downsample_factor,
                jitter_footpoint_vectors=jitter_footpoint_vectors,
                jitter_fraction=jitter_fraction,
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
