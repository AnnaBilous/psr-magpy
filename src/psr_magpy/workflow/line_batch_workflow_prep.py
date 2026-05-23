"""Shared batch load + property enrichment + trimming for line-based workflows.

Used by workflows such as :mod:`psr_magpy.workflow.make_skymap` for consistent
batch preparation from ``lines/`` or OFLR data products.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, FrozenSet, Iterator, Literal, Optional, Union

import numpy as np

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.services.batch_data_reader import BatchDataReader
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.property_calculator import PropertyCalculator
from psr_magpy.line.property_system import PropertyName
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyPropertyError
from psr_magpy.workflow.workflow_utils import get_basic_batch_template, prepare_batch_subset

logger = get_logger(__name__, console_format_brief=True)

LineDataSource = Literal["lines", "data_products_oflr"]


def get_workflow_batch_load_template() -> FieldLineTemplate:
    """Template for loading batch FITS with fields needed across workflows.

    Matches :func:`get_basic_batch_template` (tangents + ``r_line`` + OFL helpers).
    """
    return get_basic_batch_template()


def get_oflr_data_product_load_template() -> FieldLineTemplate:
    """Minimal template for OFLR reintegration FITS (typically ``r_line`` only on disk).

    After load, :func:`prepare_line_batch_after_load` enriches toward the workflow
    template (tangents from ``r_line``, etc.).
    """
    return FieldLineTemplate(save_r_line=True)


def merge_field_line_templates(*templates: FieldLineTemplate) -> FieldLineTemplate:
    """OR-merge save flags from several templates (``save_footpoint_id`` stays True)."""
    merged: dict[str, bool] = {}
    for t in templates:
        for key, val in t.get_flags_dict().items():
            merged[key] = merged.get(key, False) or bool(val)
    return FieldLineTemplate(**merged)


# Cannot be filled from ``r_line`` alone in :func:`enrich_field_line_set_from_r_line`.
_NON_APOSTERIORI: FrozenSet[PropertyName] = frozenset(
    {
        PropertyName.R_CYL_MAX,
        PropertyName.Z_ABS_MAX,
        PropertyName.REVERSED_ERROR,
    }
)

_RECON_FROM_R_LINE: FrozenSet[PropertyName] = frozenset(
    {
        PropertyName.TANGENT_THETA,
        PropertyName.TANGENT_PHI,
        PropertyName.PA_LINE,
        PropertyName.CURVATURE_LINE,
        PropertyName.B_LINE,
    }
)


def recommend_load_template_for_target(
    target_template: FieldLineTemplate,
) -> FieldLineTemplate:
    """Union of :func:`get_workflow_batch_load_template` and ``target_template``.

    Use this (or :func:`merge_field_line_templates`) when choosing a FITS load
    template so :func:`enrich_field_line_set_from_r_line` can ``set_property``
    for all along-line quantities requested by ``target_template``.
    """
    return merge_field_line_templates(get_workflow_batch_load_template(), target_template)


def experiment_number_from_config(experiment_config: ExperimentConfig) -> int:
    """Resolve 0-based experiment index from config metadata or name stem."""
    if experiment_config.metadata and "experiment_number" in experiment_config.metadata:
        return int(experiment_config.metadata["experiment_number"])
    return int(experiment_config.name.rsplit("_", 1)[1])


def resolve_lines_directory(parent_dir: Union[str, Path], series_name: str) -> Path:
    """Return ``<parent_dir>/<series_name>/lines``."""
    return Path(parent_dir).expanduser().resolve() / series_name / "lines"


def resolve_data_products_oflr_directory(
    parent_dir: Union[str, Path],
    series_name: str,
    *,
    data_products_subdir: str = "OFLR",
) -> Path:
    """Return ``<parent>/<series>/data_products/<subdir>`` for OFLR line batch FITS.

    Files use the same naming as reintegration trace output after spill/organize:
    ``OFLR_lines_{experiment_number:03d}_RLC_{r:.2f}_batch_*_out_of_*.fits`` (glob
    ``OFLR_lines_{nnn}_RLC_{r}_batch_*.fits`` via skymap workflow), or a single
    ``OFLR_lines_{nnn}_RLC_{r:.2f}.fits`` if nothing was spilled.
    """
    root = Path(parent_dir).expanduser().resolve()
    return root / series_name / "data_products" / data_products_subdir


def iter_batch_paths_lines(lines_dir: Union[str, Path], experiment_name: str) -> Iterator[Path]:
    """Yield sorted batch FITS paths under ``lines_dir`` for one experiment."""
    reader = BatchDataReader(experiment_name)
    yield from reader.iter_batch_files(str(lines_dir))


def _log_template_diff(
    reference_template: FieldLineTemplate, target_template: FieldLineTemplate
) -> None:
    refd = reference_template.get_flags_dict()
    tgtd = target_template.get_flags_dict()
    extra = sorted(k for k, v in tgtd.items() if v and not refd.get(k, False))
    omitted = sorted(k for k, v in refd.items() if v and not tgtd.get(k, False))
    if extra:
        logger.debug("target_template adds save flags vs reference: %s", extra)
    if omitted:
        logger.debug("target_template omits save flags vs reference: %s", omitted)


def _log_non_aposteriori_on_target(target_template: FieldLineTemplate) -> None:
    """Note scalars that are never synthesized from ``r_line`` (avoid noisy warnings)."""
    for prop in _NON_APOSTERIORI:
        if prop in target_template.property_system.get_enabled_properties():
            logger.debug(
                "%s is not synthesized from r_line here; must be present on disk if needed.",
                prop.key_name,
            )


def _expand_batch_save_flags_for_r_line_synthesis(
    batch: FieldLineSet, target_template: FieldLineTemplate
) -> None:
    """Turn on along-line save flags needed for :func:`enrich_field_line_set_from_r_line`.

    Reintegration FITS may be loaded with ``r_line`` only; tangents etc. are filled
    from ``r_line`` but :meth:`PropertySystem.set_property` requires those flags
    enabled on the batch.
    """
    if PropertyName.R_LINE not in batch.property_system.get_enabled_properties():
        return
    want = target_template.property_system.get_enabled_properties() & _RECON_FROM_R_LINE
    ps = batch.property_system
    for prop in want:
        if not ps.is_property_enabled(prop):
            setattr(ps.save_flags, prop.flag_name, True)
    ps.enabled_properties = ps.get_enabled_properties()


def _ensure_batch_enables_target_computable(
    batch: FieldLineSet, target_template: FieldLineTemplate
) -> None:
    """Require batch save flags for along-line targets that are not filled from ``r_line``."""
    want = target_template.property_system.get_enabled_properties() & _RECON_FROM_R_LINE
    have = batch.property_system.get_enabled_properties()
    missing = want - have
    if missing:
        names = ", ".join(p.key_name for p in sorted(missing, key=lambda p: p.key_name))
        error_msg = (
            f"batch is missing save flags for {names}. "
            "Load FITS with recommend_load_template_for_target(target_template) "
            "(or merge_field_line_templates) so these columns exist before enrichment."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _r_line_to_r_3N(r_line: Any) -> Optional[np.ndarray]:
    """Coerce stored ``r_line`` row to ``r`` with shape (3, N), ``N >= 2``, or None."""
    if r_line is None:
        return None
    arr = np.asarray(r_line, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            return None
        arr = arr.reshape(-1, 3)
    elif arr.ndim == 2:
        if arr.shape[1] == 3:
            pass
        elif arr.shape[0] == 3:
            arr = arr.T
        else:
            return None
    else:
        return None
    if arr.shape[0] < 2:
        return None
    return arr.T


def _line_R_max_from_r(r: np.ndarray) -> float:
    return float(np.max(np.sqrt(np.sum(r**2, axis=0))))


def _along_line_need_fill(
    existing: Any, n_points: int, *, min_points: int = 2
) -> bool:
    if n_points < min_points:
        return False
    if existing is None:
        return True
    try:
        ex = np.asarray(existing, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return True
    if ex.size == 0:
        return False
    if ex.size != n_points:
        return True
    if not np.all(np.isfinite(ex)):
        return True
    return False


def _get_line_along(prop_all: Any, i: int) -> Any:
    if prop_all is None:
        return None
    if isinstance(prop_all, list):
        return prop_all[i]
    # ndarray: first index is line (see FieldLineSet storage)
    return prop_all[i]


def _as_float32_1d(x: Any) -> Any:
    if x is None:
        return None
    return np.asarray(x, dtype=np.float32).reshape(-1)


def enrich_field_line_set_from_r_line(
    batch: FieldLineSet,
    target_template: FieldLineTemplate,
    *,
    threshold_line_distance: float = 0.0,
    field: Any = None,
    reference_template: Optional[FieldLineTemplate] = None,
) -> FieldLineSet:
    """Fill along-line quantities from ``r_line`` using :class:`PropertyCalculator`.

    Compares ``target_template`` to ``reference_template`` (default:
    :func:`get_workflow_batch_load_template`) for diagnostics only.

    Scalar diagnostics ``r_cyl_max``, ``z_abs_max``, and ``reversed_error`` are
    never synthesized here. If they are enabled on ``target_template``, a
    warning is logged; callers must load them from FITS when needed.

    Parameters
    ----------
    batch
        In-place update. If only ``r_line`` is loaded, along-line flags required by
        ``target_template`` are enabled automatically before enrichment.
    threshold_line_distance
        Passed to :class:`PropertyCalculator` (default ``0`` accepts any non-empty
        ``r_line`` for along-line reconstruction).
    field
        Required on the batch lines where ``save_b_line`` is requested on
        ``target_template`` and ``b_line`` needs filling.
    """
    ref = reference_template or get_workflow_batch_load_template()
    _log_template_diff(ref, target_template)
    _log_non_aposteriori_on_target(target_template)
    _expand_batch_save_flags_for_r_line_synthesis(batch, target_template)
    _ensure_batch_enables_target_computable(batch, target_template)

    r_lines = batch.get_property("r_line")
    if r_lines is None:
        error_msg = "r_line is required for enrich_field_line_set_from_r_line"
        logger.error(error_msg)
        raise PSRMagPyPropertyError(error_msg)

    want = target_template.property_system.get_enabled_properties()
    calc = PropertyCalculator(threshold_line_distance, field=field)
    fl = FieldLine.from_template(target_template)

    theta_all = batch.get_property("tangent_theta")
    phi_all = batch.get_property("tangent_phi")
    pa_all = batch.get_property("pa_line")
    curv_all = batch.get_property("curvature_line")
    b_all = batch.get_property("b_line")

    if PropertyName.B_LINE in want and field is None:
        for j in range(batch.n_field_lines):
            r_j = _r_line_to_r_3N(r_lines[j])
            npt = int(r_j.shape[1]) if r_j is not None else 0
            bl = _get_line_along(b_all, j)
            if npt >= 2 and _along_line_need_fill(bl, npt):
                logger.warning(
                    "save_b_line in target but field=None; b_line lines that need fill are unchanged"
                )
                break

    new_theta: Optional[list] = None
    new_phi: Optional[list] = None
    new_pa: Optional[list] = None
    new_curv: Optional[list] = None
    new_b: Optional[list] = None

    if PropertyName.TANGENT_THETA in want:
        new_theta = []
    if PropertyName.TANGENT_PHI in want:
        new_phi = []
    if PropertyName.PA_LINE in want:
        new_pa = []
    if PropertyName.CURVATURE_LINE in want:
        new_curv = []
    if PropertyName.B_LINE in want:
        new_b = []

    for i in range(batch.n_field_lines):
        r_line_i = r_lines[i]
        r = _r_line_to_r_3N(r_line_i)
        if r is None:
            if new_theta is not None:
                new_theta.append(_get_line_along(theta_all, i))
            if new_phi is not None:
                new_phi.append(_get_line_along(phi_all, i))
            if new_pa is not None:
                new_pa.append(_get_line_along(pa_all, i))
            if new_curv is not None:
                new_curv.append(_get_line_along(curv_all, i))
            if new_b is not None:
                new_b.append(_get_line_along(b_all, i))
            continue

        n_pts = r.shape[1]
        line_R_max = _line_R_max_from_r(r)

        th_i = _get_line_along(theta_all, i)
        ph_i = _get_line_along(phi_all, i)
        pa_i = _get_line_along(pa_all, i)
        cv_i = _get_line_along(curv_all, i)
        b_i = _get_line_along(b_all, i)

        need_th = PropertyName.TANGENT_THETA in want and _along_line_need_fill(th_i, n_pts)
        need_ph = PropertyName.TANGENT_PHI in want and _along_line_need_fill(ph_i, n_pts)
        tangent_norm = None
        if (need_th or need_ph) and line_R_max >= threshold_line_distance:
            t = np.gradient(r, axis=1)
            t_abs = np.linalg.norm(t, axis=0, keepdims=True)
            safe_t_abs = np.where(t_abs == 0.0, 1.0, t_abs)
            tangent_norm = t / safe_t_abs

        if new_theta is not None:
            if need_th:
                new_theta.append(
                    calc.prep_tangent_theta(r, line_R_max, fl, tangent_norm=tangent_norm)
                )
            else:
                new_theta.append(_as_float32_1d(th_i))

        if new_phi is not None:
            if need_ph:
                new_phi.append(
                    calc.prep_tangent_phi(r, line_R_max, fl, tangent_norm=tangent_norm)
                )
            else:
                new_phi.append(_as_float32_1d(ph_i))

        if new_pa is not None:
            if PropertyName.PA_LINE in want and _along_line_need_fill(pa_i, n_pts):
                new_pa.append(calc.prep_pa_line(r, line_R_max, fl))
            else:
                new_pa.append(_as_float32_1d(pa_i))

        if new_curv is not None:
            if PropertyName.CURVATURE_LINE in want and _along_line_need_fill(cv_i, n_pts):
                new_curv.append(calc.prep_curvature_line(r, 1.0, line_R_max, fl))
            else:
                new_curv.append(_as_float32_1d(cv_i))

        if new_b is not None:
            if (
                PropertyName.B_LINE in want
                and field is not None
                and _along_line_need_fill(b_i, n_pts)
            ):
                new_b.append(calc.prep_b_line(r, line_R_max, fl))
            else:
                new_b.append(_as_float32_1d(b_i))

    kwargs = {}
    if new_theta is not None:
        kwargs["tangent_theta"] = new_theta
    if new_phi is not None:
        kwargs["tangent_phi"] = new_phi
    if new_pa is not None:
        kwargs["pa_line"] = new_pa
    if new_curv is not None:
        kwargs["curvature_line"] = new_curv
    if new_b is not None:
        kwargs["b_line"] = new_b
    if kwargs:
        batch.set_property(**kwargs)
    return batch


# Along-line quantities kept in lockstep when trimming (match ``FieldLineSet.trim_to_light_cylinder``).
_RADIAL_SYNC_PROPERTIES: tuple[PropertyName, ...] = (
    PropertyName.R_LINE,
    PropertyName.PA_LINE,
    PropertyName.B_LINE,
    PropertyName.CURVATURE_LINE,
    PropertyName.TANGENT_THETA,
    PropertyName.TANGENT_PHI,
)


def _r_line_row_matrix(r_line: Any) -> Optional[np.ndarray]:
    """Return ``r_line`` as (N, 3) float array, or None."""
    arr = np.asarray(r_line, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            return None
        return arr.reshape(-1, 3)
    if arr.ndim == 2:
        if arr.shape[1] == 3:
            return arr
        if arr.shape[0] == 3:
            return arr.T
    return None


def _first_index_radius_ge_r_min(r_line: Any, r_min: float) -> int:
    """First point index with spherical radius ``>= r_min``; len if none qualify."""
    mat = _r_line_row_matrix(r_line)
    if mat is None or mat.shape[0] == 0:
        return 0
    radii = np.sqrt(np.sum(mat**2, axis=1))
    ok = radii >= float(r_min)
    if not np.any(ok):
        return int(mat.shape[0])
    return int(np.argmax(ok))


def _trim_remove_leading_inside_r_min(batch: FieldLineSet, r_min: float) -> None:
    """In-place: drop leading points with Euclidean ``||r|| < r_min`` on each line."""
    if PropertyName.R_LINE not in batch.property_system.get_enabled_properties():
        error_msg = "r_line must be enabled for r_min trimming"
        logger.error(error_msg)
        raise PSRMagPyPropertyError(error_msg)
    r_lines = batch.get_property("r_line")
    if r_lines is None:
        return

    starts: list[int] = []
    for i in range(batch.n_field_lines):
        starts.append(_first_index_radius_ge_r_min(r_lines[i], r_min))

    for prop in _RADIAL_SYNC_PROPERTIES:
        if prop not in batch.property_system.get_enabled_properties():
            continue
        prop_data = batch.get_property(prop.key_name)
        if prop_data is None:
            continue
        trimmed: list = []
        for i, line_data in enumerate(prop_data):
            st = starts[i]
            if line_data is None or len(line_data) == 0:
                trimmed.append(line_data)
                continue
            if prop == PropertyName.R_LINE:
                ld = np.asarray(line_data)
                if ld.ndim == 1:
                    npt = ld.size // 3
                    if st >= npt:
                        trimmed.append(np.array([]).reshape(0, 3))
                    else:
                        flat = ld.reshape(-1, 3)[st:, :].reshape(-1)
                        trimmed.append(flat.astype(ld.dtype, copy=False))
                else:
                    if st >= ld.shape[0]:
                        trimmed.append(np.array([]).reshape(0, 3))
                    else:
                        trimmed.append(ld[st:, :])
            else:
                ld = np.asarray(line_data)
                flat = ld.reshape(-1)
                if st >= flat.size:
                    trimmed.append(np.array([], dtype=ld.dtype))
                else:
                    trimmed.append(flat[st:].reshape(-1))
        batch.set_property(**{prop.key_name: trimmed})


def _maybe_filter_reversed_error(
    batch: FieldLineSet, reversed_error_threshold: float
) -> FieldLineSet:
    """Apply reversed_error filter if the property exists on this set."""
    if PropertyName.REVERSED_ERROR not in batch.property_system.get_enabled_properties():
        return batch
    try:
        return batch.filter_by_reversed_error(reversed_error_threshold)
    except PSRMagPyPropertyError:
        logger.debug("Skipping reversed_error filter (property missing or invalid on disk)")
        return batch


def prepare_line_batch_after_load(
    batch: FieldLineSet,
    *,
    source: LineDataSource,
    r_lc: float,
    trim_type: str,
    max_radius: Optional[float],
    r_min: Optional[float] = None,
    reversed_error_threshold: float = 0.01,
    target_template: Optional[FieldLineTemplate] = None,
    threshold_line_distance: float = 0.0,
    field: Any = None,
) -> Optional[FieldLineSet]:
    """Enrich along-line fields from ``r_line``, then subset / trim by source.

    Parameters
    ----------
    batch
        Loaded batch; use :func:`get_workflow_batch_load_template` for raw ``lines/``,
        :func:`get_oflr_data_product_load_template` for reintegrated OFLR products.
    target_template
        Properties to fill via :func:`enrich_field_line_set_from_r_line`. Defaults
        to the workflow load template.
    threshold_line_distance, field
        Forwarded to :func:`enrich_field_line_set_from_r_line`.
    source
        ``\"lines\"``: OFL subset + optional reversed filter + trim (existing
        :func:`prepare_batch_subset`). ``\"data_products_oflr\"``: footpoints
        already selected; optional reversed filter + radius trim only.
    r_lc
        Light-cylinder radius used for OFL subsetting in ``lines`` mode.
    trim_type
        Passed to :meth:`FieldLineSet.trim_to_light_cylinder` when ``max_radius``
        is not None (cylinder vs sphere outer boundary).
    max_radius
        Outer trim passed as ``R_LC`` to :meth:`FieldLineSet.trim_to_light_cylinder`
        after optional ``r_min`` handling (see below). None skips outer trim.
    r_min
        If set and ``> 0``, drop leading polyline points with spherical radius
        ``||r|| < r_min`` (in-place on each line, synced with tangents / optional
        along-line arrays). Applied after outer :meth:`trim_to_light_cylinder`
        when ``max_radius`` is used. None skips inner trim.
    reversed_error_threshold
        Used in ``lines`` mode inside :func:`prepare_batch_subset`; in
        ``data_products_oflr`` applied only if ``reversed_error`` is present.
    """
    tgt = target_template or get_workflow_batch_load_template()
    enrich_field_line_set_from_r_line(
        batch,
        tgt,
        threshold_line_distance=threshold_line_distance,
        field=field,
    )

    if source == "lines":
        out = prepare_batch_subset(
            batch,
            r_lc,
            trim_type=trim_type,
            max_radius=max_radius,
            reversed_error_threshold=reversed_error_threshold,
            r_min=r_min,
        )
        if out is None:
            return None
        if r_min is not None and r_min > 0:
            _trim_remove_leading_inside_r_min(out, r_min)
        return out

    if source == "data_products_oflr":
        working = _maybe_filter_reversed_error(batch, reversed_error_threshold)
        if working.n_field_lines == 0:
            return None
        if max_radius is not None:
            working.trim_to_light_cylinder(
                R_LC=max_radius, trim_type=trim_type, r_trim_min=r_min
            )
        if working.n_field_lines == 0:
            return None
        if r_min is not None and r_min > 0:
            _trim_remove_leading_inside_r_min(working, r_min)
        return working

    raise ValueError(f"Unknown LineDataSource: {source!r}")
