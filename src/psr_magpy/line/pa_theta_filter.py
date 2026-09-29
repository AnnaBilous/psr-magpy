"""Filter field-line points by tangent colatitude (observer θ)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt

from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.workflow.workflow_utils import (
    extract_line_properties,
    normalize_line_arrays,
)

logger = get_logger(__name__, console_format_brief=True)

ReferenceTheta = Union[float, npt.ArrayLike]


@dataclass
class FilteredPAData:
    """Flat PA samples after θ filtering (snapshot / overlay panels).

    Notes
    -----
    pa_values, phi_values
        Stacked samples from all lines passing the θ band.
    colatitude_values, longitude_values
        Footpoint colat/lon (degrees), one per sample; only if ``grid`` was given.
    r_line_stacked
        Emission-point positions ``(N, 3)``; only if ``r_line`` on the set.
    oflr_em_map
        Length-``grid.nline`` vector indexed by ``footpoint_id``; see
        Multipole ``PA_snapshot/OFLR_EM_MAP.md``. Non-NaN pixels are footpoints
        with ≥1 θ-filtered emission point; values are footpoint **colatitude**
        (degrees), not emission height. Historical name ``OFLR_em_map``.
    """

    pa_values: np.ndarray
    phi_values: np.ndarray
    colatitude_values: Optional[np.ndarray] = None
    longitude_values: Optional[np.ndarray] = None
    r_line_stacked: Optional[np.ndarray] = None
    oflr_em_map: Optional[np.ndarray] = None


def theta_point_mask(
    theta_line: npt.ArrayLike,
    reference_theta: ReferenceTheta,
    theta_tolerance: float,
) -> np.ndarray:
    """Boolean mask: keep points within ``theta_tolerance`` of reference θ.

    Parameters
    ----------
    theta_line
        Per-point tangent colatitude (degrees), shape ``(N,)``.
    reference_theta
        Scalar target (snapshot) or 1D grid (prepare / ``extract_PA``).
        For an array, a point is kept if it matches **any** reference value.
    theta_tolerance
        Half-width of the band in degrees.
    """
    theta_line = np.asarray(theta_line, dtype=np.float64).ravel()
    ref = np.atleast_1d(np.asarray(reference_theta, dtype=np.float64))
    if ref.size == 0:
        error_msg = "reference_theta must be non-empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    if ref.size == 1:
        return np.abs(theta_line - ref[0]) < theta_tolerance
    theta_diff = np.abs(theta_line[:, np.newaxis] - ref[np.newaxis, :])
    return np.any(theta_diff < theta_tolerance, axis=1)


def filter_fieldline_points_by_theta(
    field_line_set: FieldLineSet,
    reference_theta: ReferenceTheta,
    theta_tolerance: float = 0.2,
    *,
    grid: Optional[HealpixGrid] = None,
) -> FilteredPAData:
    """Collect θ-filtered PA, φ, and optional footpoint / position data.

    Parameters
    ----------
    field_line_set
        Lines with ``tangent_theta``, ``tangent_phi``, ``pa_line``.
    reference_theta
        Observer colatitude target(s) in degrees; see :func:`theta_point_mask`.
    theta_tolerance
        Band half-width in degrees.
    grid
        Full HEALPix grid for footpoint colat/lon and :attr:`FilteredPAData.oflr_em_map`.
    """
    tangent_theta = field_line_set.get_property("tangent_theta")
    tangent_phi = field_line_set.get_property("tangent_phi")
    pa_line = field_line_set.get_property("pa_line")
    r_line = field_line_set.get_property("r_line")
    footpoint_id = field_line_set.get_property("footpoint_id") if grid is not None else None

    if tangent_theta is None or tangent_phi is None or pa_line is None:
        error_msg = (
            "Required properties (tangent_theta, tangent_phi, pa_line) not found"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if grid is not None and footpoint_id is None:
        error_msg = "grid provided but footpoint_id property not found in FieldLineSet"
        logger.error(error_msg)
        raise ValueError(error_msg)

    pa_chunks: List[np.ndarray] = []
    phi_chunks: List[np.ndarray] = []
    colat_chunks: List[np.ndarray] = []
    lon_chunks: List[np.ndarray] = []
    r_chunks: List[np.ndarray] = []
    oflr_em_map = np.full(grid.nline, np.nan) if grid is not None else None

    grid_colat = grid_lon = None
    nline = 0
    if grid is not None:
        grid_colat = grid.colatitude
        grid_lon = grid.longitude
        nline = len(grid_colat)

    for i, (theta_line, phi_line, pa_line_data) in enumerate(
        zip(tangent_theta, tangent_phi, pa_line)
    ):
        if theta_line is None or phi_line is None or pa_line_data is None:
            continue

        theta_line = np.asarray(theta_line, dtype=np.float64).ravel()
        phi_line = np.asarray(phi_line, dtype=np.float64).ravel()
        pa_line_data = np.asarray(pa_line_data, dtype=np.float64).ravel()

        r_line_data = None
        if r_line is not None and i < len(r_line):
            r_line_data = r_line[i]
            if r_line_data is not None:
                r_line_data = np.asarray(r_line_data)
                if r_line_data.ndim == 1:
                    r_line_data = r_line_data.reshape(-1, 3)
                elif r_line_data.ndim == 2 and r_line_data.shape[1] != 3:
                    r_line_data = r_line_data.reshape(-1, 3)

        mask = theta_point_mask(theta_line, reference_theta, theta_tolerance)
        if not np.any(mask):
            continue

        filtered_pa = pa_line_data[mask]
        filtered_phi = phi_line[mask]
        pa_chunks.append(filtered_pa)
        phi_chunks.append(filtered_phi)

        if r_line is not None and r_line_data is not None:
            r_chunks.append(r_line_data[mask])

        if grid is not None and footpoint_id is not None:
            fp_id = footpoint_id[i]
            if fp_id is not None:
                fp_id = int(fp_id)
                if 0 <= fp_id < nline:
                    colat = grid_colat[fp_id]
                    lon = grid_lon[fp_id]
                    n_points = len(filtered_pa)
                    if n_points > 0:
                        colat_chunks.append(np.full(n_points, colat))
                        lon_chunks.append(np.full(n_points, lon))
                        oflr_em_map[fp_id] = colat

    pa_values = np.concatenate(pa_chunks) if pa_chunks else np.array([])
    phi_values = np.concatenate(phi_chunks) if phi_chunks else np.array([])
    colatitude_values = (
        np.concatenate(colat_chunks) if colat_chunks else None
    )
    longitude_values = np.concatenate(lon_chunks) if lon_chunks else None
    r_line_stacked = np.vstack(r_chunks) if r_chunks else None

    return FilteredPAData(
        pa_values=pa_values,
        phi_values=phi_values,
        colatitude_values=colatitude_values,
        longitude_values=longitude_values,
        r_line_stacked=r_line_stacked,
        oflr_em_map=oflr_em_map,
    )


def filter_field_line_set_by_theta(
    batch_subset: FieldLineSet,
    reference_theta: ReferenceTheta,
    theta_tolerance: float,
    save_template: FieldLineTemplate,
    *,
    require_pa: bool = True,
) -> List[FieldLine]:
    """θ-filter each line and return compact :class:`FieldLine` objects.

    Used by :mod:`~psr_magpy.workflow.extract_PA` (``reference_theta`` = observer grid).
    """
    tangent_theta_lines, tangent_phi_lines, r_line_data, pa_line_data = (
        extract_line_properties(batch_subset, require_pa=require_pa)
    )
    footpoint_ids = batch_subset.get_property("footpoint_id")
    filtered_lines: List[FieldLine] = []

    for line_idx in range(batch_subset.n_field_lines):
        theta_line = tangent_theta_lines[line_idx]
        phi_line = tangent_phi_lines[line_idx]
        r_line = r_line_data[line_idx]
        pa_line = pa_line_data[line_idx] if pa_line_data is not None else None

        if theta_line is None or phi_line is None or r_line is None:
            continue

        normalized = normalize_line_arrays(theta_line, phi_line, r_line)
        if normalized is None:
            continue
        theta_line, phi_line, r_line, _point_radii = normalized

        mask = theta_point_mask(theta_line, reference_theta, theta_tolerance)
        if not np.any(mask):
            continue

        filtered_line = FieldLine.from_template(save_template)
        filtered_line.set_property(
            footpoint_id=footpoint_ids[line_idx] if footpoint_ids is not None else None,
            r_line=r_line[mask],
            tangent_theta=theta_line[mask],
            tangent_phi=phi_line[mask],
        )
        if pa_line is not None:
            pa_arr = np.asarray(pa_line).ravel()
            filtered_line.set_property(pa_line=pa_arr[mask])
        filtered_lines.append(filtered_line)

    return filtered_lines
