"""Shared utilities for workflow processors."""

from typing import Optional, Tuple
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

from psr_magpy.experiment.grid_config import HealpixGridConfig, GridConfig
from psr_magpy.grid.grid import HealpixGrid, FootpointGrid
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError

logger = get_logger(__name__, console_format_brief=True)

# Common template for batch processing with basic properties
_BASIC_BATCH_TEMPLATE = FieldLineTemplate(
    save_footpoint_id=True,
    save_r_cyl_max=True,
    save_z_abs_max=True,
    save_tangent_theta=True,
    save_tangent_phi=True,
    save_r_line=True,
    save_reversed_error=True,
)


def create_healpix_grid(healpix_grid_config: HealpixGridConfig) -> HealpixGrid:
    """Create and validate HEALPix grid from config.
    
    Parameters
    ----------
    healpix_grid_config : HealpixGridConfig
        HEALPix grid configuration.
        
    Returns
    -------
    HealpixGrid
        Created and validated HEALPix grid.
        
    Raises
    ------
    PSRMagPyIOError
        If grid creation fails or grid is not HealpixGrid.
    """
    grid_config = GridConfig(healpix=healpix_grid_config)
    grid = grid_config.create_grid()
    
    if grid is None:
        raise PSRMagPyIOError("Failed to create grid from config")
    
    if not isinstance(grid, HealpixGrid):
        raise PSRMagPyIOError(f"Expected HealpixGrid, got {type(grid).__name__}")
    
    if grid.nline is None:
        raise PSRMagPyIOError("Grid nline is None")
    
    logger.info(f"Created HEALPix grid with nside={grid.nside}, npix={grid.nline}")
    return grid


def prepare_batch_subset(
    batch_data: FieldLineSet,
    r_lc: float,
    max_radius: Optional[float] = None,
    reversed_error_threshold: float = 0.01,
    interpolation_factor: float = 1.0,
) -> Optional[FieldLineSet]:
    """Extract OFL footpoints, filter by reversed_error, and trim by radius.
    
    Parameters
    ----------
    batch_data : FieldLineSet
        Batch data to process.
    r_lc : float
        Light cylinder radius for OFL extraction.
    max_radius : Optional[float], optional
        Maximum radius for trimming. If None, no trimming is applied.
    reversed_error_threshold : float, optional
        Threshold for reversed_error filtering. Default is 0.01.
    interpolation_factor : float, optional
        Factor to increase number of points in r_line via interpolation.
        Default is 1.0 (no interpolation). Values > 1.0 increase point density.
        
    Returns
    -------
    Optional[FieldLineSet]
        Filtered batch subset, or None if no valid data.
    """
    ofl_footpoint_ids = batch_data.get_OFL_footpoints(R_LC=r_lc)
    
    if len(ofl_footpoint_ids) == 0:
        logger.debug(f"No OFL footpoints found for R_LC={r_lc:.2f} in this batch")
        return None
    
    batch_subset = batch_data.subset_by_footpoint_ids(ofl_footpoint_ids)
    batch_subset = batch_subset.filter_by_reversed_error(reversed_error_threshold=reversed_error_threshold)
    
    # Interpolate r_line and recalculate tangent_theta and tangent_phi if factor > 1.0
    if interpolation_factor > 1.0:
        batch_subset = interpolate_and_recalculate_tangents(batch_subset, interpolation_factor=interpolation_factor)
    
    if batch_subset.n_field_lines == 0:
        logger.debug(f"No field lines remaining after filtering for R_LC={r_lc:.2f} in this batch")
        return None
    
    # Trim to maximum radius if specified
    if max_radius is not None:
        batch_subset.trim_to_light_cylinder(R_LC=max_radius, trim_type="cylinder")
    
    return batch_subset


def interpolate_and_recalculate_tangents(
    batch_subset: FieldLineSet,
    interpolation_factor: float = 2.0,
) -> FieldLineSet:
    """Interpolate r_line to increase points and recalculate tangent_theta and tangent_phi.
    
    Parameters
    ----------
    batch_subset : FieldLineSet
        Field line set to process.
    interpolation_factor : float, optional
        Factor to increase number of points. Default is 2.0 (doubles the points).
        
    Returns
    -------
    FieldLineSet
        Field line set with interpolated r_line and recalculated tangent_theta and tangent_phi.
    """
    from psr_magpy.utils.exceptions import PSRMagPyPropertyError
    
    r_line_data = batch_subset.get_property('r_line')
    if r_line_data is None:
        error_msg = "r_line property not available in batch_subset"
        logger.error(error_msg)
        raise PSRMagPyPropertyError(error_msg)
    
    n_lines = batch_subset.n_field_lines
    interpolated_r_lines = []
    interpolated_tangent_theta = []
    interpolated_tangent_phi = []
    
    for line_idx in range(n_lines):
        r_line = r_line_data[line_idx]
        
        if r_line is None or len(r_line) == 0:
            interpolated_r_lines.append(r_line)
            interpolated_tangent_theta.append(None)
            interpolated_tangent_phi.append(None)
            continue
        
        # Ensure r_line is (N, 3) format
        if r_line.ndim == 1:
            r_line = r_line.reshape(-1, 3)
        elif r_line.ndim == 2 and r_line.shape[1] != 3:
            if r_line.shape[0] == 3:
                r_line = r_line.T
            else:
                logger.warning(f"Unexpected r_line shape {r_line.shape} for line {line_idx}, skipping")
                interpolated_r_lines.append(r_line)
                interpolated_tangent_theta.append(None)
                interpolated_tangent_phi.append(None)
                continue
        
        n_points = len(r_line)
        if n_points < 2:
            # Can't interpolate with less than 2 points
            interpolated_r_lines.append(r_line)
            interpolated_tangent_theta.append(None)
            interpolated_tangent_phi.append(None)
            continue
        
        # Calculate arc length parameterization
        # Compute cumulative distance along the curve
        dr = np.diff(r_line, axis=0)
        ds = np.linalg.norm(dr, axis=1)
        s = np.concatenate([[0], np.cumsum(ds)])  # Arc length parameter
        
        # Create new arc length points with increased density
        n_new_points = int(n_points * interpolation_factor)
        s_new = np.linspace(s[0], s[-1], n_new_points)
        
        # Interpolate r_line along arc length
        r_line_interp = np.zeros((n_new_points, 3))
        for dim in range(3):
            interp_func = interp1d(s, r_line[:, dim], kind='linear', bounds_error=False, fill_value='extrapolate')
            r_line_interp[:, dim] = interp_func(s_new)
        
        interpolated_r_lines.append(r_line_interp)
        
        # Recalculate tangent_theta and tangent_phi using logic from property_calculator
        # property_calculator uses np.gradient(r, axis=1) where r is (3, N)
        # So we need to transpose r_line_interp to (3, N) for gradient, then transpose back
        r_line_T = r_line_interp.T  # (3, N)
        t = np.gradient(r_line_T, axis=1)  # (3, N) - gradient along second axis
        t_abs = np.linalg.norm(t, axis=0, keepdims=True)
        safe_t_abs = np.where(t_abs == 0.0, 1.0, t_abs)
        t_norm = t / safe_t_abs  # Normalized tangent vector (3, N)
        
        # Calculate tangent_theta (colatitude in degrees)
        x, y, z = t_norm
        r_sph = np.sqrt(x**2 + y**2 + z**2)
        theta = np.rad2deg(np.arccos(z / np.where(r_sph == 0, 1, r_sph)))
        interpolated_tangent_theta.append(theta.astype(np.float32))
        
        # Calculate tangent_phi (azimuth in degrees)
        phi = np.rad2deg(np.arctan2(y, x))
        interpolated_tangent_phi.append(phi.astype(np.float32))
    
    # Update batch_subset properties
    batch_subset.set_property(r_line=interpolated_r_lines)
    batch_subset.set_property(tangent_theta=interpolated_tangent_theta)
    batch_subset.set_property(tangent_phi=interpolated_tangent_phi)
    
    logger.info(f"Interpolated {n_lines} field lines with factor {interpolation_factor}")
    
    return batch_subset


def extract_line_properties(
    batch_subset: FieldLineSet,
    require_pa: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract tangent_theta, tangent_phi, r_line, and optionally pa_line from batch subset.
    
    Parameters
    ----------
    batch_subset : FieldLineSet
        Batch subset to extract properties from.
    require_pa : bool, optional
        Whether to require pa_line property. Default is False.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
        Tuple of (tangent_theta_lines, tangent_phi_lines, r_line_data, pa_line_data).
        pa_line_data is None if not available or not required.
        
    Raises
    ------
    PSRMagPyPropertyError
        If required properties are not available.
    """
    from psr_magpy.utils.exceptions import PSRMagPyPropertyError
    
    tangent_theta_lines = batch_subset.get_property('tangent_theta')
    tangent_phi_lines = batch_subset.get_property('tangent_phi')
    r_line_data = batch_subset.get_property('r_line')
    pa_line_data = batch_subset.get_property('pa_line') if require_pa else None
    
    if tangent_theta_lines is None or tangent_phi_lines is None or r_line_data is None:
        error_msg = "Required properties (tangent_theta, tangent_phi, or r_line) not available in batch_subset"
        logger.error(error_msg)
        raise PSRMagPyPropertyError(error_msg)
    
    return tangent_theta_lines, tangent_phi_lines, r_line_data, pa_line_data


def normalize_line_arrays(
    theta_line,
    phi_line,
    r_line,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Normalize arrays to 1D numpy arrays and calculate point radii.
    
    Parameters
    ----------
    theta_line : array-like
        Theta values for field line points.
    phi_line : array-like
        Phi values for field line points.
    r_line : array-like
        Position vectors for field line points (shape N or (N, 3)).
        
    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        Tuple of (theta_line, phi_line, r_line, point_radii) if valid, None otherwise.
    """
    # Ensure arrays are 1D numpy arrays
    theta_line = np.asarray(theta_line).flatten()
    phi_line = np.asarray(phi_line).flatten()
    r_line = np.asarray(r_line)
    
    # Handle scalar case
    if theta_line.ndim == 0:
        theta_line = np.array([theta_line])
    if phi_line.ndim == 0:
        phi_line = np.array([phi_line])
    
    if len(theta_line) == 0 or len(phi_line) == 0 or len(r_line) == 0:
        return None
    
    # Calculate radius for each point
    if r_line.ndim == 1:
        r_line = r_line.reshape(-1, 3)
    point_radii = np.sqrt(np.sum(r_line**2, axis=1))
    
    # Ensure point_radii matches length of theta_line and phi_line
    if len(point_radii) != len(theta_line) or len(point_radii) != len(phi_line):
        return None
    
    return theta_line, phi_line, r_line, point_radii


def create_radius_mask(
    point_radii: np.ndarray,
    min_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
) -> np.ndarray:
    """Create boolean mask for filtering points by radius range.
    
    Parameters
    ----------
    point_radii : np.ndarray
        Radius values for each point.
    min_radius : Optional[float], optional
        Minimum radius. If None, no lower bound.
    max_radius : Optional[float], optional
        Maximum radius. If None, no upper bound.
        
    Returns
    -------
    np.ndarray
        Boolean mask (True for points within range).
    """
    if min_radius is not None and max_radius is not None:
        return (point_radii >= min_radius) & (point_radii < max_radius)
    elif min_radius is not None:
        return point_radii >= min_radius
    elif max_radius is not None:
        return point_radii < max_radius
    else:
        return np.ones(len(point_radii), dtype=bool)


def get_basic_batch_template() -> FieldLineTemplate:
    """Get the basic batch processing template.
    
    Returns
    -------
    FieldLineTemplate
        Template with basic properties for batch processing.
    """
    return _BASIC_BATCH_TEMPLATE

