"""Shared utilities for workflow processors."""

from typing import Optional, Tuple
from pathlib import Path
import numpy as np

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
    
    if batch_subset.n_field_lines == 0:
        logger.debug(f"No field lines remaining after filtering for R_LC={r_lc:.2f} in this batch")
        return None
    
    # Trim to maximum radius if specified
    if max_radius is not None:
        batch_subset.trim_to_light_cylinder(R_LC=max_radius, trim_type="sphere")
    
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

