"""Workflow for extracting PA data filtered by theta tolerance."""

from typing import Optional, Union, Dict
from pathlib import Path
import numpy as np

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.grid_config import HealpixGridConfig, GridConfig
from psr_magpy.grid.grid import HealpixGrid, FootpointGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.io import save_data
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError, PSRMagPyPropertyError

logger = get_logger(__name__, console_format_brief=True)

# Template for loading batch data with required properties
_PA_EXTRACTION_TEMPLATE = FieldLineTemplate(
    save_footpoint_id=True,
    save_r_line=True,
    save_pa_line=True,
    save_tangent_theta=True,
    save_tangent_phi=True,
    save_reversed_error=True,
)


def make_extract_PA(
    experiment_config: ExperimentConfig,
    healpix_grid_config: HealpixGridConfig,
    R_LC_values: np.ndarray,
    theta_tolerance: float,
    grid_theta: np.ndarray,
    lines_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    trim_radius_coeff: float = 1.0,
    r_min: float = 1.0,
) -> Union[FieldLineSet, Dict[float, FieldLineSet]]:
    """Extract PA data filtered by theta tolerance from experiment data.
    
    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration containing field line data.
    healpix_grid_config : HealpixGridConfig
        HEALPix grid configuration.
    R_LC_values : np.ndarray
        Array of light cylinder radii to process.
    theta_tolerance : float
        Tolerance in degrees for filtering points by tangent_theta.
    grid_theta : np.ndarray
        Grid theta values (colatitude in degrees) to filter against.
    lines_dir : Optional[Union[str, Path]], optional
        Directory containing batch files. If None, uses experiment's lines directory.
        Default is None.
    save : bool, optional
        Whether to save the filtered FieldLineSet to file. Default is True.
    output_dir : Optional[Union[str, Path]], optional
        Directory to save filtered FieldLineSet. If None, creates data_products/extracted_PA directory.
        Default is None.
    trim_radius_coeff : float, optional
        Coefficient for trimming field lines (actual trim_radius = trim_radius_coeff * r_lc).
        Default is 1.0.
    r_min : float, optional
        Minimum radius for filtering points. Default is 1.0.
        
    Returns
    -------
    Union[FieldLineSet, Dict[float, FieldLineSet]]
        If single R_LC: single FieldLineSet.
        If multiple R_LC: dict mapping R_LC values to FieldLineSet objects.
    """
    # Create grid object
    grid_config = GridConfig(healpix=healpix_grid_config)
    grid = grid_config.create_grid()
    
    if grid is None:
        raise PSRMagPyIOError("Failed to create grid from config")
    
    if not isinstance(grid, HealpixGrid):
        raise PSRMagPyIOError(f"Expected HealpixGrid, got {type(grid).__name__}")
    
    logger.info(f"Created HEALPix grid with nside={grid.nside}, npix={grid.nline}")
    
    if lines_dir is None:
        raise PSRMagPyIOError("lines_dir must be specified")
    
    # Process each R_LC value separately
    all_results = {}  # Dict to store FieldLineSets per R_LC: {r_lc: FieldLineSet}
    
    for r_lc in R_LC_values:
        logger.info(f"Processing R_LC = {r_lc:.2f}")
        
        # Calculate actual radius: trim_radius = trim_radius_coeff * r_lc
        max_radius = trim_radius_coeff * r_lc
        
        logger.info(f"  Processing with r_min = {r_min:.2f}, trim_radius = {max_radius:.2f}")
        
        # Execute processor with radius range (from r_min to max_radius), don't save yet
        result = extract_PA_processor(
            experiment_config=experiment_config,
            grid=grid,
            r_lc=r_lc,
            lines_dir=lines_dir,
            theta_tolerance=theta_tolerance,
            grid_theta=grid_theta,
            save=False,
            output_path=None,
            radius_range=(r_min, max_radius),
        )
        
        all_results[r_lc] = result
    
    # Save if requested (one file per R_LC)
    if save:
        if output_dir is None:
            output_dir = Path(lines_dir).parent / "data_products" / "extracted_PA"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each R_LC to its own file
        for r_lc, field_line_set in all_results.items():
            # Generate filename for this R_LC
            filename = f"{experiment_config.name}_extract_PA_nside{grid.nside}_RLC_{r_lc:.2f}.fits"
            filepath = output_dir / filename
            
            # Save FieldLineSet for this R_LC
            save_data(field_line_set, filepath, overwrite=True)
            
            logger.info(f"Saved filtered FieldLineSet for R_LC={r_lc:.2f} to {filepath}")
    
    # Return structure: if single R_LC, return FieldLineSet; otherwise return dict
    if len(R_LC_values) == 1:
        return all_results[R_LC_values[0]]
    else:
        return all_results


def extract_PA_processor(
    experiment_config: ExperimentConfig,
    grid: FootpointGrid,
    r_lc: float,
    lines_dir: str,
    theta_tolerance: float,
    grid_theta: np.ndarray,
    save: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    radius_range: Optional[tuple] = None,
) -> FieldLineSet:
    """Extract field line points filtered by theta tolerance and save filtered FieldLineSet.
    
    This processor filters points on field lines where abs(tangent_theta - theta_grid) < theta_tolerance
    for each grid point, and saves a filtered FieldLineSet with properties: r_line, pa_line, 
    tangent_theta, tangent_phi.
    
    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration containing field line data.
    grid : FootpointGrid
        Grid for filtering.
    r_lc : float
        Light cylinder radius.
    lines_dir : str
        Directory containing batch files.
    theta_tolerance : float
        Tolerance in degrees for filtering points by tangent_theta.
    grid_theta : np.ndarray
        Grid theta values (colatitude in degrees) to filter against.
    save : bool, optional
        Whether to save the filtered FieldLineSet. Default is True.
    output_path : Optional[Union[str, Path]], optional
        Path to save filtered FieldLineSet. If None and save=True, constructs path from lines_dir.
        Default is None.
    radius_range : Optional[tuple], optional
        Tuple (min_radius, max_radius) to filter points by radius.
        If None, no radius filtering is applied. Default is None.
        
    Returns
    -------
    FieldLineSet
        Filtered FieldLineSet with properties r_line, pa_line, tangent_theta, tangent_phi.
    """
    if grid.nline is None:
        raise PSRMagPyIOError("Grid nline is None")
    
    # Extract radius range if provided
    min_radius = radius_range[0] if radius_range is not None else None
    max_radius = radius_range[1] if radius_range is not None else None
    
    # Process batches to filter points by theta tolerance
    def process_batch(batch_data):
        """Filter points by theta tolerance and create filtered FieldLineSet."""
        try:
            ofl_footpoint_ids = batch_data.get_OFL_footpoints(R_LC=r_lc)
            
            if len(ofl_footpoint_ids) == 0:
                logger.debug(f"No OFL footpoints found for R_LC={r_lc:.2f} in this batch")
                return FieldLineSet(_PA_EXTRACTION_TEMPLATE, copy_data=False)
            
            batch_subset = batch_data.subset_by_footpoint_ids(ofl_footpoint_ids)
            batch_subset = batch_subset.filter_by_reversed_error(reversed_error_threshold=0.01)
            
            if batch_subset.n_field_lines == 0:
                logger.debug(f"No field lines remaining after filtering for R_LC={r_lc:.2f} in this batch")
                return FieldLineSet(_PA_EXTRACTION_TEMPLATE, copy_data=False)
            
            # Trim to maximum radius if specified
            if max_radius is not None:
                batch_subset.trim_to_light_cylinder(
                    R_LC=max_radius, trim_type="sphere"
                )
            
            # Extract properties
            tangent_theta_lines = batch_subset.get_property('tangent_theta')
            tangent_phi_lines = batch_subset.get_property('tangent_phi')
            r_line_data = batch_subset.get_property('r_line')
            pa_line_data = batch_subset.get_property('pa_line')
            
            if tangent_theta_lines is None or tangent_phi_lines is None or r_line_data is None:
                error_msg = "Required properties (tangent_theta, tangent_phi, or r_line) not available in batch_subset"
                logger.error(error_msg)
                raise PSRMagPyPropertyError(error_msg)
            
            # Filter points for each field line
            filtered_lines = _filter_field_line_points(
                batch_subset=batch_subset,
                tangent_theta_lines=tangent_theta_lines,
                tangent_phi_lines=tangent_phi_lines,
                r_line_data=r_line_data,
                pa_line_data=pa_line_data,
                grid_theta=grid_theta,
                theta_tolerance=theta_tolerance,
                min_radius=min_radius,
                max_radius=max_radius,
            )
            
            # Create FieldLineSet from filtered lines
            if len(filtered_lines) > 0:
                return FieldLineSet(filtered_lines, copy_data=True)
            else:
                return FieldLineSet(_PA_EXTRACTION_TEMPLATE, copy_data=False)
            
        except Exception as e:
            logger.warning(f"Error processing batch: {e}")
            return FieldLineSet(_PA_EXTRACTION_TEMPLATE, copy_data=False)
    
    def accumulate_field_lines(acc: Optional[FieldLineSet], new: FieldLineSet) -> FieldLineSet:
        """Accumulate filtered FieldLineSets from batches."""
        if acc is None:
            return new
        if new.n_field_lines == 0:
            return acc
        # Append new lines to accumulated set
        acc.append(new)
        return acc
    
    # Process all batches incrementally
    filtered_result = experiment_config.process_batches_incremental(
        template=_PA_EXTRACTION_TEMPLATE,
        lines_dir=lines_dir,
        processor_fn=process_batch,
        accumulator_fn=accumulate_field_lines,
        initial_value=None,
    )
    
    if filtered_result is None or filtered_result.n_field_lines == 0:
        logger.warning(f"No filtered data for R_LC = {r_lc:.2f}")
        return FieldLineSet(_PA_EXTRACTION_TEMPLATE, copy_data=False)
    
    logger.info(f"Extracted {filtered_result.n_field_lines} filtered field lines for R_LC = {r_lc:.2f}")
    
    # Save if requested
    if save:
        if output_path is None:
            # Construct output path from lines_dir
            output_dir = Path(lines_dir).parent / "data_products" / "extracted_PA"
            output_dir.mkdir(parents=True, exist_ok=True)
            # Get nside from grid if it's a HealpixGrid
            nside_str = f"nside{grid.nside}" if isinstance(grid, HealpixGrid) and hasattr(grid, 'nside') else ""
            filename = f"{experiment_config.name}_extract_PA_{nside_str}_RLC_{r_lc:.2f}.fits"
            output_path = output_dir / filename
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data(filtered_result, output_path, overwrite=True)
        logger.info(f"Saved filtered FieldLineSet to {output_path}")
    
    return filtered_result


def _normalize_arrays(theta_line, phi_line, r_line):
    """Normalize arrays to 1D numpy arrays, handling edge cases."""
    theta_line = np.asarray(theta_line).flatten()
    phi_line = np.asarray(phi_line).flatten()
    r_line = np.asarray(r_line)
    
    # Handle scalar case
    if theta_line.ndim == 0:
        theta_line = np.array([theta_line])
    if phi_line.ndim == 0:
        phi_line = np.array([phi_line])
    
    if len(theta_line) == 0 or len(phi_line) == 0 or len(r_line) == 0:
        return None, None, None
    
    # Calculate radius for each point
    if r_line.ndim == 1:
        r_line = r_line.reshape(-1, 3)
    point_radii = np.sqrt(np.sum(r_line**2, axis=1))
    
    # Ensure point_radii matches length of theta_line and phi_line
    if len(point_radii) != len(theta_line) or len(point_radii) != len(phi_line):
        return None, None, None
    
    return theta_line, phi_line, r_line, point_radii


def _create_radius_mask(point_radii, min_radius, max_radius):
    """Create mask for filtering points by radius range."""
    if min_radius is not None and max_radius is not None:
        return (point_radii >= min_radius) & (point_radii < max_radius)
    elif min_radius is not None:
        return point_radii >= min_radius
    elif max_radius is not None:
        return point_radii < max_radius
    else:
        return np.ones(len(point_radii), dtype=bool)


def _create_theta_mask(theta_line, grid_theta, theta_tolerance):
    """Create mask for filtering points by theta tolerance (vectorized)."""
    # Vectorized comparison: (N_points, 1) vs (1, N_grid) -> (N_points, N_grid)
    theta_diff = np.abs(theta_line[:, np.newaxis] - grid_theta[np.newaxis, :])
    # Check if any grid point matches for each field line point
    return np.any(theta_diff < theta_tolerance, axis=1)


def _filter_field_line_points(
    batch_subset,
    tangent_theta_lines,
    tangent_phi_lines,
    r_line_data,
    pa_line_data,
    grid_theta,
    theta_tolerance,
    min_radius,
    max_radius,
):
    """Filter points on field lines by theta tolerance and radius range."""
    filtered_lines = []
    
    for line_idx in range(batch_subset.n_field_lines):
        theta_line = tangent_theta_lines[line_idx]
        phi_line = tangent_phi_lines[line_idx]
        r_line = r_line_data[line_idx]
        pa_line = pa_line_data[line_idx] if pa_line_data is not None else None
        
        if theta_line is None or phi_line is None or r_line is None:
            continue
        
        # Normalize arrays
        result = _normalize_arrays(theta_line, phi_line, r_line)
        if result is None:
            continue
        theta_line, phi_line, r_line, point_radii = result
        
        # Create masks
        radius_mask = _create_radius_mask(point_radii, min_radius, max_radius)
        theta_mask = _create_theta_mask(theta_line, grid_theta, theta_tolerance)
        
        # Combine masks
        combined_mask = radius_mask & theta_mask
        
        if not np.any(combined_mask):
            continue
        
        # Apply filter
        filtered_r_line = r_line[combined_mask]
        filtered_theta_line = theta_line[combined_mask]
        filtered_phi_line = phi_line[combined_mask]
        filtered_pa_line = pa_line[combined_mask] if pa_line is not None else None
        
        # Create FieldLine object for this filtered line
        filtered_line = FieldLine.from_template(_PA_EXTRACTION_TEMPLATE)
        filtered_line.set_property(
            footpoint_id=batch_subset.get_property('footpoint_id')[line_idx],
            r_line=filtered_r_line,
            tangent_theta=filtered_theta_line,
            tangent_phi=filtered_phi_line,
        )
        if filtered_pa_line is not None:
            filtered_line.set_property(pa_line=filtered_pa_line)
        
        filtered_lines.append(filtered_line)
    
    return filtered_lines

