"""Workflow for extracting PA data filtered by theta tolerance."""

from typing import Optional, Union, Dict
from pathlib import Path
import numpy as np

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.grid.grid import FootpointGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.workflow.workflow_utils import (
    prepare_batch_subset,
    extract_line_properties,
    normalize_line_arrays,
)
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



def patch_PA(results):
    """Calculate basis vectors and PA for all points on all field lines.
    
    Args:
        results: FieldLineSet with r_line and pa_line properties
    
    Returns:
        FieldLineSet: Modified results with calculated PA stored in pa_line property
    """
    r_lines = results.get_property('r_line')
    
    if r_lines is None or len(r_lines) == 0:
        error_msg = "No field line data found in results"
        logger.error(error_msg)
        return results
    
    e1_list = []
    e2_list = []
    n_norm_list = []
    PA_calculated = []
    
    for line_idx in range(len(r_lines)):
        if line_idx >= len(r_lines):
            continue
        
        r_line = r_lines[line_idx]
        
        if r_line is None or len(r_line) == 0:
            # Append None/empty arrays to maintain list structure
            e1_list.append(None)
            e2_list.append(None)
            n_norm_list.append(None)
            PA_calculated.append(np.array([], dtype=np.float32))
            continue
                
        # Calculate along whole line
        r64 = r_line.T.astype(np.float64)  # Shape: (3, n_points)
        
        ds = 1.0  # uniform sampling
        r1 = np.gradient(r64, ds, axis=1)
        r2 = np.gradient(r1, ds, axis=1)
        
        # Tangent vector along whole line
        t_abs = np.linalg.norm(r1, axis=0, keepdims=True)
        t_abs = np.maximum(t_abs, 1e-12)
        t_norm = r1 / t_abs
        
        # Curvature normal along whole line
        proj = np.sum(r2 * t_norm, axis=0, keepdims=True) * t_norm
        n = r2 - proj
        n_abs = np.linalg.norm(n, axis=0, keepdims=True)
        n_abs = np.maximum(n_abs, 1e-12)
        n_norm = n / n_abs
        
        
        Omega = np.array([[0.0], [0.0], [1.0]])  # shape (3, 1)

        # Project Ω onto plane perpendicular to t
        Omega_dot_t = np.sum(Omega * t_norm, axis=0, keepdims=True)
        e1 = Omega - Omega_dot_t * t_norm

        e1_norm = np.linalg.norm(e1, axis=0, keepdims=True)
        e1_norm = np.maximum(e1_norm, 1e-12)   # avoid division by zero near alignment
        e1 /= e1_norm

        # Second vector: completes right-handed frame
        e2 = np.cross(t_norm.T, e1.T).T
        e2_norm = np.linalg.norm(e2, axis=0, keepdims=True)
        e2_norm = np.maximum(e2_norm, 1e-12)
        e2 /= e2_norm

        # Project curvature normal onto this basis
        nom = np.sum(n_norm * e1, axis=0)
        denom = np.sum(n_norm * e2, axis=0)
        
        PA = np.arctan2(nom, denom)
        PA = 90 - np.rad2deg(PA)
        PA = (PA + 90) % 180 -90
        unstable_points = (n_abs[0, :] <= 1.1e-12) | (t_abs[0, :] <= 1.1e-12) | (e1_norm[0, :] <= 1.1e-12) | (e2_norm[0, :] <= 1.1e-12)
        PA[unstable_points] = np.nan
        
        e1_list.append(e1)
        e2_list.append(e2)
        n_norm_list.append(n_norm)
        # Convert to float32 to match stored PA format
        PA = PA.astype(np.float32)
        PA_calculated.append(PA)
    
    # Store calculated PA in results
    results.set_property(pa_line=PA_calculated)
    
    return results, e1_list, e2_list, n_norm_list





def make_extract_PA(
    experiment_config: ExperimentConfig,
    R_LC_values: np.ndarray,
    theta_tolerance: float,
    grid_theta: np.ndarray,
    lines_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    trim_radius_coeff: float = 1.0,
    r_min: float = 1.0,
    interpolation_factor: float = 1.0,
    trim_type: str = "cylinder",
) -> Union[FieldLineSet, Dict[float, FieldLineSet]]:
    """Extract PA data filtered by theta tolerance from experiment data.
    
    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration containing field line data.
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
    interpolation_factor : float, optional
        Factor to increase number of points in r_line via interpolation.
        Default is 1.0 (no interpolation). Values > 1.0 increase point density.
        
    Returns
    -------
    Union[FieldLineSet, Dict[float, FieldLineSet]]
        If single R_LC: single FieldLineSet.
        If multiple R_LC: dict mapping R_LC values to FieldLineSet objects.
    """
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
            r_lc=r_lc,
            lines_dir=lines_dir,
            theta_tolerance=theta_tolerance,
            grid_theta=grid_theta,
            save=False,
            output_path=None,
            radius_range=(r_min, max_radius),
            interpolation_factor=interpolation_factor,
            trim_type=trim_type,
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
            filename = f"{experiment_config.name}_extract_PA_RLC_{r_lc:05.2f}.fits"
            filepath = output_dir / filename
            
            # Save FieldLineSet for this R_LC
            save_data(field_line_set, filepath, overwrite=True)
            
            logger.info(f"Saved filtered FieldLineSet for R_LC={r_lc:05.2f} to {filepath}")
    
    # Return structure: if single R_LC, return FieldLineSet; otherwise return dict
    if len(R_LC_values) == 1:
        return all_results[R_LC_values[0]]
    else:
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
    interpolation_factor: float = 1.0,
    trim_type: str = "cylinder",
) -> FieldLineSet:
    """Extract field line points filtered by theta tolerance and save filtered FieldLineSet.
    
    This processor filters points on field lines where abs(tangent_theta - theta_grid) < theta_tolerance
    for each grid point, and saves a filtered FieldLineSet with properties: r_line, pa_line, 
    tangent_theta, tangent_phi.
    
    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration containing field line data.
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
    interpolation_factor : float, optional
        Factor to increase number of points in r_line via interpolation.
        Default is 1.0 (no interpolation). Values > 1.0 increase point density.
    trim_type : str, optional
        Type of trimming to apply ("cylinder" or "sphere"). Default is "cylinder".
        
    Returns
    -------
    FieldLineSet
        Filtered FieldLineSet with properties r_line, pa_line, tangent_theta, tangent_phi.
    """
    # Extract radius range if provided
    min_radius = radius_range[0] if radius_range is not None else None
    max_radius = radius_range[1] if radius_range is not None else None
    
    # Process batches to filter points by theta tolerance
    def process_batch(batch_data):
        """Filter points by theta tolerance and create filtered FieldLineSet."""
        try:
            # Prepare batch subset (OFL extraction, reversed_error filtering, and trimming)
            batch_subset = prepare_batch_subset(batch_data, r_lc, trim_type=trim_type, max_radius=max_radius, interpolation_factor=interpolation_factor)
            
            if batch_subset is None:
                return FieldLineSet(_PA_EXTRACTION_TEMPLATE, copy_data=False)
            
            # Patch PA
            batch_subset, _, _, _ = patch_PA(batch_subset)
            
            # Extract properties
            tangent_theta_lines, tangent_phi_lines, r_line_data, pa_line_data = extract_line_properties(
                batch_subset, require_pa=True
            )
            
            # Filter points for each field line
            filtered_lines = _filter_field_line_points(
                batch_subset=batch_subset,
                tangent_theta_lines=tangent_theta_lines,
                tangent_phi_lines=tangent_phi_lines,
                r_line_data=r_line_data,
                pa_line_data=pa_line_data,
                grid_theta=grid_theta,
                theta_tolerance=theta_tolerance,
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
            filename = f"PA_data_{experiment_config.name}_RLC_{r_lc:05.1f}.fits"
            output_path = output_dir / filename
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a new FieldLineSet with save template to avoid warnings about missing properties
        save_result = FieldLineSet(_PA_SAVE_TEMPLATE, copy_data=False)
        save_result.append(filtered_result)
        
        save_data(save_result, output_path, overwrite=True)
        logger.info(f"Saved filtered FieldLineSet to {output_path}")
    
    return filtered_result




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
):
    """Filter points on field lines by theta tolerance."""
    filtered_lines = []
    
    for line_idx in range(batch_subset.n_field_lines):
        theta_line = tangent_theta_lines[line_idx]
        phi_line = tangent_phi_lines[line_idx]
        r_line = r_line_data[line_idx]
        pa_line = pa_line_data[line_idx] if pa_line_data is not None else None
        
        if theta_line is None or phi_line is None or r_line is None:
            continue
        
        # Normalize arrays
        result = normalize_line_arrays(theta_line, phi_line, r_line)
        if result is None:
            continue
        theta_line, phi_line, r_line, point_radii = result
        
        # Create mask
        theta_mask = _create_theta_mask(theta_line, grid_theta, theta_tolerance)
        
        # Use theta mask directly
        combined_mask = theta_mask
        
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

