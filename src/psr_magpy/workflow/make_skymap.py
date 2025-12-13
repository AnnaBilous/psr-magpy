"""Workflow for creating skymaps from experiment data."""

from typing import Optional, Union, Callable, Any, Dict, List
from pathlib import Path
import numpy as np
import healpy as hp

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.grid_config import HealpixGridConfig, GridConfig
from psr_magpy.grid.grid import HealpixGrid, FootpointGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError

logger = get_logger(__name__, console_format_brief=True)


def make_skymap(
    experiment_config: ExperimentConfig,
    healpix_grid_config: HealpixGridConfig,
    R_LC_values: np.ndarray,
    processor_fn: Optional[Callable[[ExperimentConfig, FootpointGrid, float, Optional[str], Optional[tuple]], Any]] = None,
    lines_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    trim_radius_coeff: Union[float, List[float]] = 0.0,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Create a skymap from experiment data using HEALPix grid.
    
    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration containing field line data.
    healpix_grid_config : HealpixGridConfig
        HEALPix grid configuration for the skymap.
    R_LC_values : np.ndarray
        Array of light cylinder radii to process.
    processor_fn : Optional[Callable], optional
        Function that processes data for each R_LC value.
        Signature: (experiment_config, grid, R_LC, lines_dir, radius_range) -> array of values (same size as grid).
        Should use experiment_config.process_batches_incremental() to process batches.
        If None, uses default skymap_processor. Default is None.
    lines_dir : Optional[Union[str, Path]], optional
        Directory containing batch files. If None, uses experiment's lines directory.
        Default is None.
    save : bool, optional
        Whether to save the skymap to file. Default is True.
    output_dir : Optional[Union[str, Path]], optional
        Directory to save skymap. If None, creates data_products/skymaps directory.
        Default is None.
    trim_radius_coeff : Union[float, List[float]], optional
        Coefficient for trimming field lines (actual trim_radius = trim_radius_coeff * r_lc).
        If float, single coefficient used for all R_LC values.
        If list, creates multiple skymaps for intervals [trim_radius_coeff[i]*r_lc, trim_radius_coeff[i+1]*r_lc).
        Uses max(R_LC_values) to calculate intervals. Default is 5.0.
        
    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
        If trim_radius is float: single 1D array of skymap values (same size as grid).
        If trim_radius is list: list of 1D arrays, one per interval.
    """
    # Create grid object
    grid_config = GridConfig(healpix=healpix_grid_config)
    grid = grid_config.create_grid()
    
    if grid is None:
        raise PSRMagPyIOError("Failed to create grid from config")
    
    if not isinstance(grid, HealpixGrid):
        raise PSRMagPyIOError(f"Expected HealpixGrid, got {type(grid).__name__}")
    
    # Get grid size
    if grid.nline is None:
        raise PSRMagPyIOError("Grid nline is None")
    
    grid_size = grid.nline
    logger.info(f"Created HEALPix grid with nside={grid.nside}, npix={grid_size}")
    
    # Handle trim_radius_coeff: convert to list of coefficient intervals
    # Actual trim_radius will be calculated per R_LC as trim_radius_coeff * r_lc
    max_r_lc = np.max(R_LC_values)
    
    if isinstance(trim_radius_coeff, (int, float)):
        coeff_intervals = [(0.0, float(trim_radius_coeff))]
    elif isinstance(trim_radius_coeff, list):
        if len(trim_radius_coeff) < 2:
            raise ValueError("trim_radius_coeff list must have at least 2 elements")
        coeff_intervals = [(trim_radius_coeff[i], trim_radius_coeff[i+1]) for i in range(len(trim_radius_coeff) - 1)]
    else:
        raise TypeError(f"trim_radius_coeff must be float or list, got {type(trim_radius_coeff)}")
    
    # Calculate example intervals using max R_LC for logging
    example_intervals = [(coeff_min * max_r_lc, coeff_max * max_r_lc) 
                         for coeff_min, coeff_max in coeff_intervals]
    logger.info(f"Processing {len(coeff_intervals)} radius interval(s) (example for max R_LC={max_r_lc:.2f}): {example_intervals}")
    
    # Default processor
    if processor_fn is None:
        if lines_dir is None:
            raise PSRMagPyIOError("lines_dir must be specified when using default processor")
        processor_fn = skymap_processor
    
    # Process each R_LC value separately and save individual files
    all_skymaps = {}  # Dict to store skymaps per R_LC: {r_lc: [skymap_per_interval]}
    
    for r_lc in R_LC_values:
        logger.info(f"Processing R_LC = {r_lc:.2f}")
        skymaps_for_r_lc = []
        
        # Process each radius interval for this R_LC
        for interval_idx, (coeff_min, coeff_max) in enumerate(coeff_intervals):
            # Calculate actual radius interval for this R_LC: trim_radius = trim_radius_coeff * r_lc
            min_radius = coeff_min * r_lc
            max_radius = coeff_max * r_lc
            
            logger.info(f"  Processing radius interval [{min_radius:.2f}, {max_radius:.2f})")
            
            # Execute processor with radius range
            result = processor_fn(experiment_config, grid, r_lc, lines_dir, (min_radius, max_radius))
            
            # Validate result size
            if result.shape[0] != grid_size:
                raise ValueError(
                    f"Processor result size {result.shape[0]} does not match grid size {grid_size}"
                )
            
            skymaps_for_r_lc.append(result)
        
        all_skymaps[r_lc] = skymaps_for_r_lc
    
    # Save if requested (one file per R_LC)
    if save:
        if output_dir is None:
            # Determine output directory from experiment config
            if lines_dir is None:
                raise PSRMagPyIOError("Either lines_dir or output_dir must be specified")
            output_dir = Path(lines_dir).parent / "data_products" / "skymaps"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each R_LC to its own file
        for r_lc, skymaps in all_skymaps.items():
            # Generate filename for this R_LC
            filename = f"{experiment_config.name}_skymap_nside{grid.nside}_RLC_{r_lc:.2f}.fits"
            filepath = output_dir / filename
            
            # Convert coefficient intervals to actual radius intervals for this R_LC
            radius_intervals = [(coeff_min * r_lc, coeff_max * r_lc)
                               for coeff_min, coeff_max in coeff_intervals]
            
            # Save skymaps for this R_LC (all intervals as columns in one file)
            _save_skymap_fits(skymaps, grid, filepath, experiment_config.name, radius_intervals, np.array([r_lc]), coeff_intervals)
            
            logger.info(f"Saved {len(skymaps)} skymap(s) for R_LC={r_lc:.2f} to {filepath}")
    
    # Return structure: if single R_LC and single interval, return array; otherwise return dict
    if len(R_LC_values) == 1 and len(coeff_intervals) == 1:
        return all_skymaps[R_LC_values[0]][0]
    elif len(coeff_intervals) == 1:
        # Single interval, multiple R_LC: return dict {r_lc: skymap}
        return {r_lc: skymaps[0] for r_lc, skymaps in all_skymaps.items()}
    else:
        # Multiple intervals: return dict {r_lc: [skymaps]}
        return all_skymaps


def _save_skymap_fits(
    skymaps: Union[np.ndarray, List[np.ndarray]],
    grid: HealpixGrid,
    filepath: Path,
    experiment_name: str,
    radius_intervals: List[tuple],
    R_LC_values: np.ndarray,
    coeff_intervals: List[tuple],
) -> None:
    """Save skymap(s) to FITS file.
    
    Parameters
    ----------
    skymaps : Union[np.ndarray, List[np.ndarray]]
        Single skymap array or list of skymap arrays (one per radius interval).
    grid : HealpixGrid
        HEALPix grid used for the skymap.
    filepath : Path
        Path to save the FITS file.
    experiment_name : str
        Name of the experiment.
    radius_intervals : List[tuple]
        List of (min_radius, max_radius) tuples for each skymap (example using max R_LC).
    R_LC_values : np.ndarray
        Array of R_LC values used to create the skymap.
    coeff_intervals : List[tuple]
        List of (coeff_min, coeff_max) tuples representing trim_radius_coeff intervals.
    """
    from astropy.io import fits
    
    # Convert single array to list for uniform handling
    if isinstance(skymaps, np.ndarray):
        skymaps = [skymaps]
    
    if len(skymaps) != len(radius_intervals):
        raise ValueError(f"Number of skymaps ({len(skymaps)}) must match number of intervals ({len(radius_intervals)})")
    
    # Create primary HDU with header
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['EXPERIMENT'] = experiment_name
    primary_hdu.header['NSIDE'] = grid.nside
    primary_hdu.header['NPIX'] = len(skymaps[0])
    primary_hdu.header['NINTERV'] = len(radius_intervals)
    primary_hdu.header['NRLC'] = len(R_LC_values)
    primary_hdu.header['COMMENT'] = 'HEALPix skymap with radius intervals'
    
    # Add R_LC values to header
    for i, r_lc in enumerate(R_LC_values):
        primary_hdu.header[f'RLC_{i}'] = (r_lc, f'Light cylinder radius {i}')
    
    # Add coefficient intervals to header
    for i, (coeff_min, coeff_max) in enumerate(coeff_intervals):
        primary_hdu.header[f'COEFF_MIN_{i}'] = (coeff_min, f'Minimum trim_radius_coeff for interval {i}')
        primary_hdu.header[f'COEFF_MAX_{i}'] = (coeff_max, f'Maximum trim_radius_coeff for interval {i}')
    
    # Add example radius intervals to header (using max R_LC)
    max_r_lc = np.max(R_LC_values)
    for i, (min_r, max_r) in enumerate(radius_intervals):
        primary_hdu.header[f'RMIN_{i}'] = (min_r, f'Example min radius for interval {i} (at max R_LC={max_r_lc:.2f})')
        primary_hdu.header[f'RMAX_{i}'] = (max_r, f'Example max radius for interval {i} (at max R_LC={max_r_lc:.2f})')
    
    # Create columns for each interval
    columns = []
    for i, (skymap, (min_r, max_r)) in enumerate(zip(skymaps, radius_intervals)):
        col_name = f'SKYMAP_R{min_r:.2f}_{max_r:.2f}' if len(skymaps) > 1 else 'SKYMAP'
        columns.append(fits.Column(name=col_name, format='D', array=skymap))
    
    # Create data HDU
    data_hdu = fits.BinTableHDU.from_columns(columns)
    
    # Write FITS file
    hdul = fits.HDUList([primary_hdu, data_hdu])
    hdul.writeto(filepath, overwrite=True)
    hdul.close()


def skymap_processor(
    experiment_config: ExperimentConfig,
    skymap_grid: FootpointGrid,
    r_lc: float,
    lines_dir: str,
    radius_range: Optional[tuple] = None,
) -> np.ndarray:
    """Process batches incrementally and map results to skymap grid.
    
    This processor uses process_batches_incremental to process all batch files,
    extracts data per footpoint, and maps it to the skymap grid pixels.
    
    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration containing field line data.
    skymap_grid : FootpointGrid
        HEALPix grid for the skymap.
    r_lc : float
        Light cylinder radius (currently not used, for future use).
    lines_dir : str
        Directory containing batch files.
    radius_range : Optional[tuple], optional
        Tuple (min_radius, max_radius) to filter points by radius.
        If None, no radius filtering is applied. Default is None.
        
    Returns
    -------
    np.ndarray
        1D array of skymap values (same size as skymap_grid).
    """
    if skymap_grid.nline is None:
        raise PSRMagPyIOError("Skymap grid nline is None")
    
    grid_size = skymap_grid.nline
    
    # Extract radius range if provided
    min_radius = radius_range[0] if radius_range is not None else None
    max_radius = radius_range[1] if radius_range is not None else None
    
    # Create minimal template for loading batch data
    template = FieldLineTemplate(
        save_footpoint_id=True,
        save_r_cyl_max=True,
        save_z_abs_max=True,
        save_tangent_theta=True,
        save_tangent_phi=True,
        save_r_line=True,
        save_reversed_error=True,
    )
    
    # Process batches to extract OFL footpoint IDs and count points in HEALPix bins
    def process_batch(batch_data):
        """Extract OFL regions, filter, and count points in HEALPix bins."""
        try:
            ofl_footpoint_ids = batch_data.get_OFL_footpoints(R_LC=r_lc)
            
            # Check if there are any OFL footpoints
            if len(ofl_footpoint_ids) == 0:
                logger.debug(f"No OFL footpoints found for R_LC={r_lc:.2f} in this batch")
                return np.zeros(grid_size, dtype=np.int64)
            
            batch_subset = batch_data.subset_by_footpoint_ids(ofl_footpoint_ids)
            batch_subset = batch_subset.filter_by_reversed_error(reversed_error_threshold=0.01)
            
            # Check if subset is empty after filtering
            if batch_subset.n_field_lines == 0:
                logger.debug(f"No field lines remaining after filtering for R_LC={r_lc:.2f} in this batch")
                return np.zeros(grid_size, dtype=np.int64)
            
            # Trim to maximum radius if specified
            if max_radius is not None:
                batch_subset.trim_to_light_cylinder(
                    R_LC=max_radius, trim_type="sphere"
                )
            
            # Extract tangent_theta, tangent_phi, and r_line for all points in all lines
            tangent_theta_lines = batch_subset.get_property('tangent_theta')
            tangent_phi_lines = batch_subset.get_property('tangent_phi')
            r_line_data = batch_subset.get_property('r_line')
            
            if tangent_theta_lines is None or tangent_phi_lines is None or r_line_data is None:
                logger.warning("Required properties not available in batch_subset")
                return np.zeros(grid_size, dtype=np.int64)
            
            # Count points in each HEALPix pixel
            pixel_counts = np.zeros(grid_size, dtype=np.int64)
            
            for theta_line, phi_line, r_line in zip(tangent_theta_lines, tangent_phi_lines, r_line_data):
                if theta_line is None or phi_line is None or r_line is None:
                    continue
                
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
                    continue
                
                # Calculate radius for each point
                # r_line is shape (N, 3)
                if r_line.ndim == 1:
                    r_line = r_line.reshape(-1, 3)
                point_radii = np.sqrt(np.sum(r_line**2, axis=1))
                
                # Ensure point_radii matches length of theta_line and phi_line
                if len(point_radii) != len(theta_line) or len(point_radii) != len(phi_line):
                    logger.warning(f"Length mismatch: r_line={len(point_radii)}, theta={len(theta_line)}, phi={len(phi_line)}")
                    continue
                
                # Filter by radius range if specified
                if min_radius is not None and max_radius is not None:
                    radius_mask = (point_radii >= min_radius) & (point_radii < max_radius)
                elif min_radius is not None:
                    radius_mask = point_radii >= min_radius
                elif max_radius is not None:
                    radius_mask = point_radii < max_radius
                else:
                    radius_mask = np.ones(len(point_radii), dtype=bool)
                
                if not np.any(radius_mask):
                    continue
                
                # Apply radius filter to angles and ensure 1D arrays
                theta_line_filtered = np.asarray(theta_line[radius_mask], dtype=np.float64).flatten()
                phi_line_filtered = np.asarray(phi_line[radius_mask], dtype=np.float64).flatten()
                
                # Ensure they are 1D arrays (not 0-d)
                if theta_line_filtered.ndim == 0:
                    theta_line_filtered = np.array([theta_line_filtered])
                if phi_line_filtered.ndim == 0:
                    phi_line_filtered = np.array([phi_line_filtered])
                
                # Convert theta from degrees to radians (colatitude)
                theta_rad = np.deg2rad(theta_line_filtered)
                # Convert phi from degrees to radians (azimuth)
                phi_rad = np.deg2rad(phi_line_filtered)
        
                # Filter out NaN/inf values before calling ang2pix
                valid_mask = np.isfinite(theta_rad) & np.isfinite(phi_rad) & (theta_rad >= 0) & (theta_rad <= np.pi)
                if not np.any(valid_mask):
                    continue
                
                theta_rad_valid = theta_rad[valid_mask]
                phi_rad_valid = phi_rad[valid_mask]
                
                # Get HEALPix pixel indices for all valid points
                pixel_indices = hp.ang2pix(skymap_grid.nside, theta_rad_valid, phi_rad_valid)
                
                # Count points in each pixel
                unique_pixels, counts = np.unique(pixel_indices, return_counts=True)
                # Ensure pixel indices are within valid range (should always be true, but defensive)
                valid_pixel_mask = (unique_pixels >= 0) & (unique_pixels < grid_size)
                valid_pixels = unique_pixels[valid_pixel_mask]
                valid_counts = counts[valid_pixel_mask]
                pixel_counts[valid_pixels] += valid_counts
            
            return pixel_counts
        except Exception as e:
            logger.warning(f"Error processing batch: {e}")
            return np.zeros(grid_size, dtype=np.int64)
    
    def accumulate_counts(acc: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
        """Accumulate pixel counts from batches."""
        if acc is None:
            return new.copy()
        return acc + new
    
    # Process all batches incrementally to count points in HEALPix bins
    skymap_counts = experiment_config.process_batches_incremental(
        template=template,
        lines_dir=lines_dir,
        processor_fn=process_batch,
        accumulator_fn=accumulate_counts,
        initial_value=None,
    )
    
    if skymap_counts is None:
        logger.warning(f"No data processed for R_LC = {r_lc:.2f}")
        return np.zeros(grid_size, dtype=np.float64)
    
    total_points = np.sum(skymap_counts)
    logger.info(f"Counted {total_points} points in HEALPix bins for R_LC = {r_lc:.2f}")
    
    # Convert counts to float64 for consistency with skymap format
    return skymap_counts.astype(np.float64)

