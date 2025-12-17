"""Workflow for creating skymaps from experiment data."""

from typing import Optional, Union, Callable, Any, Dict, List
from pathlib import Path
import numpy as np
import healpy as hp

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.grid_config import HealpixGridConfig
from psr_magpy.grid.grid import HealpixGrid, FootpointGrid
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.workflow.workflow_utils import (
    create_healpix_grid,
    prepare_batch_subset,
    extract_line_properties,
    normalize_line_arrays,
    create_radius_mask,
    get_basic_batch_template,
)
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError, PSRMagPyPropertyError

logger = get_logger(__name__, console_format_brief=True)


def make_skymap(
    experiment_config: ExperimentConfig,
    healpix_grid_config: HealpixGridConfig,
    R_LC_values: np.ndarray,
    processor_fn: Optional[Callable[[ExperimentConfig, FootpointGrid, float, Optional[str], Optional[tuple]], Any]] = None,
    lines_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    trim_radius_coeff: float = 1.0,
    r_min: float = 1.0,
    trim_type: str = "cylinder",
) -> Union[np.ndarray, Dict[float, np.ndarray]]:
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
    trim_radius_coeff : float, optional
        Coefficient for trimming field lines (actual trim_radius = trim_radius_coeff * r_lc).
        Default is 1.0.
    r_min : float, optional
        Minimum radius for filtering points. Default is 1.0.
    trim_type : str, optional
        Type of trimming to apply ("cylinder" or "sphere"). Default is "cylinder".
        
    Returns
    -------
    Union[np.ndarray, Dict[float, np.ndarray]]
        If single R_LC: single 1D array of skymap values (same size as grid).
        If multiple R_LC: dict mapping R_LC values to skymap arrays.
    """
    # Create grid object
    grid = create_healpix_grid(healpix_grid_config)
    grid_size = grid.nline
    
    # Default processor
    if processor_fn is None:
        if lines_dir is None:
            raise PSRMagPyIOError("lines_dir must be specified when using default processor")
        # Create wrapper to pass trim_type to default processor
        def processor_wrapper(exp_config, grd, r_lc_val, lns_dir, radius_rng):
            return skymap_processor(exp_config, grd, r_lc_val, lns_dir, radius_rng, trim_type=trim_type)
        processor_fn = processor_wrapper
    
    # Process each R_LC value separately and save individual files
    all_skymaps = {}  # Dict to store skymaps per R_LC: {r_lc: skymap}
    
    for r_lc in R_LC_values:
        logger.info(f"Processing R_LC = {r_lc:.2f}")
        
        # Calculate actual radius: trim_radius = trim_radius_coeff * r_lc
        max_radius = trim_radius_coeff * r_lc
        
        logger.info(f"  Processing with r_min = {r_min:.2f}, trim_radius = {max_radius:.2f}")
        
        # Execute processor with radius range (from r_min to max_radius)
        result = processor_fn(experiment_config, grid, r_lc, lines_dir, (r_min, max_radius))
        
        # Validate result size
        if result.shape[0] != grid_size:
            raise ValueError(
                f"Processor result size {result.shape[0]} does not match grid size {grid_size}"
            )
        
        all_skymaps[r_lc] = result
    
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
        for r_lc, skymap in all_skymaps.items():
            # Generate filename for this R_LC
            filename = f"skymap_{experiment_config.name}_nside{grid.nside:03d}_RLC_{r_lc:05.2f}.fits"
            filepath = output_dir / filename
            
            # Save skymap for this R_LC
            _save_skymap_fits(skymap, grid, filepath, experiment_config.name, np.array([r_lc]), trim_radius_coeff)
            
            logger.info(f"Saved skymap for R_LC={r_lc:05.2f} to {filepath}")    
    
    # Return structure: if single R_LC, return array; otherwise return dict
    if len(R_LC_values) == 1:
        return all_skymaps[R_LC_values[0]]
    else:
        return all_skymaps


def _save_skymap_fits(
    skymap: np.ndarray,
    grid: HealpixGrid,
    filepath: Path,
    experiment_name: str,
    R_LC_values: np.ndarray,
    trim_radius_coeff: float,
) -> None:
    """Save skymap to FITS file.
    
    Parameters
    ----------
    skymap : np.ndarray
        Skymap array.
    grid : HealpixGrid
        HEALPix grid used for the skymap.
    filepath : Path
        Path to save the FITS file.
    experiment_name : str
        Name of the experiment.
    R_LC_values : np.ndarray
        Array of R_LC values used to create the skymap.
    trim_radius_coeff : float
        Trim radius coefficient used.
    """
    from astropy.io import fits
    
    # Create primary HDU with header
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['EXPERIMENT'] = experiment_name
    primary_hdu.header['NSIDE'] = grid.nside
    primary_hdu.header['NPIX'] = len(skymap)
    primary_hdu.header['NRLC'] = len(R_LC_values)
    primary_hdu.header['TRIM_COEFF'] = (trim_radius_coeff, 'Trim radius coefficient')
    primary_hdu.header['COMMENT'] = 'HEALPix skymap'
    
    # Add R_LC values to header
    for i, r_lc in enumerate(R_LC_values):
        primary_hdu.header[f'RLC_{i}'] = (r_lc, f'Light cylinder radius {i}')
    
    # Create single column for skymap
    columns = [fits.Column(name='SKYMAP', format='D', array=skymap)]
    
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
    trim_type: str = "cylinder",
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
    trim_type : str, optional
        Type of trimming to apply ("cylinder" or "sphere"). Default is "cylinder".
        
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
    
    # Get template for loading batch data
    template = get_basic_batch_template()
    
    # Process batches to extract OFL footpoint IDs and count points in HEALPix bins
    def process_batch(batch_data):
        """Extract OFL regions, filter, and count points in HEALPix bins."""
        try:
            batch_subset = prepare_batch_subset(batch_data, r_lc, trim_type=trim_type, max_radius=max_radius)
            
            if batch_subset is None:
                return np.zeros(grid_size, dtype=np.int64)
            
            # Extract properties
            tangent_theta_lines, tangent_phi_lines, r_line_data, _ = extract_line_properties(
                batch_subset, require_pa=False
            )
            
            # Count points in each HEALPix pixel
            pixel_counts = np.zeros(grid_size, dtype=np.int64)
            
            for theta_line, phi_line, r_line in zip(tangent_theta_lines, tangent_phi_lines, r_line_data):
                if theta_line is None or phi_line is None or r_line is None:
                    continue
                
                # Normalize arrays
                result = normalize_line_arrays(theta_line, phi_line, r_line)
                if result is None:
                    continue
                theta_line, phi_line, r_line, point_radii = result
                
                
                # Ensure 1D arrays
                theta_line_filtered = np.asarray(theta_line, dtype=np.float64).flatten()
                phi_line_filtered = np.asarray(phi_line, dtype=np.float64).flatten()
                
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
        logger.warning(f"No data processed for R_LC = {r_lc:05.2f}")
        return np.zeros(grid_size, dtype=np.float64)
    
    total_points = np.sum(skymap_counts)
    logger.info(f"Counted {total_points} points in HEALPix bins for R_LC = {r_lc:05.1f}")
    
    # Convert counts to float64 for consistency with skymap format
    return skymap_counts.astype(np.float64)

