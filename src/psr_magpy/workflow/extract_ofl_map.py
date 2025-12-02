"""Workflow for OFL analysis from experiment series."""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import numpy as np

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError

logger = get_logger(__name__, console_format_brief=True)


def extract_ofl_map(
    series_config: ExperimentSeriesConfig,
    R_LC_values: np.ndarray,
    h_LC_coeff: float = 3**1.5 / 2.0,
    experiment_numbers: Optional[List[int]] = None,
    save: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[int, np.ndarray]:
    """Extract OFL maps from an experiment series.

    Loads OFL data from each experiment and creates OFL maps for each R_LC value.

    Parameters
    ----------
    series_config : ExperimentSeriesConfig
        Experiment series config.
    R_LC_values : np.ndarray
        Array of light cylinder radii for edge calculation.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    experiment_numbers : Optional[List[int]], optional
        Optional list of experiment numbers to process. Default is None.
    save : bool, optional
        Whether to save the OFL maps as FITS files. Default is True.
    output_dir : Optional[Union[str, Path]], optional
        Directory to save OFL maps. If None, uses lines_dir/OFL_maps. Default is None.

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping experiment numbers to OFL maps (arrays mapping footpoint IDs to R_LC values).
    """
    # Verify directories
    status = series_config.verify_directories(create_if_missing=False)
    if not status["lines"]["exists"]:
        raise PSRMagPyIOError("Lines directory must exist")

    # Get experiment numbers
    if experiment_numbers is None:
        experiments = series_config.create_experiment_series(save_configs=False)
        experiment_numbers = list(range(len(experiments)))

    results = {}
    lines_dir = series_config.get_lines_dir()
    
    # Determine output directory
    if output_dir is None:
        exp_output_dir = Path(lines_dir) / "OFL_maps"
    else:
        exp_output_dir = Path(output_dir)

    for exp_num in experiment_numbers:
        try:
            experiment = series_config.load_experiment_config(exp_num)
            logger.info(f"Processing OFL map for experiment {exp_num}: {experiment.name}")

            
            # Create OFL map for this experiment
            ofl_map = find_OFL_regions_from_batch_files(
                experiment_config=experiment,
                lines_dir=str(lines_dir),
                R_LC_values=R_LC_values,
                h_LC_coeff=h_LC_coeff,
                save=save,
                output_dir=exp_output_dir,
            )

            results[exp_num] = ofl_map

        except Exception as e:
            error_msg = f"Failed to extract OFL map for experiment {exp_num}: {e}"
            logger.error(error_msg)
            continue

    return results


def find_OFL_regions_from_batch_files(
    experiment_config: ExperimentConfig,
    lines_dir: Union[str, Path],
    R_LC_values: np.ndarray,
    h_LC_coeff: float = 3**1.5 / 2.0,
    save: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Load all batch files for a specific experiment and create OFL map for each R_LC.

    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration object.
    lines_dir : Union[str, Path]
        Directory containing batch files.
    R_LC_values : np.ndarray
        Array of light cylinder radii for edge calculation.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    save : bool, optional
        Whether to save the OFL map as FITS file. Default is True.
    output_dir : Optional[Union[str, Path]], optional
        Directory to save OFL maps. If None, uses lines_dir/OFL_maps. Default is None.

    Returns
    -------
    np.ndarray
        Array mapping footpoint IDs to R_LC values (NaN for non-OFL).
    """
    from astropy.io import fits
    
    # Create grid from experiment config
    grid = experiment_config.grid.create_grid()
    n_footpoints = grid.nline_ori
    
    # Create NaN array for all footpoint IDs
    ofl_map = np.full(n_footpoints, np.nan)
    
   
    # Use experiment's built-in batch file loader
    logger.info(f"Loading batch files for experiment {experiment_config.name}")
    
    # Load combined batch data using the experiment's method
    combined_data = experiment_config.load_combined_batch_data(
        FieldLineTemplate.template_OFL_subset(), str(lines_dir)
    )
    
    if combined_data is None or combined_data.n_field_lines == 0:
        raise ValueError(f"No valid field line data found for experiment {experiment_config.name}")
    
    logger.info(f"Loaded {combined_data.n_field_lines} field lines from batch files")
    
    # Process R_LC values in reverse order (descending)
    for r_lc in R_LC_values:
        logger.info(f"Finding OFL regions for R_LC = {r_lc:.2f}")
        
        # Get OFL footpoints using the existing method
        ofl_footpoint_ids = combined_data.get_OFL_footpoints(
            R_LC=r_lc,
            h_LC_coeff=h_LC_coeff
        )
        
        # Assign R_LC value to the OFL footpoint IDs
        ofl_map[ofl_footpoint_ids] = r_lc
        
        logger.info(f"  Found {len(ofl_footpoint_ids)} OFL footpoints for R_LC = {r_lc:.2f}")
    
    # Save OFL map if requested
    if save:
        if output_dir is None:
            output_dir = Path(lines_dir) / "OFLR_maps"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ofl_map_path = output_dir / f"OFLR_map_{experiment_config.name}.fits"
        
        # Save as FITS using astropy with two columns: OFL map and grid indices
        col1 = fits.Column(name='OFLR_MAP', format='E', array=ofl_map)
        col2 = fits.Column(name='GRID_INDICES', format='J', array=grid.indices)
        
        hdu = fits.BinTableHDU.from_columns([col1, col2])
        hdu.writeto(ofl_map_path, overwrite=True)
        
        logger.info(f"Saved OFLR map to {ofl_map_path}")
        logger.info(f"OFLR map shape: {ofl_map.shape}")
        logger.info(f"Non-NaN values: {np.sum(~np.isnan(ofl_map))}")
    
    return ofl_map

