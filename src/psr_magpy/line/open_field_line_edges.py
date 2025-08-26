import healpy as hp
import numpy as np
from typing import NamedTuple, List, Union
from pathlib import Path
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.property_system import PropertyName
from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.utils.physical_conversions import get_R_LC, get_P_ms
from psr_magpy.utils.logging_config import get_logger


class OFLEdgeResult(NamedTuple):
    """Result structure for OFL edge calculation."""
    R_LC: np.ndarray
    edge_footpoints: List[np.ndarray]


class OFLEdgeCalculator:
    """Calculator for OFL edge detection on HEALPix grids."""
    
    def __init__(self, grid: HealpixGrid, h_LC_coeff: float = 3**1.5 / 2.0):
        """Initialize calculator.
        
        Args:
            grid: HealpixGrid object
            h_LC_coeff: Height coefficient for OFL calculation
        """
        if grid.grid_type is None or grid.grid_type.value != "healpix":
            raise ValueError("Only HEALPix grids are supported")
        
        self.grid = grid
        self.h_LC_coeff = h_LC_coeff
    
    def calculate_edges(self, lines: FieldLineSet, P_ms=None, R_LC=None) -> OFLEdgeResult:
        """Calculate OFL edges for multiple R_LC values.
        
        Args:
            lines: FieldLineSet with calculated field lines
            P_ms: Pulsar periods in ms (alternative to R_LC)
            R_LC: Light cylinder radii (alternative to P_ms)
            
        Returns:
            OFLEdgeResult with R_LC values and edge footpoint arrays
        """
        return healpix_grid_get_OFL_edges(self.grid, lines, P_ms, R_LC, self.h_LC_coeff)
    
    def calculate_single_edge(self, ftp_ids, ftp_ids_inner=None):
        """Find edge footpoints in OFL region.
        
        Args:
            ftp_ids: Footpoint IDs of OFL region
            ftp_ids_inner: Inner region footpoint IDs to exclude
            
        Returns:
            Edge footpoint IDs as numpy array
        """
        return healpix_grid_get_single_OFL_edge(self.grid, ftp_ids, ftp_ids_inner)


def healpix_grid_get_single_OFL_edge(grid, ftp_ids, ftp_ids_inner=None):
    """Find edge footpoints in OFL region.
    
    Args:
        grid: HealpixGrid object
        ftp_ids: Footpoint IDs of OFL region
        ftp_ids_inner: Inner region footpoint IDs to exclude
        
    Returns:
        Edge footpoint IDs as numpy array
    """
    assert (
        grid.grid_type.value == "healpix"
    ), "Finding OFL edge is currently supported only for healpix grid"
    
    if ftp_ids_inner is not None and len(ftp_ids_inner) > 0:
        ftp_ids_not_inner = np.setdiff1d(ftp_ids, ftp_ids_inner)
    else:
        ftp_ids_not_inner = np.copy(ftp_ids)
    n_ftp = len(ftp_ids_not_inner)
    
    # Handle empty input
    if n_ftp == 0:
        return np.array([], dtype=int)
    
    # Get all eight nearest neighbours for all footpoints at once
    neighbour_footpoints = hp.pixelfunc.get_all_neighbours(
        grid.nside, ftp_ids_not_inner
    )
    
    # Vectorized neighbor checking
    # Mark valid neighbors (remove -1 values)
    neighbour_footpoints_valid = neighbour_footpoints > 0
    
    # Count valid neighbors per footpoint
    n_neighbours_per_footpoint = np.sum(neighbour_footpoints_valid, axis=0)
    
    # Check which neighbors are in ftp_ids
    neighbours_in_ftp_ids = np.isin(neighbour_footpoints, ftp_ids)
    
    # Count neighbors that are both valid and in ftp_ids
    n_neighbours_in_ftp_ids = np.sum(
        neighbours_in_ftp_ids & neighbour_footpoints_valid, axis=0
    )
    
    # Edge detection: footpoint is edge if it has fewer neighbors in ftp_ids than expected
    is_edge = n_neighbours_in_ftp_ids < n_neighbours_per_footpoint - 1

    ftp_ids_edge = ftp_ids_not_inner[is_edge]
    return ftp_ids_edge


def healpix_grid_get_OFL_edges(grid: HealpixGrid, lines: FieldLineSet, P_ms=None, R_LC=None, h_LC_coeff=3**1.5 / 2.0):
    """Calculate OFL edges for multiple R_LC values.
    
    Args:
        grid: HealpixGrid object
        lines: FieldLineSet with calculated field lines
        P_ms: Pulsar periods in ms (alternative to R_LC)
        R_LC: Light cylinder radii (alternative to P_ms)
        h_LC_coeff: Height coefficient for OFL calculation
        
    Returns:
        OFLEdgeResult with R_LC values and edge footpoint arrays
    """
    
    # Validate input
    if P_ms is None and R_LC is None:
        raise ValueError("Either P_ms or R_LC must be provided")
    if P_ms is not None and R_LC is not None:
        raise ValueError("Provide either P_ms or R_LC, not both")
    
    # Convert to numpy arrays and validate ranges
    if P_ms is not None:
        P_ms = np.array([P_ms]) if not isinstance(P_ms, np.ndarray) else P_ms
        if np.any(P_ms <= 0):
            raise ValueError("P_ms must be positive")
        R_LC = get_R_LC(P_ms)
    else:
        R_LC = np.array([R_LC]) if not isinstance(R_LC, np.ndarray) else R_LC
        if np.any(R_LC < 1.0):
            raise ValueError("R_LC must be >= 1.0")
    
    R_LC = np.sort(R_LC)[::-1]
    
    n_values = len(R_LC)
    ftp_ids = []
    edge_ftp_ids = []
    # Initialize with empty array instead of empty list
    inner_ftp_ids = [np.array([], dtype=int)]
    
    for i, r_lc in enumerate(R_LC):
        cur_ftp_ids = lines.get_OFL_footpoints(
            r_lc, h_LC_coeff=h_LC_coeff
        )
        ftp_ids.append(cur_ftp_ids)
        inner_ftp_ids.append(cur_ftp_ids)
        
    for i in range(n_values):
        cur_edge_ftp_ids = healpix_grid_get_single_OFL_edge(
            grid, ftp_ids[i], ftp_ids_inner=inner_ftp_ids[i]
        )
        edge_ftp_ids.append(cur_edge_ftp_ids)
    return OFLEdgeResult(R_LC=R_LC, edge_footpoints=edge_ftp_ids)


def calculate_and_save_OFL_edges(
    experiment_config_path: Union[str, Path],
    lines_dir: Union[str, Path],
    template: FieldLine,
    R_LC_values: np.ndarray,
    output_path: Union[str, Path],
    h_LC_coeff: float = 3**1.5 / 2.0
) -> OFLEdgeResult:
    """
    Calculate OFL edges, filter edge lines according to template, and save as FITS file.
    
    This function performs the complete workflow:
    A) Calculate OFL edges for given R_LC values
    B) Filter edge lines according to the provided template
    C) Save filtered edge lines as FITS file
    
    Args:
        experiment_config_path: Path to experiment configuration file
        lines_dir: Directory containing field line batch files
        template: FieldLine template specifying which properties to save
        R_LC_values: Array of light cylinder radii for edge calculation
        output_path: Path where to save the filtered edge lines FITS file
        h_LC_coeff: Height coefficient for OFL calculation
        
    Returns:
        OFLEdgeResult: Result from edge calculation
        
    Raises:
        ValueError: If data loading fails or no edge footpoints found
        FileNotFoundError: If experiment config or lines directory not found
    """
    from psr_magpy.experiment.experiment_config import ExperimentConfig
    from psr_magpy.io import get_io_factory
    from psr_magpy.line.field_line_set import FieldLineSet
    
    logger = get_logger(__name__, console_format_brief=True)
    
    # Load experiment configuration
    try:
        experiment_config = ExperimentConfig.load(experiment_config_path)
        logger.info(f"Loaded experiment config: {experiment_config.name}")
    except Exception as e:
        error_msg = f"Failed to load experiment config from {experiment_config_path}: {e}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load field line data
    try:
        OFL_data = experiment_config.load_combined_batch_data(template, str(lines_dir))
        if OFL_data is None:
            error_msg = f"No field line data found in {lines_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Loaded {OFL_data.n_field_lines} field lines")
    except Exception as e:
        error_msg = f"Failed to load field line data from {lines_dir}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Create grid from experiment config
    try:
        grid = experiment_config.grid.create_grid()
        if grid is None:
            error_msg = "Failed to create grid from experiment config"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if grid.grid_type is None:
            error_msg = "Grid type is None"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Created grid: {grid.grid_type.value}")
    except Exception as e:
        error_msg = f"Failed to create grid: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Calculate OFL edges
    try:
        # Ensure we have a HealpixGrid for OFL edge calculation
        if not isinstance(grid, HealpixGrid):
            error_msg = f"OFL edge calculation requires HealpixGrid, got {type(grid).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        calculator = OFLEdgeCalculator(grid, h_LC_coeff=h_LC_coeff)
        edge_result = calculator.calculate_edges(OFL_data, R_LC=R_LC_values)
        logger.info(f"Calculated edges for {len(edge_result.R_LC)} R_LC values")
    except Exception as e:
        error_msg = f"Failed to calculate OFL edges: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Filter edge lines according to template - create subsets iteratively
    try:
        # Initialize combined data structures
        all_filtered_lines = []
        
        # Process each R_LC value and its edge footpoints
        for i, r_lc in enumerate(edge_result.R_LC):
            footpoint_indices = edge_result.edge_footpoints[i]
            logger.info(f"R_LC = {r_lc:.2f}: {len(footpoint_indices)} edge footpoints")
            
            if len(footpoint_indices) > 0:
                # Convert numpy array to list of integers
                footpoint_ids = footpoint_indices.tolist()
                
                # Create subset for this R_LC value
                subset = OFL_data.subset_by_footpoint_ids(footpoint_ids)
                
                if subset.n_field_lines > 0:
                    logger.info(f"  Created subset with {subset.n_field_lines} field lines")
                    all_filtered_lines.append(subset)
                else:
                    logger.warning(f"  No field lines found for R_LC = {r_lc:.2f}")
        
        # Save each subset to a separate file
        if all_filtered_lines:
            output_path = Path(output_path)
            base_path = output_path.parent / output_path.stem
            extension = output_path.suffix
            
            for i, subset in enumerate(all_filtered_lines):
                subset_filename = f"{base_path}_{i:03d}{extension}"
                
                logger.info(f"Saving subset {i+1}/{len(all_filtered_lines)} (R_LC = {edge_result.R_LC[i]:.2f}) to {subset_filename}")
                
                # Create I/O instance and add R_LC keyword to header
                from psr_magpy.io.data_io import FieldLineSetIO
                io_instance = FieldLineSetIO(subset)
                io_instance.add_header_keyword('R_LC', edge_result.R_LC[i], 'Light cylinder radius for this subset')
                
                # Save with the I/O instance that has the header keyword
                io_instance.save(subset, subset_filename)
            
            logger.info(f"Saved {len(all_filtered_lines)} subset files")
        else:
            logger.warning("No field lines found for any R_LC value")
    except Exception as e:
        error_msg = f"Failed to save filtered edge lines to {output_path}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return edge_result


