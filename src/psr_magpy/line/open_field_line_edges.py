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
        grid.nside, ftp_ids_not_inner, nest=False
    )
    print(ftp_ids_not_inner)
    lon = grid.longitude
    colat = grid.colatitude

    import matplotlib.pyplot as plt
    #plt.figure()
    #plt.scatter(lon[ftp_ids_not_inner], colat[ftp_ids_not_inner], c=ftp_ids_not_inner)
    #plt.show()
    #exit(1)



    for iii in range(len(ftp_ids_not_inner)):
     plt.figure()
     plt.scatter(lon[ftp_ids_not_inner], colat[ftp_ids_not_inner], c=ftp_ids_not_inner)
     curpoint = ftp_ids_not_inner[iii]
     theta0, phi0 = hp.pix2ang(grid.nside, curpoint, nest=False)
     neighbours, w4 = hp.get_interp_weights(grid.nside, theta0, phi0, nest=False)
     print(curpoint)
     print(neighbours)
     plt.scatter(lon[curpoint], colat[curpoint], color='r')
     plt.scatter(lon[neighbours], colat[neighbours], color='cyan')
     #plt.scatter(lon[neighbours_astro], colat[neighbours_astro], color='cyan')
     plt.show()

        



    print(grid.nside)
    print(np.shape(neighbour_footpoints))
    
    # Vectorized neighbor checking
    # Mark valid neighbors (remove -1 values)
    neighbour_footpoints_valid = neighbour_footpoints > 0
    
    # Count valid neighbors per footpoint
    n_neighbours_per_footpoint = np.sum(neighbour_footpoints_valid, axis=0)
    
    import matplotlib.pyplot as plt
    
    plt.plot(n_neighbours_per_footpoint, 'o')
    
    # Check which neighbors are in ftp_ids
    neighbours_in_ftp_ids = np.isin(neighbour_footpoints, ftp_ids)
    print(neighbours_in_ftp_ids)
    
    # Count neighbors that are both valid and in ftp_ids
    n_neighbours_in_ftp_ids = np.sum(
        neighbours_in_ftp_ids & neighbour_footpoints_valid, axis=0
    )
    n_neighbours_in_ftp_ids = np.sum(
        neighbours_in_ftp_ids, axis=0
    )
    plt.plot(n_neighbours_in_ftp_ids)
    plt.show()
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
   
    if grid.nside < 512:
        for i, r_lc in enumerate(R_LC):
            cur_ftp_ids = lines.get_OFL_footpoints(
                r_lc, h_LC_coeff=h_LC_coeff
            )
            ftp_ids.append(cur_ftp_ids)
            inner_ftp_ids.append(cur_ftp_ids)
    else:
        for i, r_lc in enumerate(R_LC):
            cur_ftp_ids_outer = lines.get_OFL_footpoints(r_lc, h_LC_coeff=h_LC_coeff)
            if r_lc < 3:
                cf = 1.02
            else:
                cf = 1.08
            cur_ftp_ids_inner = lines.get_OFL_footpoints(r_lc*cf, h_LC_coeff=h_LC_coeff)
            cur_edge_ftp_ids = np.setdiff1d(cur_ftp_ids_outer, cur_ftp_ids_inner) 
            edge_ftp_ids.append(cur_edge_ftp_ids)


    if grid.nside < 512:
        
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
    h_LC_coeff: float = 3**1.5 / 2.0,
    trim_lines: bool = False,
    trim_type: str = "sphere",
    trim_radius: float = 5.0
) -> OFLEdgeResult:
    """
    Calculate OFL edges, filter edge lines according to template, and save as FITS file.
    
    This function performs a two-phase workflow to handle large datasets efficiently:
    
    Phase 1: Edge Calculation
    - Load batch files with minimal template (r_cyl_max, z_abs_max, footpoint_id only)
    - Calculate OFL edges for given R_LC values
    
    Phase 2: Edge Data Extraction
    - For each edge, reload batch files with full template
    - Subset each batch file by edge footpoint IDs
    - Combine subsetted batch files and save as edge files
    
    Args:
        experiment_config_path: Path to experiment configuration file
        lines_dir: Directory containing field line batch files
        template: FieldLine template specifying which properties to save
        R_LC_values: Array of light cylinder radii for edge calculation
        output_path: Path where to save the filtered edge lines FITS file
        h_LC_coeff: Height coefficient for OFL calculation
        trim_lines: Whether to trim field lines to light cylinder before saving
        trim_type: Trimming method ("sphere" or "cylinder")
        trim_radius: Radius for trimming (light cylinder radius)
        
    Returns:
        OFLEdgeResult: Result from edge calculation
        
    Raises:
        ValueError: If data loading fails or no edge footpoints found
        FileNotFoundError: If experiment config or lines directory not found
    """
    from psr_magpy.experiment.experiment_config import ExperimentConfig
    from psr_magpy.io import get_io_factory
    from psr_magpy.line.field_line_set import FieldLineSet
    from psr_magpy.line.field_line import FieldLine
    from psr_magpy.line.property_system import PropertyName
    import glob
    
    logger = get_logger(__name__, console_format_brief=True)
    
    # Load experiment configuration
    try:
        experiment_config = ExperimentConfig.load(experiment_config_path)
        logger.info(f"Loaded experiment config: {experiment_config.name}")
    except Exception as e:
        error_msg = f"Failed to load experiment config from {experiment_config_path}: {e}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
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
    
    # Phase 1: Load minimal data for edge calculation
    logger.info("Phase 1: Loading minimal data for edge calculation")
    try:
        # Create minimal template for edge calculation
        minimal_template = FieldLine(
            save_footpoint_id=True,  # Always required
            save_r_cyl_max=True,
            save_z_abs_max=True,
            # All other properties default to False
        )
        
        OFL_data = experiment_config.load_combined_batch_data(minimal_template, str(lines_dir))
        if OFL_data is None:
            error_msg = f"No field line data found in {lines_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Loaded {OFL_data.n_field_lines} field lines with minimal template")
    except Exception as e:
        error_msg = f"Failed to load field line data from {lines_dir}: {e}"
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
        import matplotlib.pyplot as plt
        #plt.scatter(grid.longitude, grid.colatitude, c=OFL_data.r_cyl_max)
        #plt.show()

        edge_result = calculator.calculate_edges(OFL_data, R_LC=R_LC_values)
        for jj in range(5):
            ftp_ids = edge_result.edge_footpoints[jj]
            plt.scatter(grid.longitude[ftp_ids], grid.colatitude[ftp_ids], color='r')
            plt.show()

        logger.info(f"Calculated edges for {len(edge_result.R_LC)} R_LC values")
    except Exception as e:
        error_msg = f"Failed to calculate OFL edges: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Phase 2: Extract edge data with full template
    logger.info("Phase 2: Extracting edge data with full template")
    try:
        # Find all batch files
        batch_pattern = f"{lines_dir}/{experiment_config.name}_batch_*.fits"
        batch_files = sorted(glob.glob(batch_pattern))
        
        if not batch_files:
            error_msg = f"No batch files found matching pattern: {batch_pattern}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Found {len(batch_files)} batch files")
        
        # Process each R_LC value and its edge footpoints
        output_path = Path(output_path)
        base_path = output_path.parent / output_path.stem
        extension = output_path.suffix
        
        for i, r_lc in enumerate(edge_result.R_LC):
            footpoint_indices = edge_result.edge_footpoints[i]
            logger.info(f"Processing R_LC = {r_lc:.2f}: {len(footpoint_indices)} edge footpoints")
            
            if len(footpoint_indices) == 0:
                logger.warning(f"  No edge footpoints for R_LC = {r_lc:.2f}")
                continue
            
            # Convert to set for efficient lookup
            edge_footpoint_set = set(footpoint_indices.tolist())
            
            # Load and subset each batch file
            combined_edge_data = None
            
            for batch_file in batch_files:
                logger.debug(f"  Processing batch file: {batch_file}")
                
                try:
                    # Load batch file with full template
                    batch_data = get_io_factory().load_data(FieldLineSet, batch_file, template=template)
                    
                    if batch_data is None or batch_data.n_field_lines == 0:
                        logger.debug(f"    No data in batch file, skipping")
                        continue
                    
                    # Get footpoint IDs from this batch
                    batch_footpoint_ids = batch_data.property_system.get_property(PropertyName.FOOTPOINT_ID)
                    
                    if batch_footpoint_ids is None:
                        logger.debug(f"    No footpoint IDs in batch file, skipping")
                        continue
                    
                    # Find footpoint IDs that are in the edge
                    edge_footpoint_ids_in_batch = []
                    for j, fp_id in enumerate(batch_footpoint_ids):
                        if fp_id in edge_footpoint_set:
                            edge_footpoint_ids_in_batch.append(fp_id)
                    
                    if not edge_footpoint_ids_in_batch:
                        logger.debug(f"    No edge footpoints in this batch, skipping")
                        continue
                    
                    # Create subset of this batch
                    batch_subset = batch_data.subset_by_footpoint_ids(edge_footpoint_ids_in_batch)
                    
                    if batch_subset.n_field_lines > 0:
                        logger.debug(f"    Found {batch_subset.n_field_lines} edge field lines in batch")
                        
                        # Combine with previous batch subsets
                        if combined_edge_data is None:
                            combined_edge_data = batch_subset
                        else:
                            combined_edge_data.append(batch_subset)
                    else:
                        logger.debug(f"    No edge field lines found in this batch (this is normal)")
                    
                except Exception as e:
                    logger.warning(f"    Error processing batch file {batch_file}: {e}")
                    continue
            
            # Save combined edge data for this R_LC value
            if combined_edge_data is not None and combined_edge_data.n_field_lines > 0:
                subset_filename = f"{base_path}_{i:03d}{extension}"
                
                logger.info(f"  Saving {combined_edge_data.n_field_lines} edge field lines to {subset_filename}")
                
                # Trim field lines if requested
                if trim_lines:
                    logger.info(f"  Trimming field lines to R_LC = {trim_radius} using {trim_type} method")
                    combined_edge_data.trim_to_light_cylinder(R_LC=trim_radius, trim_type=trim_type)
                    logger.info(f"  Trimmed edge data now has {combined_edge_data.n_field_lines} field lines")
                
                # Create I/O instance and add R_LC keyword to header
                from psr_magpy.io.data_io import FieldLineSetIO
                io_instance = FieldLineSetIO(combined_edge_data)
                io_instance.add_header_keyword('R_LC', r_lc, 'Light cylinder radius for this subset')
                
                # Save with the I/O instance that has the header keyword
                io_instance.save(combined_edge_data, subset_filename)
                
                logger.info(f"  Successfully saved edge data for R_LC = {r_lc:.2f}")
            else:
                logger.warning(f"  No edge field lines found for R_LC = {r_lc:.2f} across all batch files")
                logger.warning(f"  Expected {len(edge_footpoint_set)} footpoint IDs but found none")
        
        logger.info("Phase 2: Edge data extraction completed")
        
    except Exception as e:
        error_msg = f"Failed to extract and save edge data: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return edge_result


