import healpy as hp
import numpy as np
from typing import NamedTuple, List, Union, Optional, Dict
from pathlib import Path
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line import FieldLine
from psr_magpy.grid.grid import HealpixGrid, FootpointGrid
from psr_magpy.utils.physical_conversions import get_R_LC
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError, PSRMagPyGridError, PSRMagPyValidationError, PSRMagPyExperimentError


class OFLEdgeResult(NamedTuple):
    """OFL edge calculation result."""

    R_LC: np.ndarray
    edge_footpoints: List[np.ndarray]
    precise_edges: bool


class OFLEdgeCalculator:
    """OFL edge detection calculator."""

    def __init__(self, grid: FootpointGrid, h_LC_coeff: float = 3**1.5 / 2.0, precise_edges: bool = False):
        """Initialize calculator.

        Parameters
        ----------
        grid : FootpointGrid
            Grid object.
        h_LC_coeff : float, optional
            Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
        precise_edges : bool, optional
            If True, use precise edge calculation method. Default is False.
        """
        self.grid = grid
        self.h_LC_coeff = h_LC_coeff
        self.precise_edges = precise_edges

    def calculate_edges(
        self, lines: FieldLineSet, P_ms=None, R_LC=None, R_LC_edge_coefficient: Union[float, List[float]] = 1.02
    ) -> OFLEdgeResult:
        """Calculate OFL edges for multiple R_LC values.

        Parameters
        ----------
        lines : FieldLineSet
            FieldLineSet with calculated field lines.
        P_ms : optional
            Pulsar periods in ms (alternative to R_LC). Default is None.
        R_LC : optional
            Light cylinder radii (alternative to P_ms). Default is None.
        R_LC_edge_coefficient : Union[float, List[float]], optional
            Edge coefficient (float or list matching R_LC length). Default is 1.02.

        Returns
        -------
        OFLEdgeResult
            OFLEdgeResult with R_LC values and edge footpoint arrays.
        """
        if self.precise_edges:
            return get_OFL_edges_precise(self.grid, lines, P_ms, R_LC, self.h_LC_coeff)
        else:
            return get_OFL_edges_approximate(self.grid, lines, P_ms, R_LC, self.h_LC_coeff, R_LC_edge_coefficient)


def healpix_get_single_OFL_edge_precise(grid, ftp_ids, ftp_ids_inner=None):
    """Find edge footpoints in OFL region.

    Parameters
    ----------
    grid : FootpointGrid
        Grid object.
    ftp_ids : np.ndarray
        Footpoint IDs of OFL region.
    ftp_ids_inner : optional
        Inner region footpoint IDs to exclude. Default is None.

    Returns
    -------
    np.ndarray
        Edge footpoint IDs as numpy array.
    """
    assert (
        grid.grid_type.value == "healpix"
    ), "Finding OFL edge is currently supported only for healpix grid"

    if ftp_ids_inner is not None and len(ftp_ids_inner) > 0:
        ftp_ids_not_inner = np.setdiff1d(ftp_ids, ftp_ids_inner)
    else:
        ftp_ids_not_inner = np.copy(ftp_ids)
    n_ftp = len(ftp_ids_not_inner)
    #import matplotlib.pyplot as plt
    #full_grid = HealpixGrid(nside=256)
    #print("Grid nline", len(grid.longitude))
    #print(grid.indices)
    #print("Full grid nline", len(full_grid.longitude))

    #full_grid_ftp_ids_not_inner = grid.indices[ftp_ids_not_inner]
    #full_grid_ftp_ids = grid.indices[ftp_ids]
    
    full_grid_ftp_ids_not_inner = ftp_ids_not_inner
    full_grid_ftp_ids = ftp_ids

    #plt.plot(full_grid.longitude, full_grid.colatitude, '.')
    #plt.plot(full_grid.longitude[full_grid_ftp_ids_not_inner], full_grid.colatitude[full_grid_ftp_ids_not_inner], 'k.')
    #plt.show()

    # Handle empty input
    if n_ftp == 0:
        return np.array([], dtype=int)

    # Get all eight nearest neighbours for all footpoints at once
    neighbour_footpoints = hp.pixelfunc.get_all_neighbours(
        grid.nside, full_grid_ftp_ids_not_inner, nest=False
    )
    #l = grid.longitude
    #c = grid.colatitude
    #print(ftp_ids_not_inner[:5])
    #print(neighbour_footpoints[:5,:])
    #exit(1)
    # Vectorized neighbor checking
    # Mark valid neighbors (remove -1 values)
    neighbour_footpoints_valid = neighbour_footpoints >= 0

    # Count valid neighbors per footpoint
    n_neighbours_per_footpoint = np.sum(neighbour_footpoints_valid, axis=0)

    # Check which neighbors are in ftp_ids
    neighbours_in_ftp_ids = np.isin(neighbour_footpoints, full_grid_ftp_ids)

    # Count neighbors that are both valid and in ftp_ids
    n_neighbours_in_ftp_ids = np.sum(
        neighbours_in_ftp_ids & neighbour_footpoints_valid, axis=0
    )
    n_neighbours_in_ftp_ids = np.sum(neighbours_in_ftp_ids, axis=0)
    # Edge detection: footpoint is edge if it has fewer neighbors in ftp_ids than expected
    is_edge = n_neighbours_in_ftp_ids < n_neighbours_per_footpoint - 1
 
    ftp_ids_edge = ftp_ids_not_inner[is_edge]

    #plt.plot(grid.longitude, grid.colatitude, 'k.')
    #plt.plot(grid.longitude[ftp_ids_edge], grid.colatitude[ftp_ids_edge], 'r.')
    #plt.show()
    #exit(1)
    return ftp_ids_edge


def get_OFL_edges_precise(
    grid: FootpointGrid,
    lines: FieldLineSet,
    P_ms: Optional[Union[float, np.ndarray]] = None,
    R_LC: Optional[Union[float, np.ndarray]] = None,
    h_LC_coeff: float = 3**1.5 / 2.0,
):
    """Calculate OFL edges using precise method.

    Note: P_ms and R_LC are automatically sorted in descending order.

    Parameters
    ----------
    grid : FootpointGrid
        Grid object.
    lines : FieldLineSet
        FieldLineSet with calculated field lines.
    P_ms : Optional[Union[float, np.ndarray]], optional
        Pulsar periods in ms (alternative to R_LC). Default is None.
    R_LC : Optional[Union[float, np.ndarray]], optional
        Light cylinder radii (alternative to P_ms). Default is None.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.

    Returns
    -------
    OFLEdgeResult
        OFLEdgeResult with R_LC values and edge footpoint arrays.
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
        # Sort P_ms in decreasing order (corresponds to R_LC in decreasing order)
        P_ms = np.sort(P_ms)[::-1]
        R_LC = get_R_LC(P_ms)
    else:
        R_LC = np.array([R_LC]) if not isinstance(R_LC, np.ndarray) else R_LC
        if np.any(R_LC < 1.0):
            raise ValueError("R_LC must be >= 1.0")
        # Sort R_LC in decreasing order
        R_LC = np.sort(R_LC)[::-1]

    n_values = len(R_LC)
    ftp_ids = []
    edge_ftp_ids = []
    inner_ftp_ids = [np.array([], dtype=int)]

    for i, r_lc in enumerate(R_LC):
        cur_ftp_ids = lines.get_OFL_footpoints(r_lc, h_LC_coeff=h_LC_coeff)
        ftp_ids.append(cur_ftp_ids)
        inner_ftp_ids.append(cur_ftp_ids)
    #print(self.grid.indices)
    print(grid.indices)
    print(len(grid.indices))
    for i in range(n_values):
        cur_edge_ftp_ids = healpix_get_single_OFL_edge_precise(
            grid, ftp_ids[i], ftp_ids_inner=inner_ftp_ids[i]
        )
        edge_ftp_ids.append(cur_edge_ftp_ids)

    return OFLEdgeResult(R_LC=R_LC, edge_footpoints=edge_ftp_ids, precise_edges=True)


def get_OFL_edges_approximate(
    grid: FootpointGrid,
    lines: FieldLineSet,
    P_ms=None,
    R_LC=None,
    h_LC_coeff=3**1.5 / 2.0,
    R_LC_edge_coefficient: Union[float, List[float]] = 1.02,
):
    """Calculate OFL edges using approximate method.
    
    Edge field lines are defined as open lines for given R_LC, but not for
    R_LC * R_LC_edge_coefficient.

    Parameters
    ----------
    grid : FootpointGrid
        Grid object.
    lines : FieldLineSet
        FieldLineSet with calculated field lines.
    P_ms : optional
        Pulsar periods in ms (alternative to R_LC). Default is None.
    R_LC : optional
        Light cylinder radii (alternative to P_ms). Default is None.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    R_LC_edge_coefficient : Union[float, List[float]], optional
        Edge coefficient. Default is 1.02.

    Returns
    -------
    OFLEdgeResult
        OFLEdgeResult with R_LC values and edge footpoint arrays.
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
        # Sort P_ms in decreasing order (corresponds to R_LC in decreasing order)
        P_ms = np.sort(P_ms)[::-1]
        R_LC = get_R_LC(P_ms)
    else:
        R_LC = np.array([R_LC]) if not isinstance(R_LC, np.ndarray) else R_LC
        if np.any(R_LC < 1.0):
            raise ValueError("R_LC must be >= 1.0")
        # Sort R_LC in decreasing order
        R_LC = np.sort(R_LC)[::-1]

    n_values = len(R_LC)
    edge_ftp_ids = []
    
    # Handle R_LC_edge_coefficient as float or list
    if isinstance(R_LC_edge_coefficient, (int, float)):
        coefficients = [R_LC_edge_coefficient] * n_values
    else:
        if len(R_LC_edge_coefficient) != n_values:
            raise ValueError(f"R_LC_edge_coefficient list length ({len(R_LC_edge_coefficient)}) must match R_LC length ({n_values})")
        # If coefficients are provided as a list, we need to sort them to match the sorted R_LC
        # First, create pairs of (R_LC_original, coefficient) before sorting
        R_LC_original = R_LC.copy()  # Save original before sorting
        R_LC_coeff_pairs = list(zip(R_LC_original, R_LC_edge_coefficient))
        # Sort pairs by R_LC in descending order
        R_LC_coeff_pairs.sort(key=lambda x: x[0], reverse=True)
        # Extract sorted R_LC and corresponding coefficients
        R_LC = np.array([pair[0] for pair in R_LC_coeff_pairs])
        coefficients = np.array([pair[1] for pair in R_LC_coeff_pairs])
    
    for i, (r_lc, coeff) in enumerate(zip(R_LC, coefficients)):
        cur_ftp_ids_outer = lines.get_OFL_footpoints(r_lc, h_LC_coeff=h_LC_coeff)
        cur_ftp_ids_inner = lines.get_OFL_footpoints(
            r_lc * coeff, h_LC_coeff=h_LC_coeff
        )
        cur_edge_ftp_ids = np.setdiff1d(cur_ftp_ids_outer, cur_ftp_ids_inner)
        edge_ftp_ids.append(cur_edge_ftp_ids)

    return OFLEdgeResult(R_LC=R_LC, edge_footpoints=edge_ftp_ids, precise_edges=False)


def _load_experiment_setup(
    experiment_config_path: Union[str, Path],
    lines_dir: Union[str, Path],
) -> tuple:
    """Load experiment configuration, grid, and minimal OFL data.

    Parameters
    ----------
    experiment_config_path : Union[str, Path]
        Path to experiment configuration file.
    lines_dir : Union[str, Path]
        Directory containing field line batch files.

    Returns
    -------
    tuple
        (experiment_config, grid, OFL_data)
    """
    from psr_magpy.experiment.experiment_config import ExperimentConfig
    from psr_magpy.line.field_line import FieldLine

    logger = get_logger(__name__, console_format_brief=True)

    # Load experiment configuration
    try:
        experiment_config = ExperimentConfig.load(experiment_config_path)
        logger.info(f"Loaded experiment config: {experiment_config.name}")
    except Exception as e:
        raise PSRMagPyIOError(
            f"Failed to load experiment config from {experiment_config_path}: {e}"
        ) from e

    # Create grid from experiment config
    try:
        grid = experiment_config.grid.create_grid()
        if grid is None:
            raise PSRMagPyGridError("Failed to create grid from experiment config")
        if grid.grid_type is None:
            raise PSRMagPyGridError("Grid type is None")
        logger.info(f"Created grid: {grid.grid_type.value}")
    except Exception as e:
        raise PSRMagPyGridError(f"Failed to create grid: {e}") from e

    # Load minimal data for OFL calculation
    try:
        minimal_template = FieldLine(
            save_footpoint_id=True,
            save_r_cyl_max=True,
            save_z_abs_max=True,
        )

        OFL_data = experiment_config.load_combined_batch_data(
            minimal_template, str(lines_dir)
        )
        if OFL_data is None:
            raise PSRMagPyIOError(f"No field line data found in {lines_dir}")
        logger.info(
            f"Loaded {OFL_data.n_field_lines} field lines with minimal template"
        )
    except Exception as e:
        raise PSRMagPyIOError(f"Failed to load field line data from {lines_dir}: {e}") from e

    return experiment_config, grid, OFL_data


def _extract_and_save_OFL_data(
    experiment_config,
    template: FieldLine,
    lines_dir: Union[str, Path],
    batch_files: Optional[List[str]],
    footpoint_indices_list: List[np.ndarray],
    R_LC_values: np.ndarray,
    output_dir: Optional[Union[str, Path]] = None,
    trim_lines: bool = False,
    trim_type: str = "sphere",
    trim_radius: Union[float, np.ndarray, Dict[float, float]] = 5.0,
    max_memory_mb: float = 500.0,
    data_type: str = "edge",
    reversed_error_threshold: Optional[float] = None,
) -> None:
    """Extract and save OFL data (edges or regions) with memory management.
    
    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration.
    template : FieldLine
        FieldLine template specifying which properties to save.
    lines_dir : Union[str, Path]
        Directory containing batch files.
    batch_files : Optional[List[str]]
        List of batch file paths. If None, will be found automatically.
    footpoint_indices_list : List[np.ndarray]
        List of footpoint index arrays, one per R_LC value.
    R_LC_values : np.ndarray
        Array of R_LC values.
    output_dir : Union[str, Path]
        Output directory for saving chunks.
    trim_lines : bool, optional
        Whether to trim field lines. Default is False.
    trim_type : str, optional
        Trimming method. Default is "sphere".
    trim_radius : Union[float, np.ndarray, Dict[float, float]], optional
        Radius for trimming. Can be:
        - float: Single radius used for all R_LC values (default: 5.0)
        - np.ndarray: Array of radii, one per R_LC value (must match length)
        - Dict[float, float]: Dictionary mapping R_LC values to trim radii
    max_memory_mb : float, optional
        Maximum memory threshold in MB. Default is 500.0.
    data_type : str, optional
        Type of data ("edge" or "region") for logging. Default is "edge".
    reversed_error_threshold : Optional[float], optional
        Threshold for filtering by reversed_error. If None, no filtering is performed.
        Default is None.
    """
    from psr_magpy.io import load_data
    from psr_magpy.line.field_line_set import FieldLineSet
    from psr_magpy.line.property_system import PropertyName
    from psr_magpy.io.data_io import FieldLineSetIO
    import glob
    
    logger = get_logger(__name__, console_format_brief=True)
    
    # Ensure reversed_error is in template only if filtering is requested
    if reversed_error_threshold is not None:
        if not template.property_system.save_flags.save_reversed_error:
            template.property_system.save_flags.save_reversed_error = True
            logger.debug("Added reversed_error to template for filtering")
    
    # Find batch files if not provided
    if batch_files is None:
        batch_pattern = f"{lines_dir}/{experiment_config.name}_batch_*.fits"
        batch_files = sorted(glob.glob(batch_pattern))

        if not batch_files:
            raise PSRMagPyIOError(f"No batch files found matching pattern: {batch_pattern}")

        logger.info(f"Found {len(batch_files)} batch files")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
        
    # Extract experiment number from config
    experiment_number = None
    if experiment_config.metadata is not None and isinstance(experiment_config.metadata, dict) and "experiment_number" in experiment_config.metadata:
        experiment_number = experiment_config.metadata["experiment_number"]
    else:
        # Try to parse from experiment name (format: {series_name}_{number:03d})
        try:
            name_parts = experiment_config.name.rsplit("_", 1)
            if len(name_parts) == 2:
                experiment_number = int(name_parts[1])
        except (ValueError, IndexError):
            logger.warning(f"Could not extract experiment number from name: {experiment_config.name}")
    
    if experiment_number is None:
        error_msg = "Experiment number not found in config metadata or name"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Determine filename stem based on data type
    if data_type == "edge":
        stem_prefix = "OFLR_edge_lines"
    else:
        stem_prefix = "OFLR_lines"
    
    # Handle trim_radius as float, array, or dict
    n_values = len(R_LC_values)
    if isinstance(trim_radius, (int, float)):
        trim_radii = [trim_radius] * n_values
    elif isinstance(trim_radius, np.ndarray):
        if len(trim_radius) != n_values:
            raise ValueError(f"trim_radius array length ({len(trim_radius)}) must match R_LC_values length ({n_values})")
        trim_radii = trim_radius.tolist()
    elif isinstance(trim_radius, dict):
        # Map R_LC values to trim radii
        trim_radii = []
        for r_lc in R_LC_values:
            if r_lc not in trim_radius:
                error_msg = f"trim_radius dictionary missing key for R_LC = {r_lc:.2f}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            trim_radii.append(trim_radius[r_lc])
    else:
        raise TypeError(f"trim_radius must be float, np.ndarray, or dict, got {type(trim_radius)}")
    
    for i, (r_lc, footpoint_indices, trim_r) in enumerate(zip(R_LC_values, footpoint_indices_list, trim_radii)):
        logger.info(
            f"Processing R_LC = {r_lc:.2f}: {len(footpoint_indices)} {data_type} footpoints"
        )

        if len(footpoint_indices) == 0:
            logger.warning(f"  No {data_type} footpoints for R_LC = {r_lc:.2f}")
            continue

        footpoint_set = set(footpoint_indices.tolist())
        combined_data = None
        chunk_counter = 0

        for batch_file in batch_files:
            logger.debug(f"  Processing batch file: {batch_file}")

            try:
                batch_data = load_data(FieldLineSet, batch_file, template=template)

                if batch_data is None or batch_data.n_field_lines == 0:
                    logger.debug(f"    No data in batch file, skipping")
                    continue

                batch_footpoint_ids = batch_data.property_system.get_property(
                    PropertyName.FOOTPOINT_ID
                )

                if batch_footpoint_ids is None:
                    logger.debug(f"    No footpoint IDs in batch file, skipping")
                    continue

                # Filter by reversed_error if threshold is provided
                if reversed_error_threshold is not None:
                    batch_data = batch_data.filter_by_reversed_error(reversed_error_threshold)
                    if batch_data.n_field_lines == 0:
                        logger.debug(f"    No field lines passed reversed_error filter, skipping")
                        continue
                    # Update footpoint_ids after filtering
                    batch_footpoint_ids = batch_data.property_system.get_property(
                        PropertyName.FOOTPOINT_ID
                    )
                    if batch_footpoint_ids is None:
                        logger.debug(f"    No footpoint IDs after filtering, skipping")
                        continue

                # Find matching footpoint IDs
                matching_footpoint_ids = [
                    fp_id for fp_id in batch_footpoint_ids if fp_id in footpoint_set
                ]

                if not matching_footpoint_ids:
                    logger.debug(f"    No {data_type} footpoints in this batch, skipping")
                    continue

                batch_subset = batch_data.subset_by_footpoint_ids(matching_footpoint_ids)

                if batch_subset.n_field_lines > 0:
                    logger.debug(
                        f"    Found {batch_subset.n_field_lines} {data_type} field lines in batch"
                    )

                    # Trim before combining
                    if trim_lines:
                        batch_subset.trim_to_light_cylinder(
                            R_LC=trim_r, trim_type=trim_type
                        )
                    
                    # Combine
                    if combined_data is None:
                        combined_data = batch_subset
                    else:
                        combined_data.append(batch_subset)
                    
                    # Check memory and save chunk if needed
                    if combined_data is not None:
                        memory_mb = combined_data.estimate_size() / (1024 * 1024)
                        if memory_mb > max_memory_mb:
                            logger.info(
                                f"  Memory threshold exceeded ({memory_mb:.2f} MB > {max_memory_mb:.2f} MB). "
                                f"Saving chunk to disk."
                            )
                            
                            chunk_filename = output_dir / f"{stem_prefix}_{experiment_number:03d}_RLC_{r_lc:05.2f}_{chunk_counter:03d}.fits"
                            chunk_io = FieldLineSetIO(combined_data)
                            chunk_io.add_header_keyword(
                                "R_LC", r_lc, "Light cylinder radius for this subset"
                            )
                            chunk_io.save(combined_data, chunk_filename)
                            
                            logger.info(
                                f"  Saved chunk {chunk_counter} ({combined_data.n_field_lines} lines) "
                                f"to {chunk_filename}"
                            )
                            
                            chunk_counter += 1
                            combined_data = None
                else:
                    logger.debug(
                        f"    No {data_type} field lines found in this batch (this is normal)"
                    )

            except Exception as e:
                logger.warning(f"    Error processing batch file {batch_file}: {e}")
                continue

        # Save final chunk
        if combined_data is not None and combined_data.n_field_lines > 0:
            chunk_filename = output_dir / f"{stem_prefix}_{experiment_number:03d}_RLC_{r_lc:05.2f}_{chunk_counter:03d}.fits"

            logger.info(
                f"  Saving final chunk {chunk_counter} ({combined_data.n_field_lines} lines) "
                f"to {chunk_filename}"
            )

            io_instance = FieldLineSetIO(combined_data)
            io_instance.add_header_keyword(
                "R_LC", r_lc, "Light cylinder radius for this subset"
            )
            io_instance.save(combined_data, chunk_filename)

            logger.info(f"  Successfully saved final chunk for R_LC = {r_lc:.2f}")

        if chunk_counter > 0 or (combined_data is not None and combined_data.n_field_lines > 0):
            total_chunks = chunk_counter + (1 if combined_data is not None and combined_data.n_field_lines > 0 else 0)
            logger.info(f"  Saved {total_chunks} chunk(s) for R_LC = {r_lc:.2f}")
        else:
            logger.warning(
                f"  No {data_type} field lines found for R_LC = {r_lc:.2f} across all batch files"
            )
            logger.warning(
                f"  Expected {len(footpoint_set)} footpoint IDs but found none"
            )


def calculate_and_save_OFL_edges(
    experiment_config_path: Union[str, Path],
    lines_dir: Union[str, Path],
    template: FieldLine,
    R_LC_values: np.ndarray,
    output_dir: Optional[Union[str, Path]] = None,
    h_LC_coeff: float = 3**1.5 / 2.0,
    trim_lines: bool = False,
    trim_type: str = "sphere",
    trim_radius: Union[float, np.ndarray, Dict[float, float]] = 5.0,
    precise_edges: bool = False,
    R_LC_edge_coefficient: float = 1.02,
    max_memory_mb: float = 500.0,
    reversed_error_threshold: Optional[float] = None,
) -> OFLEdgeResult:
    """Calculate OFL edges and save as FITS file.

    Note: R_LC_values and P_ms are automatically sorted in descending order.
    If R_LC_edge_coefficient is a list, it will be reordered to maintain
    correspondence with the sorted R_LC values.

    Parameters
    ----------
    experiment_config_path : Union[str, Path]
        Path to experiment configuration file.
    lines_dir : Union[str, Path]
        Directory containing field line batch files.
    template : FieldLine
        FieldLine template specifying which properties to save.
    R_LC_values : np.ndarray
        Array of light cylinder radii for edge calculation.
    output_dir : Optional[Union[str, Path]], optional
        Directory where to save the filtered edge lines FITS files.
        If None, defaults to lines_dir/OFLR_edges. Default is None.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    trim_lines : bool, optional
        Whether to trim field lines to light cylinder before saving. Default is False.
    trim_type : str, optional
        Trimming method ("sphere" or "cylinder"). Default is "sphere".
    trim_radius : Union[float, np.ndarray, Dict[float, float]], optional
        Radius for trimming. Can be:
        - float: Single radius used for all R_LC values (default: 5.0)
        - np.ndarray: Array of radii, one per R_LC value (must match length)
        - Dict[float, float]: Dictionary mapping R_LC values to trim radii
    precise_edges : bool, optional
        If True, use precise edge calculation method. Default is False.
    R_LC_edge_coefficient : float, optional
        Edge coefficient. Default is 1.02.
    max_memory_mb : float, optional
        Maximum memory threshold in MB. If combined edge data exceeds this,
        it will be saved to disk and cleared from memory. Default is 500.0.
    reversed_error_threshold : Optional[float], optional
        Threshold for filtering by reversed_error. If None, no filtering is performed.
        If not None, uses this value instead of experiment config default. Default is None.

    Returns
    -------
    OFLEdgeResult
        Result containing R_LC values and edge footpoint arrays.

    Returns
    -------
    OFLEdgeResult
        Result from edge calculation.

    Raises
    ------
    ValueError
        If data loading fails or no edge footpoints found.
    FileNotFoundError
        If experiment config or lines directory not found.
    """
    
    logger = get_logger(__name__, console_format_brief=True)

    # Set default output_dir if not provided
    if output_dir is None:
        output_dir = Path(lines_dir).parent / "data_products" / "OFLR_edges"
        logger.info(f"Using default output_dir: {output_dir}")

    # Ensure R_LC_values are sorted in descending order
    R_LC_values = np.array(R_LC_values) if not isinstance(R_LC_values, np.ndarray) else R_LC_values
    R_LC_original_order = R_LC_values.copy()
    if len(R_LC_values) > 1:
        sort_indices = np.argsort(R_LC_values)[::-1]
        R_LC_values = R_LC_values[sort_indices]
        logger.info(f"R_LC_values sorted in descending order: {R_LC_values}")
        
        # Reorder trim_radius array to match sorted R_LC_values
        if isinstance(trim_radius, np.ndarray) and len(trim_radius) == len(R_LC_original_order):
            trim_radius = trim_radius[sort_indices]
            logger.info(f"trim_radius reordered to match sorted R_LC_values: {trim_radius}")

    # Phase 1: Load experiment setup
    logger.info("Phase 1: Loading minimal data for edge calculation")
    experiment_config, grid, OFL_data = _load_experiment_setup(
        experiment_config_path, lines_dir
    )

    # Calculate OFL edges
    try:
        if precise_edges:
            if not isinstance(grid, HealpixGrid):
                raise PSRMagPyGridError(
                    f"Precise edge calculation requires HealpixGrid got {type(grid).__name__}."
                )


        calculator = OFLEdgeCalculator(grid, h_LC_coeff=h_LC_coeff, precise_edges=precise_edges)

        # Calculate edges using the chosen method
        edge_result = calculator.calculate_edges(OFL_data, R_LC=R_LC_values, R_LC_edge_coefficient=R_LC_edge_coefficient)
        
        logger.info(f"Calculated edges for {len(edge_result.R_LC)} R_LC values")
    except Exception as e:
        raise PSRMagPyExperimentError(f"Failed to calculate OFL edges: {e}") from e

    # Phase 2: Extract edge data with full template
    logger.info("Phase 2: Extracting edge data with full template")
    
    try:
        _extract_and_save_OFL_data(
            experiment_config=experiment_config,
            template=template,
            lines_dir=lines_dir,
            batch_files=None,  # Will be found inside helper
            footpoint_indices_list=edge_result.edge_footpoints,
            R_LC_values=edge_result.R_LC,
            output_dir=output_dir,
            trim_lines=trim_lines,
            trim_type=trim_type,
            trim_radius=trim_radius,
            max_memory_mb=max_memory_mb,
            data_type="edge",
            reversed_error_threshold=reversed_error_threshold,
        )
        logger.info("Phase 2: Edge data extraction completed")
    except Exception as e:
        raise PSRMagPyIOError(f"Failed to extract and save edge data: {e}") from e

    return edge_result


def calculate_and_save_OFL_regions(
    experiment_config_path: Union[str, Path],
    lines_dir: Union[str, Path],
    template: FieldLine,
    R_LC_values: np.ndarray,
    output_dir: Optional[Union[str, Path]] = None,
    h_LC_coeff: float = 3**1.5 / 2.0,
    trim_lines: bool = False,
    trim_type: str = "sphere",
    trim_radius: Union[float, np.ndarray, Dict[float, float]] = 5.0,
    max_memory_mb: float = 500.0,
    reversed_error_threshold: Optional[float] = None,
) -> OFLEdgeResult:
    """Calculate OFL regions and save as FITS file.

    Note: R_LC_values are automatically sorted in descending order.

    Parameters
    ----------
    experiment_config_path : Union[str, Path]
        Path to experiment configuration file.
    lines_dir : Union[str, Path]
        Directory containing field line batch files.
    template : FieldLine
        FieldLine template specifying which properties to save.
    R_LC_values : np.ndarray
        Array of light cylinder radii for region calculation.
    output_dir : Optional[Union[str, Path]], optional
        Directory where to save the filtered region lines FITS files.
        If None, defaults to lines_dir/OFLR. Default is None.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    trim_lines : bool, optional
        Whether to trim field lines to light cylinder before saving. Default is False.
    trim_type : str, optional
        Trimming method ("sphere" or "cylinder"). Default is "sphere".
    trim_radius : Union[float, np.ndarray, Dict[float, float]], optional
        Radius for trimming. Can be:
        - float: Single radius used for all R_LC values (default: 5.0)
        - np.ndarray: Array of radii, one per R_LC value (must match length)
        - Dict[float, float]: Dictionary mapping R_LC values to trim radii
    max_memory_mb : float, optional
        Maximum memory threshold in MB. If combined region data exceeds this,
        it will be saved to disk and cleared from memory. Default is 500.0.
    reversed_error_threshold : Optional[float], optional
        Threshold for filtering by reversed_error. If None, no filtering is performed.
        If not None, uses this value instead of experiment config default. Default is None.

    Returns
    -------
    OFLEdgeResult
        Result from region calculation (edge_footpoints contains all OFL footpoints).

    Raises
    ------
        ValueError: If data loading fails or no OFL footpoints found
        FileNotFoundError: If experiment config or lines directory not found
    """
    from psr_magpy.experiment.experiment_config import ExperimentConfig
    from psr_magpy.io import load_data
    from psr_magpy.line.field_line_set import FieldLineSet
    from psr_magpy.line.field_line import FieldLine
    from psr_magpy.line.property_system import PropertyName
    import glob

    logger = get_logger(__name__, console_format_brief=True)

    # Set default output_dir if not provided
    if output_dir is None:
        output_dir = Path(lines_dir) / "OFLR"
        logger.info(f"Using default output_dir: {output_dir}")

    # Ensure R_LC_values are sorted in descending order
    R_LC_values = np.array(R_LC_values) if not isinstance(R_LC_values, np.ndarray) else R_LC_values
    R_LC_original_order = R_LC_values.copy()
    if len(R_LC_values) > 1:
        sort_indices = np.argsort(R_LC_values)[::-1]
        R_LC_values = R_LC_values[sort_indices]
        logger.info(f"R_LC_values sorted in descending order: {R_LC_values}")
        
        # Reorder trim_radius array to match sorted R_LC_values
        if isinstance(trim_radius, np.ndarray) and len(trim_radius) == len(R_LC_original_order):
            trim_radius = trim_radius[sort_indices]
            logger.info(f"trim_radius reordered to match sorted R_LC_values: {trim_radius}")

    # Phase 1: Load experiment setup
    logger.info("Phase 1: Loading minimal data for OFL region calculation")
    experiment_config, grid, OFL_data = _load_experiment_setup(
        experiment_config_path, lines_dir
    )

    # Calculate OFL regions (all footpoints, not just edges)
    try:
        logger.info(f"Calculating OFL regions for {len(R_LC_values)} R_LC values")
        
        region_ftp_ids = []
        for r_lc in R_LC_values:
            cur_ftp_ids = OFL_data.get_OFL_footpoints(r_lc, h_LC_coeff=h_LC_coeff)
            region_ftp_ids.append(cur_ftp_ids)
            logger.info(f"R_LC = {r_lc:.2f}: {len(cur_ftp_ids)} OFL footpoints")
        
        # Create result object (reusing OFLEdgeResult structure)
        region_result = OFLEdgeResult(R_LC=R_LC_values, edge_footpoints=region_ftp_ids, precise_edges=False)
        
    except Exception as e:
        raise PSRMagPyExperimentError(f"Failed to calculate OFL regions: {e}") from e

    # Phase 2: Extract region data with full template
    logger.info("Phase 2: Extracting region data with full template")
    
    try:
        _extract_and_save_OFL_data(
            experiment_config=experiment_config,
            template=template,
            lines_dir=lines_dir,
            batch_files=None,  # Will be found inside helper
            footpoint_indices_list=region_result.edge_footpoints,
            R_LC_values=region_result.R_LC,
            output_dir=output_dir,
            trim_lines=trim_lines,
            trim_type=trim_type,
            trim_radius=trim_radius,
            max_memory_mb=max_memory_mb,
            data_type="region",
            reversed_error_threshold=reversed_error_threshold,
        )
        logger.info("Phase 2: Region data extraction completed")
    except Exception as e:
        raise PSRMagPyIOError(f"Failed to extract and save region data: {e}") from e

    return region_result


def load_OFLR_line_data(       
    experiment_number: int,
    R_LC_values: np.ndarray,
    lines_dir: Optional[Union[str, Path]] = None,
    template: Optional[FieldLine] = None,
    file_pattern: Optional[str] = None,
    data_type: str = "edge",
) -> Dict[float, FieldLineSet]:
    """Load OFL line data (edges or regions) from FITS files.
    
    Parameters
    ----------
    experiment_number : int
        Experiment number.
    R_LC_values : np.ndarray
        Array of R_LC values to load.
    lines_dir : Optional[Union[str, Path]], optional
        Directory containing OFL line files. Default is None.
    template : Optional[FieldLine], optional
        FieldLine template for loading. If None, uses template_all(). Default is None.
    file_pattern : Optional[str], optional
        Custom file pattern (currently unused, files are auto-detected). Default is None.
    data_type : str, optional
        Type of data to load: "edge" or "region". Default is "edge".
    
    Returns
    -------
    Dict[float, FieldLineSet]
        Dictionary mapping R_LC values to FieldLineSet objects.
    
    Raises
    ------
    ValueError
        If data_type is not "edge" or "region".
    PSRMagPyIOError
        If no files are found or files cannot be loaded.
    """
    from psr_magpy.io import load_data
    from psr_magpy.line.field_line_set import FieldLineSet
    from psr_magpy.line.field_line import FieldLine
    import glob

    logger = get_logger(__name__, console_format_brief=True)

    # Convert R_LC_values to numpy array and ensure sorted
    R_LC_values = np.array(R_LC_values) if not isinstance(R_LC_values, np.ndarray) else R_LC_values


    # Set up template
    if template is None:
        template = FieldLine.template_all()

    # Determine filename prefix based on data type
    if data_type == "edge":
        stem_prefix = "OFLR_edge_lines"
    elif data_type == "region":
        stem_prefix = "OFLR_lines"
    else:
        error_msg = f"Invalid data_type: {data_type}. Must be 'edge' or 'region'"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Find all candidate files matching the new pattern
    # Pattern: OFLR_{edge_}lines_{experiment_number:03d}_RLC_{r_lc:05.2f}_{chunk_counter:03d}.fits
    pattern = f"{stem_prefix}_{experiment_number:03d}_RLC_*.fits"
    candidate_files = sorted(glob.glob(str(lines_dir / pattern)))

    if not candidate_files:
        raise PSRMagPyIOError(
            f"No FITS files found matching patterns for experiment {experiment_number} in {lines_dir}"
        )

    logger.info(f"Found {len(candidate_files)} candidate FITS files")
    
    # Map R_LC values to files by parsing filenames
    # Pattern: OFLR_{edge_}lines_{experiment_number:03d}_RLC_{r_lc:05.2f}_{chunk_counter:03d}.fits
    R_LC_to_files = {}
    
    for filepath in candidate_files:
        try:
            filename = Path(filepath).stem
            # Extract R_LC from filename: ..._RLC_{r_lc:05.2f}_{chunk_counter:03d}
            parts = filename.split("_RLC_")
            if len(parts) != 2:
                logger.warning(f"Could not parse R_LC from filename: {filepath}")
                continue

            # Extract R_LC value (format: {r_lc:05.2f}_{chunk_counter:03d})
            r_lc_part = parts[1].split("_")[0]
            r_lc = float(r_lc_part)
            # Round to 2 decimal places for consistent matching
            r_lc_rounded = round(r_lc, 2)
            
            # Collect files for this R_LC value
            if r_lc_rounded not in R_LC_to_files:
                R_LC_to_files[r_lc_rounded] = []
            R_LC_to_files[r_lc_rounded].append(filepath)
                        
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing R_LC from filename {filepath}: {e}")
            continue

    if not R_LC_to_files:
        raise PSRMagPyIOError(
            f"No FITS files found with valid R_LC in filename for experiment {experiment_number} in {lines_dir}"
        )

    # Match requested R_LC_values to files
    # Round requested R_LC values to 2 decimal places for consistent matching
    matched_R_LC_to_files = {}
    for r_lc_requested in R_LC_values:
        r_lc_requested_rounded = round(r_lc_requested, 2)
        matching_r_lc = None
        for r_lc_stored in R_LC_to_files.keys():
            if np.isclose(r_lc_requested_rounded, r_lc_stored, rtol=1e-6):
                matching_r_lc = r_lc_stored
                break
        
        if matching_r_lc is None:
            logger.warning(f"No files found with R_LC = {r_lc_requested_rounded:.2f}")
            continue
        
        matched_R_LC_to_files[r_lc_requested_rounded] = sorted(R_LC_to_files[matching_r_lc])
        logger.debug(
            f"R_LC = {r_lc_requested_rounded:.2f} matches stored R_LC = {matching_r_lc:.2f} "
            f"({len(matched_R_LC_to_files[r_lc_requested_rounded])} files)"
        )
    
    R_LC_to_files = matched_R_LC_to_files

    if not R_LC_to_files:
        raise PSRMagPyIOError(
            f"No FITS files found with R_LC values matching {R_LC_values} for experiment {experiment_number}"
        )

    # Load FieldLineSet for each R_LC value
    result = {}
    for r_lc_requested in R_LC_values:
        r_lc_rounded = round(r_lc_requested, 2)
        if r_lc_rounded not in R_LC_to_files:
            logger.warning(f"No files found for R_LC = {r_lc_rounded:.2f}")
            continue

        # Combine all files for this R_LC value
        combined_data = None
        for filepath in sorted(R_LC_to_files[r_lc_rounded]):
            try:
                logger.debug(f"Loading {filepath} for R_LC = {r_lc_rounded:.2f}")
                field_line_set = load_data(FieldLineSet, filepath, template=template)
                
                if field_line_set is None or field_line_set.n_field_lines == 0:
                    logger.debug(f"  No data in file, skipping")
                    continue

                # Combine with previous files
                if combined_data is None:
                    combined_data = field_line_set
                else:
                    combined_data.append(field_line_set)

            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
                continue

        if combined_data is not None and combined_data.n_field_lines > 0:
            result[r_lc_rounded] = combined_data
            logger.info(
                f"Loaded {combined_data.n_field_lines} field lines for R_LC = {r_lc_rounded:.2f}"
            )
        else:
            logger.warning(f"No field lines loaded for R_LC = {r_lc_rounded:.2f}")

    return result
