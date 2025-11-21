import healpy as hp
import numpy as np
from typing import NamedTuple, List, Union, Optional
from pathlib import Path
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.property_system import PropertyName
from psr_magpy.grid.grid import HealpixGrid, SphericalGrid, RandomGrid, CustomGrid, FootpointGrid
from psr_magpy.utils.physical_conversions import get_R_LC, get_P_ms
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

    full_grid_ftp_ids_not_inner = grid.indices[ftp_ids_not_inner]
    full_grid_ftp_ids = grid.indices[ftp_ids]

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
    for iii,f in enumerate(ftp_ids_not_inner):
      if is_edge[iii] and False:  
        print(n_neighbours_per_footpoint[iii])
        for jj in range(8):
            print("\t", neighbour_footpoints[jj,iii])
        print(n_neighbours_in_ftp_ids[iii])
        print(is_edge[iii])
        print("=====" )
        plt.plot(grid.longitude, grid.colatitude, 'o', color='0.5')
        plt.plot(grid.longitude[ftp_ids_not_inner], grid.colatitude[ftp_ids_not_inner], 'k.')
        plt.scatter([l[f]], [c[f]], s=150, color='yellow')
        plt.scatter(l[neighbour_footpoints[:,iii]], c[neighbour_footpoints[:,iii]], s=150, color='green')
        plt.show()

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


def calculate_and_save_OFL_edges(
    experiment_config_path: Union[str, Path],
    lines_dir: Union[str, Path],
    template: FieldLine,
    R_LC_values: np.ndarray,
    output_path: Union[str, Path],
    h_LC_coeff: float = 3**1.5 / 2.0,
    trim_lines: bool = False,
    trim_type: str = "sphere",
    trim_radius: float = 5.0,
    precise_edges: bool = False,
    R_LC_edge_coefficient: float = 1.02,
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
    output_path : Union[str, Path]
        Path where to save the filtered edge lines FITS file.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    trim_lines : bool, optional
        Whether to trim field lines to light cylinder before saving. Default is False.
    trim_type : str, optional
        Trimming method ("sphere" or "cylinder"). Default is "sphere".
    trim_radius : float, optional
        Radius for trimming (light cylinder radius). Default is 5.0.
    precise_edges : bool, optional
        If True, use precise edge calculation method. Default is False.
    R_LC_edge_coefficient : float, optional
        Edge coefficient. Default is 1.02.

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
    from psr_magpy.experiment.experiment_config import ExperimentConfig
    from psr_magpy.io import load_data
    from psr_magpy.line.field_line_set import FieldLineSet
    from psr_magpy.line.field_line import FieldLine
    from psr_magpy.line.property_system import PropertyName
    import glob

    logger = get_logger(__name__, console_format_brief=True)

    # Ensure R_LC_values are sorted in descending order
    R_LC_values = np.array(R_LC_values) if not isinstance(R_LC_values, np.ndarray) else R_LC_values
    if len(R_LC_values) > 1:
        R_LC_values = np.sort(R_LC_values)[::-1]
        logger.info(f"R_LC_values sorted in descending order: {R_LC_values}")

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

    # Calculate OFL edges
    try:
        if precise_edges:
            if not isinstance(grid, HealpixGrid):
                raise PSRMagPyGridError(
                    f"Precise edge calculation requires HealpixGrid with nside < 512, got {type(grid).__name__}."
                )
            """
            else:
                if grid.nside >= 512:
                    raise PSRMagPyGridError(
                        f"Precise edge calculation requires HealpixGrid with nside < 512, got nside = {grid.nside}."
                    )
            """

        calculator = OFLEdgeCalculator(grid, h_LC_coeff=h_LC_coeff, precise_edges=precise_edges)

        # Calculate edges using the chosen method
        edge_result = calculator.calculate_edges(OFL_data, R_LC=R_LC_values, R_LC_edge_coefficient=R_LC_edge_coefficient)
        
        logger.info(f"Calculated edges for {len(edge_result.R_LC)} R_LC values")
    except Exception as e:
        raise PSRMagPyExperimentError(f"Failed to calculate OFL edges: {e}") from e

    # Phase 2: Extract edge data with full template
    logger.info("Phase 2: Extracting edge data with full template")
    
    # Ensure reversed_error is in template if needed for filtering
    if not template.property_system.save_flags.save_reversed_error:
        template.property_system.save_flags.save_reversed_error = True
        logger.debug("Added reversed_error to template for filtering")
    
    try:
        # Find all batch files
        batch_pattern = f"{lines_dir}/{experiment_config.name}_batch_*.fits"
        batch_files = sorted(glob.glob(batch_pattern))

        if not batch_files:
            raise PSRMagPyIOError(f"No batch files found matching pattern: {batch_pattern}")

        logger.info(f"Found {len(batch_files)} batch files")

        # Process each R_LC value and its edge footpoints
        output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        base_path = output_path.parent / output_path.stem
        extension = output_path.suffix

        for i, r_lc in enumerate(edge_result.R_LC):
            footpoint_indices = edge_result.edge_footpoints[i]
            logger.info(
                f"Processing R_LC = {r_lc:.2f}: {len(footpoint_indices)} edge footpoints"
            )

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
                    batch_data = load_data(
                        FieldLineSet, batch_file, template=template
                    )

                    if batch_data is None or batch_data.n_field_lines == 0:
                        logger.debug(f"    No data in batch file, skipping")
                        continue

                    # Get footpoint IDs from this batch
                    batch_footpoint_ids = batch_data.property_system.get_property(
                        PropertyName.FOOTPOINT_ID
                    )

                    if batch_footpoint_ids is None:
                        logger.debug(f"    No footpoint IDs in batch file, skipping")
                        continue

                    # Get reversed_error if available for filtering
                    batch_reversed_error = batch_data.property_system.get_property(
                        PropertyName.REVERSED_ERROR
                    )
                    reversed_error_threshold = getattr(experiment_config.tracing, 'reversed_error_threshold', 1.0e-3)
                    
                    # Find footpoint IDs that are in the edge and pass error threshold
                    edge_footpoint_ids_in_batch = []
                    for j, fp_id in enumerate(batch_footpoint_ids):
                        if fp_id in edge_footpoint_set:
                            # Filter by reversed_error if available
                            if batch_reversed_error is not None and len(batch_reversed_error) > j:
                                if batch_reversed_error[j] > reversed_error_threshold:
                                    continue  # Skip this footpoint due to high error
                            edge_footpoint_ids_in_batch.append(fp_id)

                    if not edge_footpoint_ids_in_batch:
                        logger.debug(f"    No edge footpoints in this batch, skipping")
                        continue

                    # Create subset of this batch
                    batch_subset = batch_data.subset_by_footpoint_ids(
                        edge_footpoint_ids_in_batch
                    )

                    if batch_subset.n_field_lines > 0:
                        logger.debug(
                            f"    Found {batch_subset.n_field_lines} edge field lines in batch"
                        )

                        # Combine with previous batch subsets
                        if combined_edge_data is None:
                            combined_edge_data = batch_subset
                        else:
                            combined_edge_data.append(batch_subset)
                    else:
                        logger.debug(
                            f"    No edge field lines found in this batch (this is normal)"
                        )

                except Exception as e:
                    logger.warning(f"    Error processing batch file {batch_file}: {e}")
                    continue

            # Save combined edge data for this R_LC value
            if combined_edge_data is not None and combined_edge_data.n_field_lines > 0:
                subset_filename = f"{base_path}_{i:03d}{extension}"

                logger.info(
                    f"  Saving {combined_edge_data.n_field_lines} edge field lines to {subset_filename}"
                )

                # Trim field lines if requested
                if trim_lines:
                    logger.info(
                        f"  Trimming field lines to R_LC = {trim_radius} using {trim_type} method"
                    )
                    combined_edge_data.trim_to_light_cylinder(
                        R_LC=trim_radius, trim_type=trim_type
                    )
                    logger.info(
                        f"  Trimmed edge data now has {combined_edge_data.n_field_lines} field lines"
                    )

                # Create I/O instance and add R_LC keyword to header
                from psr_magpy.io.data_io import FieldLineSetIO

                io_instance = FieldLineSetIO(combined_edge_data)
                io_instance.add_header_keyword(
                    "R_LC", r_lc, "Light cylinder radius for this subset"
                )

                # Save with the I/O instance that has the header keyword
                io_instance.save(combined_edge_data, subset_filename)

                logger.info(f"  Successfully saved edge data for R_LC = {r_lc:.2f}")
            else:
                logger.warning(
                    f"  No edge field lines found for R_LC = {r_lc:.2f} across all batch files"
                )
                logger.warning(
                    f"  Expected {len(edge_footpoint_set)} footpoint IDs but found none"
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
    output_path: Union[str, Path],
    h_LC_coeff: float = 3**1.5 / 2.0,
    trim_lines: bool = False,
    trim_type: str = "sphere",
    trim_radius: float = 5.0,
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
    output_path : Union[str, Path]
        Path where to save the filtered region lines FITS file.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    trim_lines : bool, optional
        Whether to trim field lines to light cylinder before saving. Default is False.
    trim_type : str, optional
        Trimming method ("sphere" or "cylinder"). Default is "sphere".
    trim_radius : float, optional
        Radius for trimming (light cylinder radius). Default is 5.0.

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

    # Ensure R_LC_values are sorted in descending order
    R_LC_values = np.array(R_LC_values) if not isinstance(R_LC_values, np.ndarray) else R_LC_values
    if len(R_LC_values) > 1:
        R_LC_values = np.sort(R_LC_values)[::-1]
        logger.info(f"R_LC_values sorted in descending order: {R_LC_values}")

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

    # Phase 1: Load minimal data for OFL region calculation
    logger.info("Phase 1: Loading minimal data for OFL region calculation")
    try:
        # Create minimal template for OFL calculation
        minimal_template = FieldLine(
            save_footpoint_id=True,  # Always required
            save_r_cyl_max=True,
            save_z_abs_max=True,
            # All other properties default to False
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
    
    # Ensure reversed_error is in template for filtering
    reversed_error_threshold = getattr(experiment_config.tracing, 'reversed_error_threshold', 1.0e-3)
    if not template.property_system.save_flags.save_reversed_error:
        template.property_system.save_flags.save_reversed_error = True
        logger.debug("Added reversed_error to template for filtering")
    
    try:
        # Find all batch files
        batch_pattern = f"{lines_dir}/{experiment_config.name}_batch_*.fits"
        batch_files = sorted(glob.glob(batch_pattern))

        if not batch_files:
            raise PSRMagPyIOError(f"No batch files found matching pattern: {batch_pattern}")

        logger.info(f"Found {len(batch_files)} batch files")

        # Process each R_LC value and its region footpoints
        output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        base_path = output_path.parent / output_path.stem
        extension = output_path.suffix

        for i, r_lc in enumerate(region_result.R_LC):
            footpoint_indices = region_result.edge_footpoints[i]  # Contains all OFL footpoints
            logger.info(
                f"Processing R_LC = {r_lc:.2f}: {len(footpoint_indices)} OFL footpoints"
            )

            if len(footpoint_indices) == 0:
                logger.warning(f"  No OFL footpoints for R_LC = {r_lc:.2f}")
                continue

            # Convert to set for efficient lookup
            region_footpoint_set = set(footpoint_indices.tolist())

            # Load and subset each batch file
            combined_region_data = None

            for batch_file in batch_files:
                logger.debug(f"  Processing batch file: {batch_file}")

                try:
                    # Load batch file with full template
                    batch_data = load_data(
                        FieldLineSet, batch_file, template=template
                    )

                    if batch_data is None or batch_data.n_field_lines == 0:
                        logger.debug(f"    No data in batch file, skipping")
                        continue

                    # Get footpoint IDs from this batch
                    batch_footpoint_ids = batch_data.property_system.get_property(
                        PropertyName.FOOTPOINT_ID
                    )

                    if batch_footpoint_ids is None:
                        logger.debug(f"    No footpoint IDs in batch file, skipping")
                        continue

                    # Get reversed_error if available for filtering
                    batch_reversed_error = batch_data.property_system.get_property(
                        PropertyName.REVERSED_ERROR
                    )
                    
                    # Find footpoint IDs that are in the region and pass error threshold
                    region_footpoint_ids_in_batch = []
                    for j, fp_id in enumerate(batch_footpoint_ids):
                        if fp_id in region_footpoint_set:
                            # Filter by reversed_error if available
                            if batch_reversed_error is not None and len(batch_reversed_error) > j:
                                if batch_reversed_error[j] > reversed_error_threshold:
                                    continue  # Skip this footpoint due to high error
                            region_footpoint_ids_in_batch.append(fp_id)

                    if not region_footpoint_ids_in_batch:
                        logger.debug(f"    No region footpoints in this batch, skipping")
                        continue

                    # Create subset of this batch
                    batch_subset = batch_data.subset_by_footpoint_ids(
                        region_footpoint_ids_in_batch
                    )

                    if batch_subset.n_field_lines > 0:
                        logger.debug(
                            f"    Found {batch_subset.n_field_lines} region field lines in batch"
                        )

                        # Combine with previous batch subsets
                        if combined_region_data is None:
                            combined_region_data = batch_subset
                        else:
                            combined_region_data.append(batch_subset)
                    else:
                        logger.debug(
                            f"    No region field lines found in this batch (this is normal)"
                        )

                except Exception as e:
                    logger.warning(f"    Error processing batch file {batch_file}: {e}")
                    continue

            # Save combined region data for this R_LC value
            if combined_region_data is not None and combined_region_data.n_field_lines > 0:
                subset_filename = f"{base_path}_{i:03d}{extension}"

                logger.info(
                    f"  Saving {combined_region_data.n_field_lines} region field lines to {subset_filename}"
                )

                # Trim field lines if requested
                if trim_lines:
                    logger.info(
                        f"  Trimming field lines to R_LC = {trim_radius} using {trim_type} method"
                    )
                    combined_region_data.trim_to_light_cylinder(
                        R_LC=trim_radius, trim_type=trim_type
                    )
                    logger.info(
                        f"  Trimmed region data now has {combined_region_data.n_field_lines} field lines"
                    )

                # Create I/O instance and add R_LC keyword to header
                from psr_magpy.io.data_io import FieldLineSetIO

                io_instance = FieldLineSetIO(combined_region_data)
                io_instance.add_header_keyword(
                    "R_LC", r_lc, "Light cylinder radius for this subset"
                )

                # Save with the I/O instance that has the header keyword
                io_instance.save(combined_region_data, subset_filename)

                logger.info(f"  Successfully saved region data for R_LC = {r_lc:.2f}")
            else:
                logger.warning(
                    f"  No region field lines found for R_LC = {r_lc:.2f} across all batch files"
                )
                logger.warning(
                    f"  Expected {len(region_footpoint_set)} footpoint IDs but found none"
                )

        logger.info("Phase 2: Region data extraction completed")

    except Exception as e:
        raise PSRMagPyIOError(f"Failed to extract and save region data: {e}") from e

    return region_result
