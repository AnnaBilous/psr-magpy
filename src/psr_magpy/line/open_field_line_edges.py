import healpy as hp
import numpy as np
from typing import NamedTuple, List
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.utils.physical_conversions import get_R_LC, get_P_ms


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


