import numpy as np
import healpy as hp
from astropy.coordinates.representation import CartesianRepresentation


def get_R_LC(P_ms: float):
    return 4.77 * P_ms


def get_OFL_footpoints(stat_dict, R_LC, h_LC_coeff=3**1.5 / 2.0):
    R_cyl_max = np.array(stat_dict["R_cyl_max"])
    z_abs_max = np.array(stat_dict["z_abs_max"])
    footpoint_ids = np.array(stat_dict["footpoint_id"])
    open_fl = (R_cyl_max > R_LC) | (
        (R_cyl_max < R_LC) & (np.abs(z_abs_max) > (h_LC_coeff * R_LC))
    )
    return footpoint_ids[open_fl]


def trim_field_line(field_line, R_LC, h_LC_coeff=3**1.5 / 2.0):
    z_abs = np.abs(field_line.z)
    R_cyl = (field_line.x**2 + field_line.y**2) ** 0.5
    outside_LC = (R_cyl > R_LC) | ((R_cyl < R_LC) & (z_abs > (h_LC_coeff * R_LC)))
    ind = np.arange(len(z_abs))
    if len(ind[outside_LC]) > 0:
        first_LC_penetration_point = ind[outside_LC][0]
        field_line_trimmed = CartesianRepresentation(
            x=field_line.x[:first_LC_penetration_point],
            y=field_line.y[:first_LC_penetration_point],
            z=field_line.z[:first_LC_penetration_point],
        )
        return field_line_trimmed
    else:
        return field_line


def healpix_grid_get_single_OFL_edge(grid, ftp_ids, ftp_ids_inner=[]):
    assert (
        grid.grid_type == "healpix"
    ), "Finding OFL edge is currently supported only for healpix grid"
    if not (ftp_ids_inner == []):
        ftp_ids_not_inner = np.setdiff1d(ftp_ids, ftp_ids_inner)
    else:
        ftp_ids_not_inner = np.copy(ftp_ids)
    n_ftp = len(ftp_ids_not_inner)
    is_edge = np.zeros(n_ftp, dtype=bool)

    # Eight nearest neighbours
    neighbour_footpoints = hp.pixelfunc.get_all_neighbours(
        grid.nside, ftp_ids_not_inner
    )
    for ftp_ind in range(n_ftp):
        local_neighbours = np.copy(neighbour_footpoints[:, ftp_ind])
        # Removing non-existent neighbours marked with ipix = -1
        local_neighbours = local_neighbours[local_neighbours > 0]
        n_local_neighbours = len(local_neighbours)
        n_local_neighbours_in_ftp_ids = len(
            local_neighbours[np.isin(local_neighbours, ftp_ids)]
        )
        is_edge[ftp_ind] = n_local_neighbours_in_ftp_ids < n_local_neighbours - 1

    ftp_ids_edge = ftp_ids_not_inner[is_edge]
    return ftp_ids_edge


def healpix_grid_get_OFL_edges(grid, P_grid, stat_dict, h_LC_coeff=3**1.5 / 2.0):
    if not (isinstance(P_grid, list)):
        P_grid = [P_grid]
    P_grid = np.sort(P_grid)[::-1]
    n_P = len(P_grid)
    R_LC_grid = get_R_LC(P_grid)
    ftp_ids = []
    edge_ftp_ids = []
    inner_ftp_ids = [[]]
    for i, R_LC in enumerate(R_LC_grid):
        cur_ftp_ids = get_OFL_footpoints(stat_dict, R_LC, h_LC_coeff)
        ftp_ids.append(cur_ftp_ids)
        inner_ftp_ids.append(cur_ftp_ids)
    for i in range(n_P):
        cur_edge_ftp_ids = healpix_grid_get_single_OFL_edge(
            grid, ftp_ids[i], ftp_ids_inner=inner_ftp_ids[i]
        )
        edge_ftp_ids.append(cur_edge_ftp_ids)
    return edge_ftp_ids
