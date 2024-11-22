from astropy.io import fits as pf
import numpy as np
from FootpointGrid import FootpointGrid
from CalculateField import MultipoleField
import FootpointGrid_InputOutput as fp_io
from astropy.coordinates.representation import CartesianRepresentation


def read_from_fits(fitsname, vars_to_retrieve=""):
    grid = retrieve_grid(fitsname)
    magnetic_field_components = retrieve_field_components(fitsname)
    stat_dict = retrieve_field_data(fitsname, vars_to_retrieve)
    return grid, magnetic_field_components, stat_dict


def retrieve_grid(fitsname):
    print("Retrieving grid parameters and coordinates from %s" % fitsname)
    grid_type, parameters_dict, r0 = fp_io.get_parameters_from_fits(fitsname)
    grid = FootpointGrid(grid_type, parameters_dict, r0)
    hdu, _, hdu_field_data = pf.open(fitsname)
    hdu_header = hdu.header
    grid.nline_ori = hdu_header["NLINE_ORI"]
    grid.nline = hdu_header["NLINE"]
    grid.partial = hdu_header["PARTIAL"]

    if "footpoint_id" in hdu_field_data.columns.names:
        grid.indices = hdu_field_data.data["footpoint_id"]

    if "r_ini" in hdu_field_data.columns.names:
        footpoint_coordinates = hdu_field_data.data["r_ini"]
        grid.footpoint_coordinates = CartesianRepresentation(
            footpoint_coordinates.T, unit=r0.unit
        )
    else:
        print("Recalculating footpoint coordinates...")
        assert grid.grid_type != "random", print(
            "Footpoint coordinates for random grid can not be recovered"
        )
        grid.calculate_grid()
        # grid.subset_grid(footpoint_coordinates)
    return grid


def retrieve_field_components(fitsname):
    _, hdu_field_components, _ = pf.open(fitsname)

    data = hdu_field_components.data
    n_components = np.shape(data)[0]
    names_dict = {
        "component_number": "component_number",
        "moment_type": "moment_type",
        "B_magnitude": "B_magnitude",
        "moment_orientation_latitude_degree": "orientation_theta_deg",
        "moment_orientation_longitude_degree": "orientation_phi_deg",
        "offset_x_r_ns": "offset_x_R_NS",
        "offset_y_r_ns": "offset_y_R_NS",
        "offset_z_r_ns": "offset_z_R_NS",
    }
    components_data = [
        {key: None for key in names_dict.keys()} for _ in range(n_components)
    ]

    for key, val in names_dict.items():
        curr_val = data[val]
        for i in range(n_components):
            components_data[i][key] = curr_val[i]
            # Some weird mismatch in integer types
            if key == "component_number":
                components_data[i][key] = int(curr_val[i])

    magnetic_field_components = MultipoleField()
    magnetic_field_components.initialize(components_data)
    return magnetic_field_components


def retrieve_field_data(fitsname, vars_to_retrieve=""):
    _, _, hdu_field_data = pf.open(fitsname)
    existing_variables = set(hdu_field_data.columns.names)

    if vars_to_retrieve == "":
        vars_to_retrieve = existing_variables
    else:
        vars_to_retrieve = set(vars_to_retrieve.replace(" ", "").split(","))
        not_allowed_vars = vars_to_retrieve - existing_variables
        vars_to_retrieve = vars_to_retrieve & existing_variables
        assert len(not_allowed_vars) == 0, print(
            f"Following variables are not recognized: {not_allowed_vars}.\nPlease use the following: {existing_variables}"
        )
    stat_dict = {}
    for key in vars_to_retrieve:
        stat_dict[key] = hdu_field_data.data[key]
    if "field_lines" in stat_dict.keys():
        for i in range(len(stat_dict["field_lines"])):
            field_line_points = len(stat_dict["field_lines"][i])
            if field_line_points > 0:
                stat_dict["field_lines"][i] = stat_dict["field_lines"][i].reshape(
                    (field_line_points // 3, 3)
                )

    return stat_dict
