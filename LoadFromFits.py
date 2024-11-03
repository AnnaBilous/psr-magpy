from astropy.io import fits as pf
import numpy as np
from FootpointGrid import FootpointGrid
from CalculateField import MultipoleField
import FootpointGrid_InputOutput as fp_io
from astropy.coordinates.representation import CartesianRepresentation

    
def retrieve_grid(fitsname):
    print("Retrieving grid parameters and coordinates from %s" % fitsname)
    grid_type, parameters_dict, r0 = fp_io.get_parameters_from_fits(fitsname)
    grid = FootpointGrid(grid_type, parameters_dict, r0)
    hdu_header, _, hdu_field_data = pf.open(fitsname)
    grid.nline_ori = hdu_header["NLINE_ORI"]
    grid.nline = hdu_header["NLINE"]
    grid.partial = hdu_header.header["PARTIAL"]
    
    if "footpoint_id" in hdu_field_data.columns.names:
        # There shouldn't be a need for this conversion. Investigate later.
        grid.indices = np.array([row[0] for row in hdu_field_data.data["footpoint_id"]])
    
    if "r_ini" in hdu_field_data.columns.names:
        footpoint_coordinates = hdu_field_data.data["r_ini"]
        grid.footpoint_coordinates = CartesianRepresentation(footpoint_coordinates, unit=r0.unit)
    else:
        print("Recalculating footpoint coordinates...")
        assert (grid.grid_type != "random"), print("Footpoint coordinates for random grid can not be recovered")
        grid.calculate_grid()
        grid.subset_grid(footpoint_coordinates)
    return grid

def retrieve_field_components(fitsname):
    _, hdu_field_components, _ = pf.open(fitsname)

    data = hdu_field_components.data
    components_data = []

    for row in data:
        component_data = {
            "component_number": row["component_number"],
            "moment_type": row['moment_type'],
            "B_magnitude": row['B_magnitude'],
            "moment_orientation_latitude_degree": row['orientation_theta_deg'],
            "moment_orientation_longitude_degree": row['orientation_phi_deg'],
            "offset_x_r_ns": row['offset_x_R_NS'],
            "offset_y_r_ns": row['offset_y_R_NS'],
            "offset_z_r_ns": row['offset_z_R_NS']
            }
        components_data.append(component_data)

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
        vars_to_retrieve = vars & existing_variables
        not_allowed_vars = vars - existing_variables
        assert len(not_allowed_vars) == 0, print(
            f"Following variables are not recognized: {not_allowed_vars}.\nPlease use the following: {existing_variables}"
        )
    stat_dict = []
    for key in vars_to_retrieve:
        stat_dict[key] = hdu_field_data.data[key]
        
    return stat_dict
    
        
