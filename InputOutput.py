from astropy.io import fits as pf
import numpy as np
from FootpointGrid import FootpointGrid


def write_to_fits(
    multipole_components, footpoints_grid, vars_to_save_dict, filename, Rmin, Rmax
):
    print(f"Saving {vars_to_save_dict.keys()}")
    hdu_main = pf.PrimaryHDU()
    header_main = hdu_main.header
    header_main.set("GRIDTYPE", footpoints_grid.grid_type, "Grid type")
    header_main.set("NLINE", footpoints_grid.nline, "Number of field lines")
    header_main.set(
        "R0", footpoints_grid.r0.value, "Radius of the sphere with footpoints"
    )
    header_main.set("R0UNITS", footpoints_grid.r0.unit.name, "Unit of r0")
    if footpoints_grid.grid_type == "spherical":
        header_main.set("NLON", footpoints_grid.nlon, "Number of longitude divisions")
        header_main.set("NLAT", footpoints_grid.nlat, "Number of latitude divisions")
        header_main.set(
            "POLELAT", footpoints_grid.pole_lat, "Latitude of grid pole, degree"
        )
    if footpoints_grid.grid_type == "healpix":
        header_main.set("NSIDE", footpoints_grid.nside, "Healpix parameter nside")

    magnetic_field_list = [vars(component) for component in multipole_components]
    column_names = ["component_number", "moment_type", "B_magnitude"]
    column_formats = ["K", "10A", "D"]
    magnetic_field_columns = []
    for i, key in enumerate(column_names):
        value = [v[key] for v in magnetic_field_list]
        col = pf.Column(name=key, format=column_formats[i], array=value)
        magnetic_field_columns.append(col)

    value = [v["moment_orientation"].phi.degree for v in magnetic_field_list]
    col = pf.Column(name="orientation_phi_deg", format="D", array=value)
    magnetic_field_columns.append(col)
    value = [v["moment_orientation"].theta.degree for v in magnetic_field_list]
    col = pf.Column(name="orientation_theta_deg", format="D", array=value)
    magnetic_field_columns.append(col)

    value = [v["moment_offset_vector"].x for v in magnetic_field_list]
    col = pf.Column(name="offset_x_R_NS", format="D", array=value)
    magnetic_field_columns.append(col)
    value = [v["moment_offset_vector"].y for v in magnetic_field_list]
    col = pf.Column(name="offset_y_R_NS", format="D", array=value)
    magnetic_field_columns.append(col)
    value = [v["moment_offset_vector"].z for v in magnetic_field_list]
    col = pf.Column(name="offset_z_R_NS", format="D", array=value)
    magnetic_field_columns.append(col)
    hdu_field_components = pf.BinTableHDU.from_columns(
        magnetic_field_columns, name="FLD_PRM"
    )

    fits_columns = []
    col_formats = dict.fromkeys(vars_to_save_dict, "D")
    col_formats["footpoint_id"] = "K"
    col_formats["field_lines"] = "PE"
    col_formats["r_ini"] = "3D"

    for key, value in vars_to_save_dict.items():
        print(key)
        col = pf.Column(name=key, format=col_formats[key], array=value)

        fits_columns.append(col)

    hdu_field_data = pf.BinTableHDU.from_columns(fits_columns, name="FLD_CLC")
    hdu_field_data.header["R_MIN"] = Rmin
    hdu_field_data.header["R_MAX"] = Rmax
    hdulist = [hdu_main, hdu_field_components, hdu_field_data]

    print(f"Writing {filename}.")
    hdulist = pf.HDUList(hdulist)
    hdulist.writeto(filename, overwrite=True)
    return 0


def read_from_fits(filename):
    hdu_header, hdu_field_descr, hdu_field_data = pf.open(filename)
    header_data = hdu_header.header
    grid_type = header_data["GRIDTYPE"]
    parameters_dict = {}

    if grid_type == "spherical":
        parameters_dict["nlon"] = header_data["NLON"]
        parameters_dict["nlon"] = header_data["NLON"]
        header_main.set("NLAT", footpoints_grid.nlat, "Number of latitude divisions")
        header_main.set(
            "POLELAT", footpoints_grid.pole_lat, "Latitude of grid pole, degree"
        )
    if footpoints_grid.grid_type == "healpix":
        header_main.set("NSIDE", footpoints_grid.nside, "Healpix parameter nside")

    # Footpoint grid
    footpoints_grid = FootpointGrid()
    r0 = 1 * r_ns

    footpoints_grid.calculate_grid(grid_type, parameters_dict, r0)

    return footpoints_grid, magnetic_field, vars_dict, field_lines
