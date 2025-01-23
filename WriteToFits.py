from astropy.io import fits as pf


class Field:
    def __init__(self, multipole_components):
        self.multipole_components = multipole_components

    def create_hdu_field_components(self):
        magnetic_field_list = [vars(component) for component in self.multipole_components]
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
        return hdu_field_components


def write_to_fits(
    multipole_components, footpoints_grid, vars_to_save_dict, filename, Rmin, Rmax
):
    print(f"Saving {vars_to_save_dict.keys()}")
    hdu_main = pf.PrimaryHDU()
    hdu_main = update_main_header(hdu_main, footpoints_grid)
    
    # Create Field instance and use its method
    field = Field(multipole_components)
    hdu_field_components = field.create_hdu_field_components()
    
    hdu_field_data = create_hdu_field_data(vars_to_save_dict, Rmin, Rmax)

    hdulist = [hdu_main, hdu_field_components, hdu_field_data]
    print(f"Writing {filename}.")
    hdulist = pf.HDUList(hdulist)
    hdulist.writeto(filename, overwrite=True)
    return


def update_main_header(hdu_main, footpoints_grid):
    header_main = hdu_main.header
    header_main.set("GRIDTYPE", footpoints_grid.grid_type, "Grid type")
    header_main.set(
        "NLINE_ORI", footpoints_grid.nline_ori, "Original number of footpoints"
    )
    header_main.set("NLINE", footpoints_grid.nline, "Number of footpoints")
    header_main.set("PARTIAL", footpoints_grid.partial, "Partial grid")

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
    return hdu_main


def create_hdu_field_data(vars_to_save_dict, Rmin, Rmax):
    fits_columns = []
    col_formats = dict.fromkeys(vars_to_save_dict, "D")
    col_formats["footpoint_id"] = "K"
    col_formats["field_lines"] = "PE()"
    col_formats["r_ini"] = "3D"

    for key, value in vars_to_save_dict.items():
        col = pf.Column(name=key, format=col_formats[key], array=value)
        fits_columns.append(col)

    hdu_field_data = pf.BinTableHDU.from_columns(fits_columns, name="FLD_CLC")
    hdu_field_data.header["R_MIN"] = Rmin
    hdu_field_data.header["R_MAX"] = Rmax
    return hdu_field_data
