import argparse as ap
import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates.representation import CartesianRepresentation
from astropy.coordinates import Angle
from astropy.io import fits as pf
from astropy.nddata import CCDData
from FootpointGrid import FootpointGrid


r_ns = u.def_unit("Neutron star radius, 10 km", 10 * u.km)
u.add_enabled_units([r_ns])


def get_parameters_from_command_line():
    # Create the main parser
    parser = ap.ArgumentParser(
        description="Initialize the grid of field line footpoints"
    )

    # Create subparsers for different values of argument #1
    subparsers = parser.add_subparsers(dest="grid_type", help="Tesselation type")

    # Subparser for spherical
    parser_spherical = subparsers.add_parser(
        "spherical",
        help="Uniform division in longitude/latitude. \
                                             Number of footpoints: nlat*(nlong-2)+2",
    )
    parser_spherical.add_argument(
        "parameters",
        nargs="*",
        help="Grid parameters: number of footpoints nlat=<int> nlon=<int>. Pole latitude in degrees pole_lat=0.",
    )

    r0 = parser.add_parser(
        "r0", help="Radial distance of footpoint in R_NS. Default value: 1."
    )

    # Subparser for random
    parser_random = subparsers.add_parser(
        "random", help="Random uniform distribution of points on sphere"
    )
    parser_random.add_argument(
        "parameters",
        nargs="*",
        help="Grid parameters: number of footpoints nline=<int>",
    )
    # parser_random.add_argument('--nline', help='Number of footpoints')

    # Subparser for healpix
    parser_healpix = subparsers.add_parser("healpix", help="Healpix tesselation")
    parser_healpix.add_argument(
        "parameters",
        nargs="*",
        help="Grid parameters: Number of pixels per side nside=<int>. Number of footpoints: 12*nside^2.",
    )

    # Parse the arguments
    args = parser.parse_args()
    grid_type = args.grid_type
    r0 = args.r0 * r_ns
    parameters_dict = {}

    for item in args.parameters:
        if "=" in item:
            key, value = item.split("=", 1)
            parameters_dict[key] = value
        else:
            print(f"Invalid item format: {item}. Expected format: key=value")
    if grid_type == "spherical":
        if not ("pole_lat" in parameters_dict.keys()):
            parameters_dict["pole_lat"] = Angle(0 * u.degree)
        else:
            parameters_dict["pole_lat"] = Angle(
                float(parameters_dict["pole_lat"]) * u.degree
            )

    return grid_type, parameters_dict, r0


def get_parameters_from_fits(fitsname):
    print("Retrieving grid parameters from %s" % fitsname)
    hdulist = pf.open(fitsname)
    header_main = hdulist[0].header
    grid_type = header_main["GRIDTYPE"]
    r0 = header_main["R0"] * u.Unit(header_main["R0UNITS"])
    parameters_dict = {}
    if grid_type == "spherical":
        parameters_dict["nlon"] = header_main["NLON"]
        parameters_dict["nlat"] = header_main["NLAT"]
        parameters_dict["pole_lat"] = header_main["POLELAT"]
    if grid_type == "healpix":
        parameters_dict["nside"] = header_main["NSIDE"]
    if grid_type == "random":
        parameters_dict["nline"] = header_main["NLINE"]

    return grid_type, parameters_dict, r0


def save_grid_to_fits(grid: FootpointGrid, fitsname: str) -> None:
    hdu_main = pf.PrimaryHDU()
    header_main = hdu_main.header
    header_main.set("GRIDTYPE", grid.grid_type, "Grid type")
    header_main.set("NLINE", grid.nline_ori, "Number of field lines in a full grid")
    header_main.set("R0", grid.r0.value, "Radius of the sphere with footpoints")
    header_main.set("R0UNITS", grid.r0.unit.name, "Unit of r0")
    header_main.set("PARTIAL", grid.partial, "Partial grid")

    if grid.grid_type == "spherical":
        header_main.set("NLON", grid.nlon, "Number of longitude divisions")
        header_main.set("NLAT", grid.nlat, "Number of latitude divisions")
        header_main.set("POLELAT", grid.pole_lat, "Latitude of grid pole, degree")
    if grid.grid_type == "healpix":
        header_main.set("NSIDE", grid.nside, "Healpix parameter nside")

    data = CCDData(
        grid.footpoint_coordinates.xyz, unit=grid.footpoint_coordinates.xyz.unit
    )
    hdu_footpoints = data.to_hdu(as_image_hdu=True)[0]
    hdu_footpoints.name = "FTP_CRD"
    print("HERE")
    print(grid.indices)
    col = pf.Column(name="footpoint_id", format="K", array=grid.indices)
    hdu_indices = pf.BinTableHDU.from_columns([col], name="FTP_ID")
    hdul = pf.HDUList([hdu_main, hdu_indices, hdu_footpoints])
    print(f"Saving footpoints to {fitsname}")
    hdul.writeto(fitsname, overwrite=True)
    return


def load_grid_from_fits(fitsname: str) -> FootpointGrid:
    print("Retrieving grid parameters and coordinates from %s" % fitsname)
    grid_type, parameters_dict, r0 = get_parameters_from_fits(fitsname)
    grid = FootpointGrid(grid_type, parameters_dict, r0)
    hdulist = pf.open(fitsname)
    footpoint_coordinates = hdulist["FTP_CRD"].data
    grid.footpoint_coordinates = CartesianRepresentation(
        footpoint_coordinates, unit=r0.unit
    )
    # There shouldn't be a need for this conversion. Investigate later.
    grid.indices = np.array([row[0] for row in hdulist["FTP_ID"].data])
    grid.partial = hdulist[0].header["PARTIAL"]
    return grid
