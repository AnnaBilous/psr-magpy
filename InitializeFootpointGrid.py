import argparse as ap
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation
from astropy.coordinates import Angle
from scipy.spatial.transform import Rotation as rot
from astropy.io import fits as pf
from astropy.nddata import CCDData


r_ns = u.def_unit("Neutron star radius, 10 km", 10 * u.km)
u.add_enabled_units([r_ns])


class FootpointGrid:
    """Grid of field line foopoints.
    Parameters:
        grid_type: "spherical", "random", "healpix"
    """

    def __init__(self):
        pass

    def calculate_grid(self, grid_type: str, params: dict, r0: u.Quantity) -> None:
        self.grid_type = grid_type
        self.r0 = r0
        if grid_type == "spherical":
            assert "nlon" in params.keys(), "Missing nlon"
            assert "nlat" in params.keys(), "Missing nlat"
            self.nlon = int(params["nlon"])
            self.nlat = int(params["nlat"])
            self.pole_lat = params["pole_lat"]
            footpoints = self.__prepare_spherical_footpoints(
                self.nlon, self.nlat, self.r0, self.pole_lat
            )
            print("Calculating footpoints for spherical grid")
        if grid_type == "healpix":
            assert "nside" in params.keys(), "Missing nside"
            self.nside = int(params["nside"])
            footpoints = self.__prepare_healpix_footpoints(self.nside, r0)
            print("Calculating footpoints for healpix grid")
        if grid_type == "random":
            assert "nline" in params.keys(), "Missing nline"
            self.nline = int(params["nline"])
            footpoints = self.__prepare_random_footpoints(self.nline, r0)
            print("Calculating footpoints for random grid")
        self.footpoint_coordinates = footpoints
        return

    def save_grid_to_fits(self, fitsname: str) -> None:
        hdu_main = pf.PrimaryHDU()
        # hdu_footpoints = pf.ImageHDU(name="footpoints_coordinates")
        header_main = hdu_main.header
        header_main.set("GRIDTYPE", self.grid_type, "Grid type")
        header_main.set("NLINE", self.nline, "Number of field lines")
        header_main.set("R0", self.r0.value, "Radius of the sphere with footpoints")
        header_main.set("R0UNITS", self.r0.unit.name, "Unit of r0")
        # header_main.set('CRDUNITS', self.footpoint_coordinates.x.unit.name, 'Unit of footpoints coords')

        if self.grid_type == "spherical":
            header_main.set("NLON", self.nlon, "Number of longitude divisions")
            header_main.set("NLAT", self.nlat, "Number of latitude divisions")
            header_main.set(
                "POLELAT", self.pole_lat.degree, "Latitude of grid pole, degree"
            )
        if self.grid_type == "healpix":
            header_main.set("NSIDE", self.nside, "Healpix parameter nside")

        data = CCDData(
            self.footpoint_coordinates.xyz, unit=self.footpoint_coordinates.xyz.unit
        )
        hdu_footpoints = data.to_hdu(as_image_hdu=True)[0]
        hdul = pf.HDUList([hdu_main, hdu_footpoints])
        hdul.writeto(fitsname, overwrite=True)
        return

    def load_grid_from_fits(self, fitsname: str) -> None:
        print("Retrieving grid from %s" % fitsname)
        hdulist = pf.open(fitsname)
        header_main = hdulist[0].header
        self.grid_type = header_main["GRIDTYPE"]
        self.nline = header_main["NLINE"]
        self.r0 = header_main["R0"] * u.Unit(header_main["R0UNITS"])
        self.nline = header_main["NLINE"]
        if self.grid_type == "spherical":
            self.nlon = header_main["NLON"]
            self.nlat = header_main["NLAT"]
            self.pole_lat = header_main["POLELAT"] * u.degree
        if self.grid_type == "healpix":
            self.nside = header_main["NSIDE"]
        footpoint_data = CCDData.read(fitsname, hdu=1)
        self.footpoint_coordinates = CartesianRepresentation(x=footpoint_data.data)
        print("Grid type: %s" % self.grid_type)
        print("Number of field lines: %s" % self.nline)
        return

    def plot_footpoint_coordinates(self) -> None:
        footpoints_sph = self.footpoint_coordinates.represent_as(
            PhysicsSphericalRepresentation
        )
        plt.plot(footpoints_sph.phi.degree, footpoints_sph.theta.degree, "r.")

        plt.xlim(0, 360)
        plt.xlabel("Longitude, deg")
        plt.ylim(180, 0)
        plt.ylabel("Latitude, deg")
        plt.show()

        return

    def __prepare_healpix_footpoints(
        self, nside: int, r0: u.Quantity
    ) -> CartesianRepresentation:
        self.nside = nside
        self.nline = hp.nside2npix(nside)
        print(
            " Approx res at NSIDE {} is {:.2} deg ({} points)".format(
                nside, hp.nside2resol(nside, arcmin=True) / 60, self.nline
            )
        )
        theta_val, phi_val = hp.pix2ang(nside=self.nside, ipix=np.arange(self.nline))
        phi = Angle(phi_val * u.rad)
        theta = Angle(theta_val * u.rad)
        r = r0 * np.ones(self.nline)
        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=r)
        footpoints_cart = footpoints.represent_as(CartesianRepresentation)
        return footpoints_cart

    def __prepare_spherical_footpoints(
        self, nlon: int, nlat: int, r0: u.Quantity, pole_lat: Angle
    ) -> CartesianRepresentation:
        phi_grid = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        theta_grid = np.linspace(0, np.pi, nlat)

        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        phi_mesh = np.ravel(phi_mesh)
        theta_mesh = np.ravel(theta_mesh)
        duplicate = np.where(
            (phi_mesh > 0) & ((theta_mesh == 0) | (theta_mesh == np.pi))
        )
        theta_pruned = np.delete(theta_mesh, duplicate)
        phi_pruned = np.delete(phi_mesh, duplicate)
        theta = Angle(theta_pruned * u.rad)
        self.nline = len(theta_pruned)

        phi = Angle(phi_pruned * u.rad)
        r = r0 * np.ones(self.nline)
        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=r)

        footpoints_cart = footpoints.represent_as(CartesianRepresentation)
        if not (pole_lat == 0):
            rotation_operator = rot.from_euler("yzx", [pole_lat.rad, 0, 0])
            xyz = rotation_operator.apply(footpoints_cart.xyz.T) * r_ns
            footpoints_cart_rot = CartesianRepresentation(xyz.T)
            return footpoints_cart_rot
        else:
            return footpoints_cart

    def __prepare_random_footpoints(
        self, nline: int, r0: u.Quantity
    ) -> CartesianRepresentation:
        # Muller method
        self.nline = nline
        u = np.random.normal(loc=0, scale=1, size=(3, nline))
        norm = (np.sum(u**2, axis=0)) ** 0.5
        (x, y, z) = u / norm * r0
        footpoints_cart = CartesianRepresentation(x, y, z)
        return footpoints_cart


def parse_arguments():
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

    return grid_type, parameters_dict


if __name__ == "__main__":

    grid_type, parameters_dict = parse_arguments()

    grid = FootpointGrid()
    r0 = 1 * r_ns
    grid.calculate_grid(grid_type, parameters_dict, r0)
    grid.save_grid_to_fits("test.fits")
    grid.load_grid_from_fits("test.fits")
    grid.plot_footpoint_coordinates()
