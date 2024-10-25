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

    def __init__(self, grid_type: str, params: dict, r0: u.Quantity):
        self.grid_type = grid_type
        self.r0 = r0
        if grid_type == "spherical":
            assert "nlon" in params.keys(), "Missing nlon"
            assert "nlat" in params.keys(), "Missing nlat"
            self.nlon = int(params["nlon"])
            self.nlat = int(params["nlat"])
            self.pole_lat = params["pole_lat"]
        if grid_type == "healpix":
            assert "nside" in params.keys(), "Missing nside"
            self.nside = int(params["nside"])
        if grid_type == "random":
            assert "nline" in params.keys(), "Missing nline"
            self.nline = int(params["nline"])


    def calculate_grid(self) -> None:
        if self.grid_type == "spherical":
            footpoints = self.__prepare_spherical_footpoints(
                self.nlon, self.nlat, self.r0, self.pole_lat
            )
            print("Calculating footpoints for spherical grid")
        if self.grid_type == "healpix":
            footpoints = self.__prepare_healpix_footpoints(self.nside, self.r0)
            print("Calculating footpoints for healpix grid")
        if self.grid_type == "random":
            footpoints = self.__prepare_random_footpoints(self.nline, self.r0)
            print("Calculating footpoints for random grid")
        self.footpoint_coordinates = footpoints
        return

    def set_footpoint_coordinates(self, coordinates, unit) -> None:
        self.footpoint_coordinates = CartesianRepresentation(coordinates, unit=unit)
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


"""
if __name__ == "__main__":

    grid_type, parameters_dict = parse_arguments()

    grid = FootpointGrid()
    r0 = 1 * r_ns
    grid.calculate_grid(grid_type, parameters_dict, r0)
    grid.save_grid_to_fits("test.fits")
    grid.load_grid_from_fits("test.fits")
    grid.plot_footpoint_coordinates()
"""