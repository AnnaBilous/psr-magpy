import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation
from astropy.coordinates import Angle
from scipy.spatial.transform import Rotation as rot

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
            assert "nlon" in params.keys(), "Missing nlon for spherical grid"
            assert "nlat" in params.keys(), "Missing nlat for spherical grid"
            self.nlon = int(params["nlon"])
            self.nlat = int(params["nlat"])
            if "pole_lat" in params.keys():
                self.pole_lat = params["pole_lat"]
            else:
                self.pole_lat = 0
        if grid_type == "healpix":
            assert "nside" in params.keys(), "Missing nside for healpix grid"
            self.nside = int(params["nside"])
        if grid_type == "random":
            assert "nline_ori" in params.keys(), "Missing nline_ori for random grid"
            self.nline_ori = int(params["nline_ori"])
        self.partial = False

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
            footpoints = self.__prepare_random_footpoints(self.nline_ori, self.r0)
            print("Calculating footpoints for random grid")
        self.footpoint_coordinates = footpoints
        self.indices = np.arange(self.nline_ori)
        self.nline = self.nline_ori
        return

    def subset_grid(self, grid_indices: np.array) -> None:
        not_allowed_indices = set(grid_indices) - set(self.indices)
        assert len(not_allowed_indices) == 0, print(
            f"Following indices are not present in this grid: {not_allowed_indices}."
        )
        self.partial = True
        self.footpoint_coordinates = self.footpoint_coordinates[grid_indices]
        self.indices = grid_indices
        self.nline = len(grid_indices)
        print(f"Subsetting grid from {self.nline_ori} to {self.nline} indices")
        return

    def plot_footpoint_coordinates(self) -> None:
        # Rename footpoint_coordinates to just coordinates
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
        self.nline_ori = hp.nside2npix(nside)
        print(
            " Approx res at NSIDE {} is {:.2} deg ({} points)".format(
                nside, hp.nside2resol(nside, arcmin=True) / 60, self.nline_ori
            )
        )
        theta_val, phi_val = hp.pix2ang(
            nside=self.nside, ipix=np.arange(self.nline_ori)
        )
        phi = Angle(phi_val * u.rad)
        theta = Angle(theta_val * u.rad)
        r = r0 * np.ones(self.nline_ori)
        footpoints = PhysicsSphericalRepresentation(phi, theta=theta, r=r)
        footpoints_cart = footpoints.represent_as(CartesianRepresentation)
        return footpoints_cart

    def __prepare_spherical_footpoints(
        self, nlon: int, nlat: int, r0: u.Quantity, pole_lat: Angle
    ) -> CartesianRepresentation:
        phi_grid = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
        theta_grid = np.linspace(0, np.pi, nlat)
        print(
            f"Latitude respolution {np.rad2deg(theta_grid[1] - theta_grid[0])} degrees"
        )

        phi_mesh, theta_mesh = np.meshgrid(phi_grid, theta_grid)
        phi_mesh = np.ravel(phi_mesh)
        theta_mesh = np.ravel(theta_mesh)
        duplicate = np.where(
            (phi_mesh > 0) & ((theta_mesh == 0) | (theta_mesh == np.pi))
        )
        theta_pruned = np.delete(theta_mesh, duplicate)
        phi_pruned = np.delete(phi_mesh, duplicate)
        theta = Angle(theta_pruned * u.rad)
        self.nline_ori = len(theta_pruned)

        phi = Angle(phi_pruned * u.rad)
        r = r0 * np.ones(self.nline_ori)
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
        self, nline_ori: int, r0: u.Quantity
    ) -> CartesianRepresentation:
        # Muller method
        self.nline_ori = nline_ori
        u = np.random.normal(loc=0, scale=1, size=(3, nline_ori))
        norm = (np.sum(u**2, axis=0)) ** 0.5
        (x, y, z) = u / norm * r0
        footpoints_cart = CartesianRepresentation(x, y, z)
        return footpoints_cart
