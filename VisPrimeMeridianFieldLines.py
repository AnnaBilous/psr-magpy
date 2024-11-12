import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from FootpointGrid import FootpointGrid
from CalculateField import MultipoleField
from astropy import units as u
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation


r_ns = u.def_unit("Neutron star radius, 10 km", 10 * u.km)
u.add_enabled_units([r_ns])


def plot_prime_meridian_field_lines(
    nlat: int,
    field_components: dict,
    ax: mpl.axes.Axes,
    R_max=10.0,
    arrows_loc=[3.0, 7.0],
    lc="k",
    lw=0.2,
    arrow_width=0.1,
):

    grid_type = "spherical"
    r0 = 1 * r_ns
    parameters_dict = {"nlon": 2, "nlat": nlat}
    grid = FootpointGrid(grid_type, parameters_dict, r0)
    grid.calculate_grid()

    magnetic_field = MultipoleField()
    magnetic_field.initialize(field_components)
    grid, field_components, stat_dict = magnetic_field.get_field_lines(
        grid,
        vars_to_save="footpoint_id, r_ini, B_R_ini_angle, field_lines",
        R_max=R_max,
        save_file=False,
    )

    field_lines = stat_dict["field_lines"]

    for i in range(len(field_lines)):
        fl_cart = CartesianRepresentation(x=field_lines[i].T)
        fl_sph = PhysicsSphericalRepresentation.from_cartesian(fl_cart)
        r = fl_sph.r
        K = int(np.sign(np.cos(stat_dict["B_R_ini_angle"][i])))

        # Plotting field lines
        ax.plot(fl_cart.x, fl_cart.z, color=lc, lw=lw)
        # Plotting arrows

        for r_arr in arrows_loc:
            eps = np.mean(np.abs(np.diff(r)))
            ind = np.arange(len(r))
            ind_ar = ind[np.abs(r - r_arr) < 0.51 * eps]
            if len(ind_ar) > 0:
                for indc in ind_ar:
                    x = fl_cart.x[indc]
                    y = fl_cart.z[indc]
                    dx = fl_cart.x[indc + K] - fl_cart.x[indc]
                    dy = fl_cart.z[indc + K] - fl_cart.z[indc]
                    arrow = patches.FancyArrow(
                        x,
                        y,
                        dx,
                        dy,
                        color=lc,
                        width=0,
                        head_width=arrow_width,
                        head_length=2 * arrow_width,
                        length_includes_head=True,
                        overhang=0.3,
                        zorder=10,
                    )
                    ax.add_patch(arrow)
    return
