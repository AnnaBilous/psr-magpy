import FootpointGrid_InputOutput as fp_io
from FootpointGrid import FootpointGrid
from astropy import units as u

r_ns = u.def_unit("Neutron star radius, 10 km", 10 * u.km)
u.add_enabled_units([r_ns])

if __name__ == "__main__":

    #Is this needed at all?
    #grid_type, parameters_dict, r0 = fp_io.get_parameters_from_command_line()
    grid_type = "random" #"healpix"
    r0 = 1 * r_ns
    parameters_dict = {"nline": 100} #{"nside": 1}
    grid = FootpointGrid(grid_type, parameters_dict, r0)

    grid.calculate_grid()
    fp_io.save_grid_to_fits(grid, "grid_test.fits")
    grid = fp_io.load_grid_from_fits("grid_test.fits")
    grid.plot_footpoint_coordinates()

    