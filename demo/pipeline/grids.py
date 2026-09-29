"""Build the four start grids and print how many footpoints each has."""

from psr_magpy.experiment.grid_config import (
    CustomGridConfig,
    GridConfig,
    HealpixGridConfig,
    RandomGridConfig,
    SphericalGridConfig,
)
from psr_magpy.grid.grid import CustomGrid, HealpixGrid, RandomGrid, SphericalGrid


def main() -> None:
    configs = [
        GridConfig(spherical=SphericalGridConfig(nlon=4, nlat=3, pole_lat=0.0)),
        GridConfig(healpix=HealpixGridConfig(nside=1)),
        GridConfig(random=RandomGridConfig(nline_ori=4)),
        GridConfig(
            custom=CustomGridConfig(
                nline_ori=2,
                coordinates=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            )
        ),
    ]
    for config in configs:
        grid = config.create_grid()
        kind = type(grid).__name__
        assert isinstance(grid, (SphericalGrid, HealpixGrid, RandomGrid, CustomGrid))
        print(f"{kind} nline={grid.nline}")


if __name__ == "__main__":
    main()
