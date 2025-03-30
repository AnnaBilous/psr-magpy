#!/usr/bin/env python
"""Demo for footpoint grid generation."""

import argparse
from pathlib import Path
import sys

# Import directly from the package
from psr_magpy.cli.setup_environment import setup_environment

# Set up environment
env = setup_environment(output_dir=Path(__file__).parent / "output")
logger = env["logger"]
fits_dir = env["directories"]["fits"]
plot_dir = env["directories"]["plots"]
log_dir = env["directories"]["logs"]

import numpy as np
import matplotlib.pyplot as plt
from src.psr_magpy.grid.grid import HealpixGrid, SphericalGrid, RandomGrid


def plot_grid_points(grid, title, save_path=None, show=True):
    """Plot grid points in longitude-latitude projection.

    Parameters
    ----------
    grid : FootpointGrid
        Grid to plot
    title : str
        Plot title
    save_path : Path, optional
        Path to save the plot
    show : bool, optional
        Whether to display the plot, by default True
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # Get coordinates in degrees
    lon = grid.longitude
    colat = grid.colatitude
    lat = 90 - colat  # Convert colatitude to latitude

    # Plot points
    ax.scatter(lon, lat, s=5, alpha=0.6)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.5)

    # Set limits and labels
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close()


def main():
    # 1. Demonstrate HEALPix grid
    logger.info("Creating HEALPix grid")
    healpix_grid = HealpixGrid(nside=8)  # 768 points
    logger.info(f"Created HEALPix grid with {healpix_grid.nline} points")
    plot_grid_points(
        healpix_grid,
        f"HEALPix Grid (nside={healpix_grid.nside})",
        plot_dir / "healpix_grid.png",
    )

    # Save and load HEALPix grid
    fits_path = fits_dir / "healpix_grid.fits"
    healpix_grid.save_to_fits(fits_path)
    loaded_healpix = HealpixGrid.load_from_fits(fits_path)
    logger.info("Successfully saved and loaded HEALPix grid")

    # 2. Demonstrate Spherical grid
    logger.info("Creating Spherical grid")
    spherical_grid = SphericalGrid(
        nlon=24,  # 24 longitude divisions
        nlat=12,  # 12 latitude divisions
        pole_lat=45.0,  # Tilt grid by 45 degrees
    )
    logger.info(f"Created Spherical grid with {spherical_grid.nline} points")
    plot_grid_points(
        spherical_grid,
        f"Spherical Grid (nlon={spherical_grid.nlon}, nlat={spherical_grid.nlat})",
        plot_dir / "spherical_grid.png",
    )

    # 3. Demonstrate Random grid
    logger.info("Creating Random grid")
    random_grid = RandomGrid(nline_ori=500)
    logger.info(f"Created Random grid with {random_grid.nline} points")
    plot_grid_points(
        random_grid,
        f"Random Grid ({random_grid.nline} points)",
        plot_dir / "random_grid.png",
    )

    # 4. Demonstrate grid subsetting
    logger.info("Demonstrating grid subsetting")
    # Take random subset of HEALPix grid
    n_subset = healpix_grid.nline // 4
    subset_indices = np.random.choice(healpix_grid.nline, size=n_subset, replace=False)
    subset_grid = healpix_grid.subset_grid(subset_indices)
    logger.info(f"Created subset grid with {subset_grid.nline} points")
    plot_grid_points(
        subset_grid,
        f"HEALPix Grid Subset ({subset_grid.nline} points)",
        plot_dir / "healpix_subset.png",
    )

    # 5. Demonstrate grid properties
    logger.info("Demonstrating grid properties")
    for grid, name in [
        (healpix_grid, "HEALPix"),
        (spherical_grid, "Spherical"),
        (random_grid, "Random"),
    ]:
        logger.info(f"\n{name} Grid properties:")
        logger.info(f"Number of points: {grid.nline}")
        logger.info(f"Grid type: {grid.grid_type}")
        logger.info(f"Is partial grid: {grid.partial}")

        # Calculate angular separation statistics
        colat = grid.colatitude
        lon = grid.longitude
        logger.info(f"Colatitude range: {colat.min():.1f}° to {colat.max():.1f}°")
        logger.info(f"Longitude range: {lon.min():.1f}° to {lon.max():.1f}°")


if __name__ == "__main__":
    try:
        main()
        logger.info("Demo completed successfully")
    except Exception as e:
        logger.exception("Demo failed with error")
        raise
