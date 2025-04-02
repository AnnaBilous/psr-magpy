#!/usr/bin/env python
"""Demo for footpoint grid generation with configurable logging levels."""

from pathlib import Path
import argparse
import os

# Import directly from the package
from psr_magpy.utils.logging_config import get_logger, set_debug_for_modules
from psr_magpy.cli.setup_environment import setup_environment

def parse_args():
    """Parse command line arguments for logging configuration."""
    parser = argparse.ArgumentParser(description="Footpoint grid demo with configurable logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for all modules")
    parser.add_argument("--debug-grid", action="store_true", help="Enable debug mode for grid module only")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output (warnings and errors only)")
    parser.add_argument("--no-filename", action="store_true", help="Don't show filename in logs")
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Configure logging based on arguments
log_level = "WARNING" if args.quiet else "INFO"
debug_mode = args.debug

# Set environment variable for debug mode if specified
if debug_mode:
    os.environ["PSR_MAGPY_DEBUG"] = "1"

# Set up environment with custom logging settings
env = setup_environment(
    output_dir=Path(__file__).parent / "output",
    log_level=log_level,
    debug_mode=debug_mode,
    show_filename=not args.no_filename
)

logger = env['logger']
fits_dir = env['directories']['fits']
plot_dir = env['directories']['plots']
log_dir = env['directories']['logs']

# Enable debug for grid module only if requested
if args.debug_grid and not debug_mode:
    set_debug_for_modules(['psr_magpy.grid'], enable=True)
    logger.info("Debug logging enabled for grid module only")

import numpy as np
import matplotlib.pyplot as plt
from psr_magpy.grid.grid import HealpixGrid, SphericalGrid, RandomGrid


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
    logger.debug(f"Plotting grid points for {title}")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # Get coordinates in degrees
    lon = grid.longitude
    colat = grid.colatitude
    lat = 90 - colat  # Convert colatitude to latitude
    
    logger.debug(f"Grid coordinate ranges: lon [{lon.min():.1f}, {lon.max():.1f}], "
                f"lat [{lat.min():.1f}, {lat.max():.1f}]")
    
    # Plot points
    ax.scatter(lon, lat, s=5, alpha=0.6)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Set limits and labels
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(title)
    
    if save_path:
        logger.debug(f"Saving plot to {save_path}")
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close()


def demonstrate_healpix_grid():
    """Demonstrate HEALPix grid creation and properties."""
    logger.info("=== HEALPix Grid Demonstration ===")
    logger.info("Creating HEALPix grid")
    healpix_grid = HealpixGrid(nside=8)  # 768 points
    logger.info(f"Created HEALPix grid with {healpix_grid.nline} points")
    
    plot_grid_points(
        healpix_grid, 
        f"HEALPix Grid (nside={healpix_grid.nside})",
        plot_dir / "healpix_grid.png",
        show=False
    )
    
    # Save and load HEALPix grid
    fits_path = fits_dir / "healpix_grid.fits"
    logger.info(f"Saving HEALPix grid to {fits_path}")
    healpix_grid.save_to_fits(fits_path)
    
    logger.info(f"Loading HEALPix grid from {fits_path}")
    loaded_healpix = HealpixGrid.load_from_fits(fits_path)
    logger.info("Successfully saved and loaded HEALPix grid")
    
    return healpix_grid


def demonstrate_spherical_grid():
    """Demonstrate Spherical grid creation and properties."""
    logger.info("=== Spherical Grid Demonstration ===")
    logger.info("Creating Spherical grid")
    spherical_grid = SphericalGrid(
        nlon=24,  # 24 longitude divisions
        nlat=12,  # 12 latitude divisions
        pole_lat=45.0  # Tilt grid by 45 degrees
    )
    logger.info(f"Created Spherical grid with {spherical_grid.nline} points")
    
    plot_grid_points(
        spherical_grid, 
        f"Spherical Grid (nlon={spherical_grid.nlon}, nlat={spherical_grid.nlat})",
        plot_dir / "spherical_grid.png",
        show=False
    )
    
    return spherical_grid


def demonstrate_random_grid():
    """Demonstrate Random grid creation and properties."""
    logger.info("=== Random Grid Demonstration ===")
    logger.info("Creating Random grid")
    random_grid = RandomGrid(nline_ori=500)
    logger.info(f"Created Random grid with {random_grid.nline} points")
    
    plot_grid_points(
        random_grid, 
        f"Random Grid ({random_grid.nline} points)",
        plot_dir / "random_grid.png",
        show=False
    )
    
    return random_grid


def demonstrate_grid_subsetting(grid):
    """Demonstrate grid subsetting functionality."""
    logger.info("=== Grid Subsetting Demonstration ===")
    # Take random subset of grid
    n_subset = grid.nline // 4
    logger.info(f"Creating subset with {n_subset} points from {grid.nline} total points")
    
    subset_indices = np.random.choice(grid.nline, size=n_subset, replace=False)
    logger.debug(f"Selected {len(subset_indices)} random indices for subsetting")
    
    subset_grid = grid.subset_grid(subset_indices)
    logger.info(f"Created subset grid with {subset_grid.nline} points")
    
    plot_grid_points(
        subset_grid,
        f"Grid Subset ({subset_grid.nline} points)",
        plot_dir / "grid_subset.png",
        show=False
    )
    
    return subset_grid


def print_grid_properties(grids):
    """Print properties of multiple grids."""
    logger.info("=== Grid Properties Comparison ===")
    
    for grid, name in grids:
        logger.info(f"\n{name} Grid properties:")
        logger.info(f"Number of points: {grid.nline}")
        logger.info(f"Grid type: {grid.grid_type}")
        logger.info(f"Is partial grid: {grid.partial}")
        
        # Calculate angular separation statistics
        colat = grid.colatitude
        lon = grid.longitude
        logger.info(f"Colatitude range: {colat.min():.1f}° to {colat.max():.1f}°")
        logger.info(f"Longitude range: {lon.min():.1f}° to {lon.max():.1f}°")
        
        # More detailed statistics in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Colatitude mean: {np.mean(colat):.2f}°, std: {np.std(colat):.2f}°")
            logger.debug(f"Longitude mean: {np.mean(lon):.2f}°, std: {np.std(lon):.2f}°")
            
            # Calculate nearest neighbor distances (computationally expensive)
            logger.debug("Calculating nearest neighbor statistics...")
            xyz = grid.footpoint_coordinates.xyz.value.T
            from scipy.spatial import cKDTree
            tree = cKDTree(xyz)
            distances, _ = tree.query(xyz, k=2)  # k=2 to get nearest non-self neighbor
            min_dist = np.min(distances[:, 1])
            max_dist = np.max(distances[:, 1])
            mean_dist = np.mean(distances[:, 1])
            logger.debug(f"Nearest neighbor distances: min={min_dist:.4f}, max={max_dist:.4f}, mean={mean_dist:.4f}")


def main():
    """Run the footpoint grid demonstration with various logging levels."""
    logger.info("Starting footpoint grid demonstration")
    logger.debug("Debug logging is enabled")
    
    # Import logging for level checks
    import logging
    
    # Create different grid types
    healpix_grid = demonstrate_healpix_grid()
    spherical_grid = demonstrate_spherical_grid()
    random_grid = demonstrate_random_grid()
    
    # Demonstrate grid subsetting
    subset_grid = demonstrate_grid_subsetting(healpix_grid)
    
    # Print grid properties
    print_grid_properties([
        (healpix_grid, "HEALPix"),
        (spherical_grid, "Spherical"),
        (random_grid, "Random"),
        (subset_grid, "Subset")
    ])
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    try:
        # Import logging here to use in print_grid_properties
        import logging
        main()
    except Exception as e:
        logger.exception("Demo failed with error")
        raise
