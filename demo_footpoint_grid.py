from logging_config import setup_logging, cleanup_logging
import logging
import numpy as np
import matplotlib.pyplot as plt
from FootpointGrid import (
    FootpointGrid, 
    GridType, 
    SphericalGridParams,
    HealpixGridParams,
    RandomGridParams
)


# Create logger for this script
logger = logging.getLogger(__name__)

def demo_spherical_grid():
    """Demonstrate spherical grid creation and manipulation."""
    logger.info("Starting spherical grid demo")
    
    try:
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Demo 1: Standard grid
        logger.info("Creating standard spherical grid")
        params_standard = SphericalGridParams(nlon=36, nlat=18, pole_lat=0)
        grid_standard = FootpointGrid(GridType.SPHERICAL, params_standard)
        
        # Plot standard grid
        grid_standard.plot_footpoint_coordinates(ax=ax1, color='blue', label='Full Grid')
        
        # Create and plot subset of standard grid
        random_indices = np.random.choice(grid_standard.nline_ori, size=100, replace=False)
        grid_standard.subset_grid(random_indices)
        grid_standard.plot_footpoint_coordinates(ax=ax1, color='red', label='Subset')
        
        ax1.legend()
        ax1.set_title("Standard Spherical Grid")
        
        # Demo 2: Rotated grid
        logger.info("Creating rotated spherical grid (pole_lat=45°)")
        params_rotated = SphericalGridParams(nlon=36, nlat=18, pole_lat=45)
        grid_rotated = FootpointGrid(GridType.SPHERICAL, params_rotated)
        
        # Plot rotated grid
        grid_rotated.plot_footpoint_coordinates(ax=ax2, color='green', label='Full Grid')
        
        # Create and plot subset of rotated grid
        random_indices = np.random.choice(grid_rotated.nline_ori, size=100, replace=False)
        grid_rotated.subset_grid(random_indices)
        grid_rotated.plot_footpoint_coordinates(ax=ax2, color='orange', label='Subset')
        
        ax2.legend()
        ax2.set_title("Rotated Spherical Grid (45°)")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig("spherical_grid_demo.png")
        logger.info("Successfully saved spherical grid plots")
        
    except Exception as e:
        logger.error(f"Error in spherical grid demo: {e}")
        raise

def demo_healpix_grid():
    """Demonstrate HEALPix grid creation and manipulation."""
    logger.info("Starting HEALPix grid demo")
    
    try:
        # Create HEALPix grid
        params = HealpixGridParams(nside=4)
        grid = FootpointGrid(GridType.HEALPIX, params)
        
        # Plot the full grid
        fig, ax = plt.subplots()
        grid.plot_footpoint_coordinates(ax=ax, color='green', label='Full Grid')
        
        # Create and plot a subset
        subset_size = grid.nline_ori // 2  # Take half of the points
        random_indices = np.random.choice(grid.nline_ori, size=subset_size, replace=False)
        grid.subset_grid(random_indices)
        grid.plot_footpoint_coordinates(ax=ax, color='orange', label='Subset')
        
        ax.legend()
        plt.title("HEALPix Grid Demo")
        plt.savefig("healpix_grid_demo.png")
        logger.info("Successfully saved HEALPix grid plot")
        
    except Exception as e:
        logger.error(f"Error in HEALPix grid demo: {e}")
        raise

def demo_random_grid():
    """Demonstrate random grid creation and manipulation."""
    logger.info("Starting random grid demo")
    
    try:
        # Create random grid
        params = RandomGridParams(nline_ori=200)
        grid = FootpointGrid(GridType.RANDOM, params)
        
        # Plot the full grid
        fig, ax = plt.subplots()
        grid.plot_footpoint_coordinates(ax=ax, color='purple', label='Full Grid')
        
        # Create and plot a subset
        subset_indices = np.random.choice(grid.nline_ori, size=50, replace=False)
        grid.subset_grid(subset_indices)
        grid.plot_footpoint_coordinates(ax=ax, color='magenta', label='Subset')
        
        ax.legend()
        plt.title("Random Grid Demo")
        plt.savefig("random_grid_demo.png")
        logger.info("Successfully saved random grid plot")
        
    except Exception as e:
        logger.error(f"Error in random grid demo: {e}")
        raise

def main():
    """Run all grid demos."""
    logger.info("Starting FootpointGrid demonstrations")
    
    try:
        setup_logging(level=logging.INFO, log_file="logs/app.log")
        demo_spherical_grid()
        demo_healpix_grid()
        demo_random_grid()
        plt.show()
        logger.info("All demonstrations completed successfully")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        cleanup_logging()
        plt.close('all')

if __name__ == "__main__":
    main() 