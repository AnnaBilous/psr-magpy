import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
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

# Create output directories relative to script location
script_dir = Path(__file__).parent
output_dir = script_dir / "output"
fits_dir = output_dir / "fits"
plot_dir = output_dir / "plots"
log_dir = output_dir / "logs"

for directory in [fits_dir, plot_dir, log_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Create logger for this script
logger = logging.getLogger(__name__)

def demo_spherical_grid():
    """Demonstrate spherical grid creation, manipulation, and I/O."""
    logger.info("Starting spherical grid demo")
    
    try:
        # Create and plot standard grid
        params_standard = SphericalGridParams(nlon=36, nlat=18, pole_lat=70.)
        grid_standard = FootpointGrid(GridType.SPHERICAL, params_standard)
        
        # Save standard grid
        grid_standard.save_to_fits(fits_dir / "spherical_standard.fits")
        logger.info("Saved standard grid")
        
        # Load standard grid
        loaded_standard = FootpointGrid.from_fits(fits_dir / "spherical_standard.fits")
        logger.info("Loaded standard grid")
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original grid
        grid_standard.plot_footpoint_coordinates(ax=ax1, color='blue', label='Original')
        ax1.set_title("Original Grid")
        ax1.legend()
        
        # Plot loaded grid
        loaded_standard.plot_footpoint_coordinates(ax=ax2, color='green', label='Loaded')
        ax2.set_title("Loaded Grid")
        ax2.legend()
        
        # Create and save subset
        random_indices = np.random.choice(loaded_standard.nline_ori, size=100, replace=False)
        loaded_standard.subset_grid(random_indices)
        loaded_standard.save_to_fits(fits_dir / "spherical_subset.fits")
        logger.info("Saved subset grid")
        
        # Load and plot subset
        subset_grid = FootpointGrid.from_fits(fits_dir / "spherical_subset.fits")
        subset_grid.plot_footpoint_coordinates(ax=ax3, color='red', label='Subset')
        ax3.set_title("Subset Grid")
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / "spherical_grid_demo.png")
        logger.info("Successfully saved spherical grid plots")
        
    except Exception as e:
        logger.error(f"Error in spherical grid demo: {e}")
        raise

def demo_healpix_grid():
    """Demonstrate HEALPix grid creation, manipulation, and I/O."""
    logger.info("Starting HEALPix grid demo")
    
    try:
        # Create HEALPix grid
        params = HealpixGridParams(nside=4)
        grid = FootpointGrid(GridType.HEALPIX, params)
        
        # Save full grid
        grid.save_to_fits(fits_dir / "healpix_full.fits")
        logger.info("Saved full HEALPix grid")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original grid
        grid.plot_footpoint_coordinates(ax=ax1, color='green', label='Full Grid')
        
        # Create and save subset
        subset_size = grid.nline_ori // 2
        random_indices = np.random.choice(grid.nline_ori, size=subset_size, replace=False)
        grid.subset_grid(random_indices)
        grid.save_to_fits(fits_dir / "healpix_subset.fits")
        logger.info("Saved subset HEALPix grid")
        
        # Load and plot subset
        subset_grid = FootpointGrid.from_fits(fits_dir / "healpix_subset.fits")
        subset_grid.plot_footpoint_coordinates(ax=ax2, color='orange', label='Loaded Subset')
        
        ax1.set_title("Full Grid")
        ax2.set_title("Loaded Subset")
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / "healpix_grid_demo.png")
        logger.info("Successfully saved HEALPix grid plots")
        
    except Exception as e:
        logger.error(f"Error in HEALPix grid demo: {e}")
        raise

def demo_random_grid():
    """Demonstrate random grid creation, manipulation, and I/O."""
    logger.info("Starting random grid demo")
    
    try:
        # Create random grid
        params = RandomGridParams(nline_ori=200)
        grid = FootpointGrid(GridType.RANDOM, params)
        
        # Save full grid
        grid.save_to_fits(fits_dir / "random_full.fits")
        logger.info("Saved full random grid")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original grid
        grid.plot_footpoint_coordinates(ax=ax1, color='purple', label='Full Grid')
        
        # Create and save subset
        subset_indices = np.random.choice(grid.nline_ori, size=50, replace=False)
        grid.subset_grid(subset_indices)
        grid.save_to_fits(fits_dir / "random_subset.fits")
        logger.info("Saved subset random grid")
        
        # Load and plot subset
        subset_grid = FootpointGrid.from_fits(fits_dir / "random_subset.fits")
        subset_grid.plot_footpoint_coordinates(ax=ax2, color='magenta', label='Loaded Subset')
        
        ax1.set_title("Full Grid")
        ax2.set_title("Loaded Subset")
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / "random_grid_demo.png")
        logger.info("Successfully saved random grid plots")
        
    except Exception as e:
        logger.error(f"Error in random grid demo: {e}")
        raise

def main():
    """Run all grid demos."""
    logger.info("Starting FootpointGrid demonstrations")
    
    try:
        setup_logging(
            level=logging.INFO,
            log_file=log_dir / "footpoint_grid_demo.log"
        )
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