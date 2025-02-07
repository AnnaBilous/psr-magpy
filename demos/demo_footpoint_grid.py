import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from logging_config import setup_logging, cleanup_logging
import logging
import numpy as np
import matplotlib.pyplot as plt
from FootpointGrid import (FootpointGrid, SphericalGrid, HealpixGrid, RandomGrid)


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
        grid_standard = SphericalGrid(nlon=36, nlat=18, pole_lat=70.)
        
        # Save standard grid
        grid_standard.save_to_fits(fits_dir / "spherical_standard.fits")
        logger.info("Saved standard grid")
        
        # Load standard grid
        loaded_standard = FootpointGrid.load_from_fits(fits_dir / "spherical_standard.fits")
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
        random_indices = np.random.choice(loaded_standard.nline_ori, size=400, replace=False)
        subset_grid = loaded_standard.subset_grid(random_indices)
        subset_grid.save_to_fits(fits_dir / "spherical_subset.fits")
        logger.info("Saved subset grid")
        
        # Load and plot subset
        subset_grid = FootpointGrid.load_from_fits(fits_dir / "spherical_subset.fits")
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
    """Demonstrate healpix grid creation, manipulation, and I/O."""
    logger.info("Starting healpix grid demo")
    
    try:
        # Create and plot standard grid
        grid_standard = HealpixGrid(nside=8)  # NSIDE=8 gives 768 pixels
        
        # Save standard grid
        grid_standard.save_to_fits(fits_dir / "healpix_standard.fits")
        logger.info("Saved standard grid")
        
        # Load standard grid
        loaded_standard = FootpointGrid.load_from_fits(fits_dir / "healpix_standard.fits")
        logger.info("Loaded standard grid")
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original grid
        grid_standard.plot_footpoint_coordinates(ax=ax1, color='blue', label='Original')
        ax1.set_title("Original HEALPix Grid")
        ax1.legend()
        
        # Plot loaded grid
        loaded_standard.plot_footpoint_coordinates(ax=ax2, color='green', label='Loaded')
        ax2.set_title("Loaded HEALPix Grid")
        ax2.legend()
        
        # Create and save subset
        random_indices = np.random.choice(loaded_standard.nline_ori, size=200, replace=False)
        subset_grid = loaded_standard.subset_grid(random_indices)
        subset_grid.save_to_fits(fits_dir / "healpix_subset.fits")
        logger.info("Saved subset grid")
        
        # Load and plot subset
        subset_grid = FootpointGrid.load_from_fits(fits_dir / "healpix_subset.fits")
        subset_grid.plot_footpoint_coordinates(ax=ax3, color='red', label='Subset')
        ax3.set_title("Subset HEALPix Grid")
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / "healpix_grid_demo.png")
        logger.info("Successfully saved healpix grid plots")
        
    except Exception as e:
        logger.error(f"Error in healpix grid demo: {e}")
        raise

def demo_random_grid():
    """Demonstrate random grid creation, manipulation, and I/O."""
    logger.info("Starting random grid demo")
    
    try:
        # Create and plot standard grid
        grid_standard = RandomGrid(nline_ori=1000)  # Create 1000 random points
        
        # Save standard grid
        grid_standard.save_to_fits(fits_dir / "random_standard.fits")
        logger.info("Saved standard grid")
        
        # Load standard grid
        loaded_standard = FootpointGrid.load_from_fits(fits_dir / "random_standard.fits")
        logger.info("Loaded standard grid")
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original grid
        grid_standard.plot_footpoint_coordinates(ax=ax1, color='blue', label='Original')
        ax1.set_title("Original Random Grid")
        ax1.legend()
        
        # Plot loaded grid
        loaded_standard.plot_footpoint_coordinates(ax=ax2, color='green', label='Loaded')
        ax2.set_title("Loaded Random Grid")
        ax2.legend()
        
        # Create and save subset
        random_indices = np.random.choice(loaded_standard.nline_ori, size=300, replace=False)
        subset_grid = loaded_standard.subset_grid(random_indices)
        subset_grid.save_to_fits(fits_dir / "random_subset.fits")
        logger.info("Saved subset grid")
        
        # Load and plot subset
        subset_grid = FootpointGrid.load_from_fits(fits_dir / "random_subset.fits")
        subset_grid.plot_footpoint_coordinates(ax=ax3, color='red', label='Subset')
        ax3.set_title("Subset Random Grid")
        ax3.legend()
        
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
        
        # Run all demos
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