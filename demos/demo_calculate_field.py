"""
Demo script showing how to use the MultipoleField class to calculate magnetic field lines
for a combination of dipole and quadrupole components.
"""
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from logging_config import setup_logging, cleanup_logging
import logging
import numpy as np
from FieldComponent import FieldComponent, MultipoleType
from FootpointGrid import FootpointGrid, GridType, SphericalGridParams
from CalculateField import MultipoleField


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

def main():
    """Run magnetic field calculation demo."""
    logger.info("Starting magnetic field calculation demonstration")
    
    try:
        # Create a spherical grid of footpoints
        logger.info("Creating spherical grid of footpoints")
        params = SphericalGridParams(
            nlon=36,  # number of longitude points
            nlat=18,  # number of latitude points
            pole_lat=70.  # latitude of the pole in degrees
        )
        grid = FootpointGrid(GridType.SPHERICAL, params)
        
        # Create field components
        logger.info("Creating field components")
        # Simple dipole
        dipole = FieldComponent(
            moment_type=MultipoleType.DIPOLE,
            B_magnitude=1.0,  # Default strength
            moment_orientation_latitude_degree=90.0,
            moment_orientation_longitude_degree=0.0
        )
        logger.debug("Created dipole component")
        
        # Offset quadrupole with custom parameters
        quadrupole = FieldComponent(
            moment_type=MultipoleType.QUADRUPOLE,
            B_magnitude=0.5,  # Half the strength of dipole
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=90.0,
            offset_x_r_ns=0.1,
            offset_y_r_ns=0.0,
            offset_z_r_ns=0.1
        )
        logger.debug("Created quadrupole component")
        
        # Combine components
        logger.info("Creating combined multipole field")
        field = MultipoleField([dipole, quadrupole])
        
        # Calculate and save field lines
        logger.info("Starting field line calculations")
        footpoints_grid, components, results = field.get_field_lines(
            footpoints_grid=grid,
            filename="demo_field_lines",
            vars_to_save="footpoint_id,R_max,R_cyl_max,z_abs_max,r_ini,field_lines",
            R_max=5.0,  # Maximum radius to trace field lines
            R_min=1.0,  # Minimum radius (NS surface)
            dt=0.01,    # Integration time step
            max_file_size_Mb=10  # Maximum size of output files
        )
        
        logger.info("Field line calculation completed successfully")
        logger.info(f"Number of field lines calculated: {len(results['footpoint_id'])}")
        logger.info(f"Maximum reached radius: {max(results['R_max']):.2f} R_NS")
        
        print("\nField line calculation completed!")
        print(f"Number of field lines calculated: {len(results['footpoint_id'])}")
        print(f"Maximum reached radius: {max(results['R_max']):.2f} R_NS")

    except Exception as e:
        logger.error(f"Error in field line calculation demo: {e}")
        raise
    finally:
        cleanup_logging()

if __name__ == "__main__":
    setup_logging(level=logging.INFO, log_file="logs/calculate_field_demo.log")
    main() 