import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
import numpy as np
from FieldLineCalculator import FieldLineCalculator, TracingConfig
from FieldLineDefinition import FieldLine
from MultipoleField import MultipoleField
from FieldComponent import FieldComponent, MultipoleType
from FootpointGrid import FootpointGrid, GridType, HealpixGridParams, SphericalGridParams

script_dir = Path(__file__).parent

def run_healpix_demo():
    """Demo using healpix grid with dipole and quadrupole field."""
    print("\n=== Running Healpix Grid Demo ===")
    
    # Create magnetic field components
    dipole = FieldComponent(
        moment_type=MultipoleType.DIPOLE,
        B_magnitude=1.0,
        moment_orientation_latitude_degree=90.0,
    )
    quadrupole = FieldComponent(
        moment_type=MultipoleType.QUADRUPOLE,
        B_magnitude=3.0
    )
    field = MultipoleField([dipole, quadrupole])

    # Create footpoint grid
    params_healpix = HealpixGridParams(nside=32)  # 12288 points
    grid = FootpointGrid(grid_type=GridType.HEALPIX, params=params_healpix)
    print(f"Number of field lines to calculate: {grid.nline}")

    # Configure field line template
    template = FieldLine(
        save_r_max=True,
        save_b_r_ini_angle=True,
        save_r_line=True
    )

    # Initialize calculator
    calculator = FieldLineCalculator(field, grid)

    # Set up tracing configuration
    config = TracingConfig(
        R_max=10.0,
        R_min=1.0,
        dt=0.01,
        max_memory_mb=100.0,
        n_cores=4,
        save_results=True,
        filename=script_dir/"output/fits/demo_healpix_output"
    )

    # Calculate field lines
    results = calculator.get_field_lines(
        template=template,
        config=config
    )

def run_spherical_demo():
    """Demo using spherical grid with pure dipole field."""
    print("\n=== Running Spherical Grid Demo ===")
    
    # Create simple dipole field
    dipole = FieldComponent(
        moment_type=MultipoleType.DIPOLE,
        B_magnitude=1.0
    )
    field = MultipoleField([dipole])

    # Create spherical grid
    params_spherical = SphericalGridParams(
        n_latitude=16,
        n_longitude=32,
        min_latitude=-60,
        max_latitude=60
    )
    grid = FootpointGrid(grid_type=GridType.SPHERICAL, params=params_spherical)
    print(f"Number of field lines to calculate: {grid.nline}")

    # Configure field line template
    template = FieldLine(save_r_line=True)

    # Initialize calculator
    calculator = FieldLineCalculator(field, grid)

    # Set up tracing configuration
    config = TracingConfig(
        R_max=5.0,
        R_min=1.0,
        dt=0.01,
        max_memory_mb=500.0,
        n_cores=2,
        save_results=False  # Keep results in memory
    )

    # Calculate field lines
    results = calculator.get_field_lines(
        template=template,
        config=config
    )
    
    if results:
        print(f"\nResults summary:")
        print(f"Number of field lines: {len(results.field_lines)}")
        r_max_values = [fl.r_max for fl in results.field_lines if fl.r_max is not None]
        if r_max_values:
            print(f"Average maximum radius: {np.mean(r_max_values):.2f}")

def main():
    print("Field Line Calculator Demo")
    print("=========================")
    
    try:
        run_healpix_demo()
        run_spherical_demo()
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        raise

if __name__ == "__main__":
    main() 