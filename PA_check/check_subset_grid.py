"""Simple check script: calculate field lines, find OFLs, subset grid, and save it."""
import numpy as np
from pathlib import Path
from psr_magpy import *

def get_R_LC(P_ms: float) -> float:
    """Calculate light cylinder radius from period."""
    return 4.77 * P_ms

def main():
    # Field: simple dipole
    dipole_config = FieldComponentFactoryConfig(
        moment_type="dipole",
        B_magnitude=1.0,
        moment_orientation_latitude_degree=90.0,
    )
    field_factory = FieldFactoryConfig(components=[[dipole_config]])
    
    # Grid: Healpix
    grid_factory = GridFactoryConfig(grids=[HealpixGridFactoryConfig(nside=4)])
    
    # Tracing config
    tracing_config = TracingFactoryConfig(
        max_integration_radius=10.0,
        save_results_to_disk=True,
        max_memory_mb=100,
        skip_memory_estimation=True,
        field_backend="numba",
        solver_backend="scipy",
    )
    
    # Template: minimal (only what we need for OFL calculation)
    template = FieldLine(
        save_footpoint_id=True,
        save_r_cyl_max=True,
        save_z_abs_max=True,
    )
    
    # Create experiment series config
    series_config = ExperimentSeriesConfig(
        name="check_subset",
        field=field_factory,
        grid=grid_factory,
        tracing=tracing_config,
        template=template,
    )
    
    # Run experiment series
    print("Running experiment series...")
    from psr_magpy.experiment.experiment_series_runner import run_experiment_series
    series_result = run_experiment_series(
        series_config=series_config,
        save_individual_configs=True,
        save_grids=True,
    )
    
    # Get first experiment result
    if not series_result.experiment_results or not series_result.experiment_results[0].success:
        print(f"❌ Experiment failed")
        return
    
    result = series_result.experiment_results[0]
    
    # Load field line data from disk
    experiment_config = series_config.load_experiment_config(0)
    lines_dir = series_config.get_lines_dir()
    field_line_data = experiment_config.load_combined_batch_data(template, str(lines_dir))
    
    if field_line_data is None:
        print("❌ Failed to load field line data")
        return
    
    print(f"✓ Calculated {field_line_data.n_field_lines} field lines")
    
    # Find OFL footpoints
    P_ms = 1.0  # Example period
    R_LC = get_R_LC(P_ms)
    OFL_footpoints = field_line_data.get_OFL_footpoints(R_LC)
    print(f"✓ Found {len(OFL_footpoints)} OFL footpoints")
    
    # Create grid and subset it
    experiment_config = series_config.load_experiment_config(0)
    grid = experiment_config.grid.create_grid()
    if grid is None:
        print("❌ Failed to create grid")
        return
    
    print(f"✓ Created grid with {grid.nline} points")
    
    # Subset grid with OFL footpoints
    subset_grid = grid.subset_grid(OFL_footpoints)
    print(f"✓ Created subset grid with {subset_grid.nline} points")
    
    # Save subset grid
    output_dir = Path("check_subset_output")
    output_dir.mkdir(exist_ok=True)
    grid_filepath = output_dir / "OFL_subset_grid.fits"
    subset_grid.save_to_fits(str(grid_filepath))
    print(f"✓ Saved subset grid to: {grid_filepath}")
    
    # Print some info about indices
    print(f"\nGrid indices (first 10): {grid.indices[:10]}")
    print(f"Subset grid indices (first 10): {subset_grid.indices[:10]}")
    print(f"OFL footpoints (first 10): {OFL_footpoints[:10]}")
    
    


    # Create a new grid config with subset information
    from psr_magpy.experiment.experiment_config import GridConfig, HealpixGridConfig
    
    subset_healpix_config = HealpixGridConfig(
        nside=4,
        grid_file_path=str(grid_filepath),
        subset=True
    )
  
    # Create a new series config that includes subset grid information
    subset_series_config = ExperimentSeriesConfig(
        name=f"check_subset_subset",
        output_dir=str(output_dir),
        field=series_config.field,
        grid=GridConfig(healpix=subset_healpix_config),
        tracing=series_config.tracing,
        template=series_config.template,
    )
    


  
    series_result = run_experiment_series(
        series_config=subset_series_config,
        save_individual_configs=True,
        save_grids=False,
        continue_on_failure=True,
        delete_existing_lines_and_summary=True,
    )

if __name__ == "__main__":
    main()

