#!/usr/bin/env python
"""Demo for multipole magnetic field visualization."""

from pathlib import Path

# Import directly from the package
from psr_magpy.cli.setup_environment import setup_environment

# Set up environment
env = setup_environment(output_dir=Path(__file__).parent / "output")
logger = env['logger']
fits_dir = env['directories']['fits']
plot_dir = env['directories']['plots']
log_dir = env['directories']['logs']

from src.psr_magpy.experiment.experiment_config import (
    GridConfig,
    OutputConfig
)
from src.psr_magpy.line.calculator import TracingConfig
from src.psr_magpy.experiment.field_factory_config import (
    FieldComponentFactory,
    MultipoleFieldFactory
)
from src.psr_magpy.experiment.experiment_series_config import (
    ExperimentSeriesConfig,
    run_experiment_series,
    resume_experiment_series,
    analyze_experiment_log
)

def create_sample_series_config() -> ExperimentSeriesConfig:
    """Create a sample experiment series configuration."""
    
    # Create field component factories with parameter ranges
    dipole_factory = FieldComponentFactory(
        moment_type="dipole",
        B_magnitude=[0.5, 1.0],  # Multiple field strengths
        moment_orientation_latitude_degree=90,  #
        moment_orientation_longitude_degree=0
    )
    
    quadrupole_factory = FieldComponentFactory(
        moment_type="quadrupole",
        B_magnitude={"1": 0.5, "2": 1.0},
        moment_orientation_latitude_degree={"1": 45, "2": 90},
        moment_orientation_longitude_degree=0
    )
    
    # Create a multipole field factory with both components
    field_factories = MultipoleFieldFactory([dipole_factory, quadrupole_factory], [dipole_factory])
    
    # Create the experiment series configuration
    return ExperimentSeriesConfig(
        name="dipole_orientation_series",
        description="Testing different dipole orientations with fixed quadrupole",
        grid=GridConfig(grid_type="healpix", nside=4),  # Use fewer points for demo
        tracing=TracingConfig(
            R_max=10.0,
            R_min=1.0,
            dt=0.01,
            max_memory_mb=100.0,
            n_cores=2,
            max_batch_size=500,
        ),
        field_factories=field_factories,
        output=OutputConfig(
            output_dir=fits_dir,
            file_prefix="",
            save_r_max=True,
            save_b_r_ini_angle=True,
            save_r_line=True
        )
    )

def main():
    # Create configuration
    config = create_sample_series_config()
    
    # Save configuration to JSON
    config_path = f"{fits_dir}/{config.name}_series_config.json"
    config.save(config_path)
    logger.info(f"Series configuration saved to: {config_path}")
    
    # Expand to individual experiments
    experiments = config.expand_to_experiments()
    logger.info(f"Generated {len(experiments)} experiment configurations")
    
    # Print details of each experiment
    for i, exp in enumerate(experiments):
        logger.info(f"Experiment {i+1}:")
        for comp in exp.field_components:
            logger.info(f"  - {comp.type}: B={comp.B_magnitude}, lat={comp.moment_orientation_latitude_degree}°, lon={comp.moment_orientation_longitude_degree}°")
    
    # Run the experiment series with a fresh log file
    results = run_experiment_series(config, clear_log=True)
    
    # Load and verify saved configuration
    loaded_config = ExperimentSeriesConfig.load(config_path)
    experiments = loaded_config.expand_to_experiments()
    logger.info(f"Generated {len(experiments)} experiment configurations")
    logger.info("Successfully loaded series configuration")
    
    # Analyze the log file
    summary = analyze_experiment_log(config)
    logger.info(f"Found {summary['completed']} completed experiments out of {summary['total_experiments']}")
    
    # Resume the experiment series if needed
    if summary['completed'] < summary['total_experiments']:
        logger.info("Resuming experiment series...")
        results = resume_experiment_series(config)
    else:
        logger.info("All experiments have been completed.")

if __name__ == "__main__":
    main() 