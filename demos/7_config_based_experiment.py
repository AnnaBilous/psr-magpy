from utils import setup_demo
fits_dir, plot_dir, log_dir, logger = setup_demo(__file__)


from src.experiment.experiment_config import (
    ExperimentConfig, 
    GridConfig,
    FieldComponentConfig,
    OutputConfig
)
from src.line.calculator import TracingConfig


def create_sample_config() -> ExperimentConfig:
    """Create a sample experiment configuration."""
    return ExperimentConfig(
        name="dipole_quadrupole_test",
        description="Test experiment with dipole and quadrupole fields",
        grid=GridConfig(grid_type="random", n_random=100),
        tracing=TracingConfig(
            R_max=10.0,
            R_min=1.0,
            dt=0.01,
            max_memory_mb=100.0,
            n_cores=2,
            max_batch_size=100,
            save_results=True,
            filename="dipole_quadrupole_test"
        ),
        field_components=[
            FieldComponentConfig(
                type="dipole",
                B_magnitude=1.0,
                moment_orientation_latitude_degree=0.0,
                moment_orientation_longitude_degree=0.0
            ),
            FieldComponentConfig(
                type="quadrupole",
                B_magnitude=0.5,
                moment_orientation_latitude_degree=45.0,
                moment_orientation_longitude_degree=0.0
            )
        ],
        output=OutputConfig(
            output_dir=fits_dir,
            save_r_max=True,
            save_b_r_ini_angle=True,
            save_r_line=True
        )
    )

def main():
    # Create configuration
    config = create_sample_config()
    
    # Save configuration to JSON
    config_path = f"{fits_dir}/{config.name}_config.json"
    config.save(config_path)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Run the experiment
    #results = run_experiment(config)
    
    # Load and verify saved configuration
    loaded_config = ExperimentConfig.load(config_path)
    logger.info("Successfully loaded configuration")

if __name__ == "__main__":
    main() 