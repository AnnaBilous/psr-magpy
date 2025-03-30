import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
from pathlib import Path

from src.psr_magpy.grid.grid import GridType, HealpixGrid, SphericalGrid, RandomGrid
from src.psr_magpy.field.components import (
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
)
from src.psr_magpy.line.calculator import TracingConfig
from src.psr_magpy.line.field_line import PropertyName


@dataclass
class GridConfig:
    """Configuration for the field line grid."""

    grid_type: Union[GridType, str]

    # HEALPix parameters
    nside: Optional[int] = None

    # Spherical grid parameters
    nlon: Optional[int] = None
    nlat: Optional[int] = None
    pole_lat: float = 0.0

    # Random grid parameters
    n_random: Optional[int] = None

    def __post_init__(self):
        """Convert grid_type from string to GridType if necessary."""
        if isinstance(self.grid_type, str):
            try:
                self.grid_type = GridType(self.grid_type.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid grid type: {self.grid_type}. Must be one of {[e.value for e in GridType]}"
                )

    def create_grid(self) -> Union["HealpixGrid", "SphericalGrid", "RandomGrid"]:
        """Create a grid based on the configuration."""
        if self.grid_type == GridType.HEALPIX:
            if self.nside is None:
                raise ValueError("nside must be specified for healpix grid")
            return HealpixGrid(nside=self.nside)
        elif self.grid_type == GridType.SPHERICAL:
            if self.nlon is None or self.nlat is None:
                raise ValueError("nlon and nlat must be specified for spherical grid")
            return SphericalGrid(nlon=self.nlon, nlat=self.nlat, pole_lat=self.pole_lat)
        elif self.grid_type == GridType.RANDOM:
            if self.n_random is None:
                raise ValueError("n_random must be specified for random grid")
            return RandomGrid(nline_ori=self.n_random)
        else:
            raise ValueError(f"Unknown grid type: {self.grid_type}")


@dataclass
class FieldComponentConfig:
    """Configuration for a magnetic field component."""

    type: str  # 'dipole' or 'quadrupole'
    B_magnitude: float
    moment_orientation_latitude_degree: float
    moment_orientation_longitude_degree: float

    def create_component(self) -> Union[DipoleFieldComponent, QuadrupoleFieldComponent]:
        """Create a field component based on the configuration."""
        if self.type == "dipole":
            return DipoleFieldComponent(
                B_magnitude=self.B_magnitude,
                moment_orientation_latitude_degree=self.moment_orientation_latitude_degree,
                moment_orientation_longitude_degree=self.moment_orientation_longitude_degree,
            )
        elif self.type == "quadrupole":
            return QuadrupoleFieldComponent(
                B_magnitude=self.B_magnitude,
                moment_orientation_latitude_degree=self.moment_orientation_latitude_degree,
                moment_orientation_longitude_degree=self.moment_orientation_longitude_degree,
            )
        else:
            raise ValueError(f"Unknown field component type: {self.type}")


class OutputConfig:
    """Configuration for output settings."""

    def __init__(self, output_dir="output", file_prefix="", **kwargs):
        """Initialize with output directory and optional save flags.

        Args:
            output_dir: Directory to save output files
            file_prefix: Prefix to add to output filenames
            **kwargs: Save flags (e.g., save_r_max=True)
        """
        self.output_dir = output_dir
        self.file_prefix = file_prefix

        # Set default values for all save flags
        for prop in PropertyName:
            # FOOTPOINT_ID is always True, others default to False
            default_value = prop == PropertyName.FOOTPOINT_ID
            setattr(self, prop.flag_name, default_value)

        # Update with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """Convert to dictionary for serialization."""
        result = {"output_dir": self.output_dir, "file_prefix": self.file_prefix}

        # Add all save flags
        for prop in PropertyName:
            result[prop.flag_name] = getattr(self, prop.flag_name)

        return result

    def get_save_flags(self):
        """Get a dictionary of all save flags.

        Returns:
            dict: Dictionary of save flags for use with FieldLine constructor
        """
        save_flags = {}
        for prop in PropertyName:
            flag_name = prop.flag_name
            save_flags[flag_name] = getattr(self, flag_name)
        return save_flags

    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary."""
        return cls(**data)

    def get_output_filename(self, extension=""):
        if self.file_prefix:
            filename = f"{self.file_prefix}_{extension}"
        else:
            filename = f"{extension}"

        return Path(self.output_dir) / filename


@dataclass
class ExperimentConfig:
    """Main configuration class for magnetic field line experiments."""

    # Experiment metadata
    name: str
    description: Optional[str] = None

    # Component configurations
    grid: GridConfig = None
    tracing: TracingConfig = None
    field_components: List[FieldComponentConfig] = None

    # Output configuration
    output: OutputConfig = OutputConfig()

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert field components data to proper config objects
        if "field_components" in data:
            data["field_components"] = [
                FieldComponentConfig(**comp) for comp in data["field_components"]
            ]

        # Convert nested configs
        data["grid"] = GridConfig(**data["grid"])
        data["tracing"] = TracingConfig(**data["tracing"])
        data["output"] = OutputConfig.from_dict(data["output"])

        return cls(**data)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        # Convert to dictionary
        config_dict = asdict(self)

        # Convert grid_type to string
        if isinstance(config_dict["grid"]["grid_type"], GridType):
            config_dict["grid"]["grid_type"] = config_dict["grid"]["grid_type"].value

        # Handle OutputConfig serialization
        config_dict["output"] = self.output.to_dict()

        # Convert output_dir Path to string if needed
        if isinstance(config_dict["output"]["output_dir"], Path):
            config_dict["output"]["output_dir"] = str(
                config_dict["output"]["output_dir"]
            )

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=4)


def main():
    """Example usage of ExperimentConfig."""
    import os
    import logging
    from pathlib import Path

    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create a temporary directory for outputs
    output_dir = Path("temp_output")
    output_dir.mkdir(exist_ok=True)

    # Create a sample configuration
    config = ExperimentConfig(
        name="example_experiment",
        description="Example experiment configuration",
        grid=GridConfig(grid_type="random", n_random=50),
        tracing=TracingConfig(R_max=10.0, R_min=1.0, dt=0.01),
        field_components=[
            FieldComponentConfig(
                type="dipole",
                B_magnitude=1.0,
                moment_orientation_latitude_degree=0.0,
                moment_orientation_longitude_degree=0.0,
            )
        ],
        output=OutputConfig(
            output_dir=str(output_dir), save_r_max=True, save_r_line=True
        ),
    )

    # Save the configuration to a file
    config_path = output_dir / "example_config.json"
    config.save(config_path)
    logger.info(f"Configuration saved to: {config_path}")

    # Load the configuration from the file
    loaded_config = ExperimentConfig.load(config_path)
    logger.info("Successfully loaded configuration")

    # Print some details about the loaded configuration
    logger.info(f"Experiment name: {loaded_config.name}")
    logger.info(f"Grid type: {loaded_config.grid.grid_type}")
    logger.info(f"Number of field components: {len(loaded_config.field_components)}")

    # Example of how to access save flags
    logger.info(f"Save r_max: {loaded_config.output.save_r_max}")
    logger.info(f"Save r_line: {loaded_config.output.save_r_line}")

    return loaded_config


if __name__ == "__main__":
    main()
