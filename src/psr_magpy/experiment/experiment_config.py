import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Any
import numpy as np
from pathlib import Path

from psr_magpy.grid.grid import GridType, HealpixGrid, SphericalGrid, RandomGrid
from psr_magpy.field.components import (
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
)
from psr_magpy.line.calculator import TracingConfig
from psr_magpy.line.field_line import PropertyName, PropertyGroup
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.json_utils import NumpyEncoder

logger = get_logger(__name__, console_format_brief=True)

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
        logger.debug(f"Initializing GridConfig with grid_type={self.grid_type}")
        if isinstance(self.grid_type, str):
            try:
                self.grid_type = GridType(self.grid_type.lower())
                logger.debug(f"Converted string grid_type to enum: {self.grid_type}")
            except ValueError:
                error_msg = f"Invalid grid type: {self.grid_type}. Must be one of {[e.value for e in GridType]}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def create_grid(self) -> Union["HealpixGrid", "SphericalGrid", "RandomGrid"]:
        """Create a grid based on the configuration."""
        logger.debug(f"Creating grid of type {self.grid_type}")
        if self.grid_type == GridType.HEALPIX:
            if self.nside is None:
                error_msg = "nside must be specified for healpix grid"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug(f"Creating HEALPix grid with nside={self.nside}")
            return HealpixGrid(nside=self.nside)
        elif self.grid_type == GridType.SPHERICAL:
            if self.nlon is None or self.nlat is None:
                error_msg = "nlon and nlat must be specified for spherical grid"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug(f"Creating Spherical grid with nlon={self.nlon}, nlat={self.nlat}, pole_lat={self.pole_lat}")
            return SphericalGrid(nlon=self.nlon, nlat=self.nlat, pole_lat=self.pole_lat)
        elif self.grid_type == GridType.RANDOM:
            if self.n_random is None:
                error_msg = "n_random must be specified for random grid"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug(f"Creating Random grid with n_random={self.n_random}")
            return RandomGrid(nline_ori=self.n_random)
        else:
            error_msg = f"Unknown grid type: {self.grid_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)


@dataclass
class FieldComponentConfig:
    """Configuration for a magnetic field component."""

    type: str  # 'dipole' or 'quadrupole'
    B_magnitude: float
    moment_orientation_latitude_degree: float
    moment_orientation_longitude_degree: float
    offset_r_ns: Optional[np.ndarray] = None
    

    def create_component(self) -> Union[DipoleFieldComponent, QuadrupoleFieldComponent]:
        """Create a field component based on the configuration."""
        logger.debug(f"Creating field component of type {self.type}")
        
        # Common parameters
        params = {
            "B_magnitude": self.B_magnitude,
            "moment_orientation_latitude_degree": self.moment_orientation_latitude_degree,
            "moment_orientation_longitude_degree": self.moment_orientation_longitude_degree,
        }
        
        # Add offset if provided
        if self.offset_r_ns is not None:
            params["offset_r_ns"] = self.offset_r_ns
            
        if self.type == "dipole":
            logger.debug(f"Creating Dipole field component with B={self.B_magnitude}, "
                       f"lat={self.moment_orientation_latitude_degree}°, "
                       f"lon={self.moment_orientation_longitude_degree}°"
                       f"{', offset=' + str(self.offset_r_ns) if self.offset_r_ns is not None else ''}")
            return DipoleFieldComponent(**params)
        elif self.type == "quadrupole":
            logger.debug(f"Creating Quadrupole field component with B={self.B_magnitude}, "
                       f"lat={self.moment_orientation_latitude_degree}°, "
                       f"lon={self.moment_orientation_longitude_degree}°"
                       f"{', offset=' + str(self.offset_r_ns) if self.offset_r_ns is not None else ''}")
            return QuadrupoleFieldComponent(**params)
        else:
            error_msg = f"Unknown field component type: {self.type}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class OutputConfig:
    """Configuration for output settings."""

    def __init__(self, output_dir="output", file_prefix="", **kwargs):
        """Initialize with output directory and optional save flags.

        Args:
            output_dir: Directory to save output files
            file_prefix: Prefix to add to output filenames
            **kwargs: Save flags (e.g., save_r_max=True) or property groups (e.g., save_all=True)
                      Available property groups:
                      - save_all: Enable all properties
                      - save_line_data: Enable r_line, pa_line, curvature_line
                      - save_line_max: Enable r_max, r_cyl_max, z_abs_max
                      - save_b_ini: Enable b_ini, b_r_ini_angle
        """
        logger.debug(f"Initializing OutputConfig with dir={output_dir}, prefix={file_prefix}")
        self.output_dir = output_dir
        self.file_prefix = file_prefix

        # Process property groups first
        property_flags: Dict[str, bool] = {}
        
        # Check for property groups
        for group in PropertyGroup:
            group_name = group.name.lower()
            if group_name in kwargs and kwargs[group_name]:
                # Enable all properties in this group
                for prop in group.properties:
                    property_flags[prop.flag_name] = True
                logger.debug(f"Enabled property group: {group_name}")

        # Set default values for all save flags
        for prop in PropertyName:
            # FOOTPOINT_ID is always True, others default to False
            default_value = True if prop == PropertyName.FOOTPOINT_ID else False
            # Allow override from kwargs, but force footpoint_id to True
            value = kwargs.get(prop.flag_name, property_flags.get(prop.flag_name, default_value))
            if prop == PropertyName.FOOTPOINT_ID:
                value = True
            setattr(self, prop.flag_name, value)

        # Update with provided kwargs for any other attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                logger.debug(f"Setting additional output attribute {key}={value}")
                setattr(self, key, value)

    def to_dict(self):
        """Convert to dictionary for serialization."""
        logger.debug("Converting OutputConfig to dictionary")
        result = {"output_dir": self.output_dir, "file_prefix": self.file_prefix}

        # Add all save flags
        for prop in PropertyName:
            result[prop.flag_name] = getattr(self, prop.flag_name)

        # Add any additional attributes
        for key, value in vars(self).items():
            if key not in result and not key.startswith('_'):
                result[key] = value

        return result

    def get_save_flags(self):
        """Get a dictionary of all save flags.

        Returns:
            dict: Dictionary of save flags for use with FieldLine constructor
        """
        logger.debug("Getting save flags from OutputConfig")
        save_flags = {}
        for prop in PropertyName:
            flag_name = prop.flag_name
            save_flags[flag_name] = getattr(self, flag_name)
        
        # Add property groups
        for group in PropertyGroup:
            group_name = group.name.lower()
            # Check if all properties in this group are enabled
            all_enabled = all(getattr(self, prop.flag_name) for prop in group.properties)
            save_flags[group_name] = all_enabled
            
        return save_flags

    @classmethod
    def from_dict(cls, data):
        """Create instance from dictionary."""
        logger.debug(f"Creating OutputConfig from dictionary: {data}")
        return cls(**data)

    def get_output_filename(self, extension=""):
        """Get the full output filename with path."""
        if self.file_prefix:
            filename = f"{self.file_prefix}_{extension}"
        else:
            filename = f"{extension}"

        full_path = Path(self.output_dir) / filename
        logger.debug(f"Generated output filename: {full_path}")
        return full_path


@dataclass
class ExperimentConfig:
    """Main configuration class for magnetic field line experiments."""

    # Experiment metadata
    name: str
    description: Optional[str] = None

    # Component configurations
    grid: Union[GridConfig, str, Path] = None  # Can be GridConfig or path to grid file
    tracing: TracingConfig = None
    field_components: List[FieldComponentConfig] = None

    # Output configuration
    output: OutputConfig = OutputConfig()

    def __post_init__(self):
        """Log the creation of the experiment configuration."""
        logger.debug(f"Created experiment configuration: {self.name}")
        if self.description:
            logger.debug(f"Description: {self.description}")
        
        # Log component counts
        if self.field_components:
            logger.debug(f"Number of field components: {len(self.field_components)}")
        if self.grid:
            if isinstance(self.grid, GridConfig):
                logger.debug(f"Grid type: {self.grid.grid_type}")
            else:
                logger.debug(f"Grid from file: {self.grid}")
        if self.tracing:
            logger.debug(f"Tracing config: R_max={self.tracing.R_max}, min_R_max_line={self.tracing.min_R_max_line}, dt={self.tracing.dt}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from a JSON file."""
        filepath = Path(filepath)
        logger.info(f"Loading experiment configuration from {filepath}")
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                logger.debug(f"Successfully read JSON data from {filepath}")
        except FileNotFoundError:
            error_msg = f"Configuration file not found: {filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file {filepath}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Convert field components data to proper config objects
            if "field_components" in data:
                logger.debug(f"Converting {len(data['field_components'])} field components")
                data["field_components"] = [
                    FieldComponentConfig(**comp) for comp in data["field_components"]
                ]

            # Convert nested configs
            logger.debug("Converting grid configuration")
            if isinstance(data["grid"], dict):
                data["grid"] = GridConfig(**data["grid"])
            # If grid is a string, keep it as a path
            
            logger.debug("Converting tracing configuration")
            data["tracing"] = TracingConfig(**data["tracing"])
            
            logger.debug("Converting output configuration")
            data["output"] = OutputConfig.from_dict(data["output"])

            config = cls(**data)
            logger.info(f"Successfully loaded configuration for experiment: {config.name}")
            return config
            
        except KeyError as e:
            error_msg = f"Missing required field in configuration: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except TypeError as e:
            error_msg = f"Invalid configuration structure: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        filepath = Path(filepath)
        logger.info(f"Saving experiment configuration to {filepath}")
        
        # Convert to dictionary
        config_dict = asdict(self)

        # Handle grid - could be GridConfig or string path
        if isinstance(self.grid, GridConfig):
            # Convert grid_type to string
            if isinstance(config_dict["grid"]["grid_type"], GridType):
                logger.debug("Converting grid_type enum to string")
                config_dict["grid"]["grid_type"] = config_dict["grid"]["grid_type"].value
        # If grid is a string or Path, keep it as is
        elif isinstance(self.grid, (str, Path)):
            logger.debug(f"Grid is a path: {self.grid}")
            config_dict["grid"] = str(self.grid)

        # Handle OutputConfig serialization
        logger.debug("Converting output configuration to dictionary")
        config_dict["output"] = self.output.to_dict()

        # Convert output_dir Path to string if needed
        if isinstance(config_dict["output"]["output_dir"], Path):
            logger.debug("Converting output_dir Path to string")
            config_dict["output"]["output_dir"] = str(
                config_dict["output"]["output_dir"]
            )

        # Ensure directory exists
        try:
            logger.debug(f"Creating directory if needed: {filepath.parent}")
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save to file with the NumpyEncoder
            with open(filepath, "w") as f:
                json.dump(config_dict, f, indent=4, cls=NumpyEncoder)
                logger.info(f"Configuration successfully saved to {filepath}")
        except PermissionError:
            error_msg = f"Permission denied when writing to {filepath}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
        except OSError as e:
            error_msg = f"Error saving configuration to {filepath}: {str(e)}"
            logger.error(error_msg)
            raise


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
    logger.debug(f"Created output directory: {output_dir}")

    # Create a sample configuration
    logger.debug("Creating sample experiment configuration")
    config = ExperimentConfig(
        name="example_experiment",
        description="Example experiment configuration",
        grid=GridConfig(grid_type="random", n_random=50),
        tracing=TracingConfig(R_max=10.0, min_R_max_line=1.0, dt=0.01),
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
    logger.debug(f"Configuration saved to: {config_path}")

    # Load the configuration from the file
    loaded_config = ExperimentConfig.load(config_path)
    logger.debug("Successfully loaded configuration")

    # Print some details about the loaded configuration
    logger.debug(f"Experiment name: {loaded_config.name}")
    logger.debug(f"Grid type: {loaded_config.grid.grid_type}")
    logger.debug(f"Number of field components: {len(loaded_config.field_components)}")

    # Example of how to access save flags
    logger.debug(f"Save r_max: {loaded_config.output.save_r_max}")
    logger.debug(f"Save r_line: {loaded_config.output.save_r_line}")

    return loaded_config


if __name__ == "__main__":
    main()
