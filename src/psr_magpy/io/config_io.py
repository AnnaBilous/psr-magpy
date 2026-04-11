"""YAML I/O implementations for configurations."""

from pathlib import Path
from typing import Any, Union, Dict, List, cast
import yaml

from .base import ConfigIO, IOUtils
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError

logger = get_logger(__name__, console_format_brief=True)


class YamlIO(ConfigIO):
    """Base class for YAML I/O operations."""

    def __init__(self, validator=None, memory_monitor=None):
        """Initialize YAML I/O with optional validator and memory monitor."""
        super().__init__(validator, memory_monitor)

    def save(self, config: Any, path: Union[str, Path], **kwargs) -> None:
        """Save configuration to YAML file with validation and recovery."""
        # Validate with recovery options
        if not self.validate_with_recovery(config, path, "save", **kwargs):
            raise PSRMagPyIOError(f"Validation failed for save operation: {path}")

        path = IOUtils.ensure_directory(path)

        with self._track_io_progress("save", path, config):
            with IOUtils.safe_file_operation("save", path):
                config_dict = self._prepare_for_save(config)
                with open(path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def load(self, path: Union[str, Path], **kwargs) -> Any:
        """Load configuration from YAML file with validation and recovery."""
        # Validate with recovery options
        if not self.validate_with_recovery(None, path, "load", **kwargs):
            raise PSRMagPyIOError(f"Validation failed for load operation: {path}")

        path = IOUtils.validate_path(path, must_exist=True)

        with self._track_io_progress("load", path):
            with IOUtils.safe_file_operation("load", path):
                with open(path, "r") as f:
                    config_dict = yaml.safe_load(f)
                return self._reconstruct_from_load(config_dict, **kwargs)

    def validate(self, config: Any) -> bool:
        """Validate configuration."""
        return True  # Override in subclasses

    def print_config(self, config: Any) -> None:
        """Print configuration in a readable format."""
        try:
            config_dict = self._prepare_for_save(config)
            yaml_string = yaml.safe_dump(
                config_dict, default_flow_style=False, indent=2, sort_keys=False
            )
            print("=== Configuration ===")
            print(yaml_string)
            print("=====================")
        except Exception as e:
            raise PSRMagPyIOError(f"Failed to print configuration: {e}") from e

    def _prepare_for_save(self, config: Any) -> dict:
        """Prepare configuration for saving."""
        # Override in subclasses
        raise NotImplementedError("_prepare_for_save not implemented yet")

    def _reconstruct_from_load(self, data: dict, **kwargs) -> Any:
        """Reconstruct configuration from loaded data."""
        # Override in subclasses
        raise NotImplementedError("_reconstruct_from_load not implemented yet")


class ExperimentConfigIO(YamlIO):
    """YAML I/O for experiment configurations."""

    def validate(self, config: Any) -> bool:
        """Validate experiment configuration."""
        try:
            # Basic validation - check required attributes
            required_attrs = ["name", "field", "grid", "tracing", "template"]
            for attr in required_attrs:
                if not hasattr(config, attr):
                    logger.error(f"Missing required attribute: {attr}")
                    return False

            # Validate nested objects
            if not hasattr(config.field, "get_complete_components"):
                logger.error("Invalid field configuration")
                return False

            if not hasattr(config.grid, "model_dump"):
                logger.error("Invalid grid configuration")
                return False

            if not hasattr(config.tracing, "get_complete_config"):
                logger.error("Invalid tracing configuration")
                return False

            logger.debug("Experiment configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def _prepare_for_save(self, config: Any) -> Dict[Any, Any]:
        """Prepare experiment configuration for saving."""
        # Convert to dict, handling nested objects
        grid_dict = config.grid.model_dump()
        
        # Remove coordinates from custom grid configs (only grid_file_path matters)
        if "custom" in grid_dict and isinstance(grid_dict["custom"], dict):
            if "coordinates" in grid_dict["custom"]:
                del grid_dict["custom"]["coordinates"]
        
        config_dict = {
            "name": config.name,
            "description": config.description,
            "field": {"components": config.field.get_complete_components()},
            "grid": grid_dict,
            "tracing": config.tracing.get_complete_config(),
            "template": config.template.property_system.get_flags_dict(),
            "metadata": config.metadata or {},
        }

        # Convert tuples and numpy arrays to lists for YAML compatibility
        return cast(Dict[Any, Any], IOUtils.convert_for_yaml(config_dict))

    def _reconstruct_from_load(self, data: dict, **kwargs) -> Any:
        """Reconstruct experiment configuration from loaded data."""
        from psr_magpy.experiment.experiment_config import ExperimentConfig
        from psr_magpy.experiment.field_config import FieldConfig
        from psr_magpy.experiment.grid_config import GridConfig
        from psr_magpy.experiment.tracing_config import TracingConfig
        from psr_magpy.line.field_line_template import FieldLineTemplate

        if data is None:
            raise PSRMagPyIOError(
                "Experiment configuration YAML is empty or contains only a null document."
            )
        if not isinstance(data, dict):
            raise PSRMagPyIOError(
                f"Experiment configuration must be a YAML mapping, got {type(data).__name__}"
            )

        # Reconstruct template
        template_data = data.get("template", {})

        # Ensure backwards compatibility for new properties
        default_flags = {
            "save_b_line": False,
            "save_curvature_line": False,
        }

        # Add missing properties with defaults
        for prop_name, default_value in default_flags.items():
            if prop_name not in template_data:
                template_data[prop_name] = default_value
                logger.debug(
                    f"Applied default {default_value} for missing property {prop_name} in experiment config"
                )

        template = FieldLineTemplate(**template_data)

        # Reconstruct field
        field = FieldConfig.model_validate(data["field"])

        # Reconstruct grid
        # Explicitly handle nested grid configs to ensure proper reconstruction
        grid_data = data["grid"]
        if isinstance(grid_data, dict):
            # Check if it's a nested structure with grid type keys
            if "healpix" in grid_data and isinstance(grid_data["healpix"], dict):
                from psr_magpy.experiment.grid_config import HealpixGridConfig
                healpix_config = HealpixGridConfig.model_validate(grid_data["healpix"])
                grid = GridConfig(healpix=healpix_config)
                logger.info(f"Reconstructed GridConfig with healpix: nside={healpix_config.nside}, subset={healpix_config.subset}, grid_file_path={healpix_config.grid_file_path}")
            elif "spherical" in grid_data and isinstance(grid_data["spherical"], dict):
                from psr_magpy.experiment.grid_config import SphericalGridConfig
                spherical_config = SphericalGridConfig.model_validate(grid_data["spherical"])
                grid = GridConfig(spherical=spherical_config)
            elif "random" in grid_data and isinstance(grid_data["random"], dict):
                from psr_magpy.experiment.grid_config import RandomGridConfig
                random_config = RandomGridConfig.model_validate(grid_data["random"])
                grid = GridConfig(random=random_config)
            elif "custom" in grid_data and isinstance(grid_data["custom"], dict):
                from psr_magpy.experiment.grid_config import CustomGridConfig
                custom_config = CustomGridConfig.model_validate(grid_data["custom"])
                grid = GridConfig(custom=custom_config)
            else:
                # Fallback to model_validate
                logger.warning(f"Grid data has no recognized grid type dicts, using model_validate. Keys: {list(grid_data.keys())}, types: {[(k, type(v).__name__) for k, v in grid_data.items()]}")
                grid = GridConfig.model_validate(grid_data)
        else:
            grid = GridConfig.model_validate(grid_data)

        # Reconstruct tracing
        tracing = TracingConfig(**data["tracing"])

        # Create config object
        config = ExperimentConfig(
            name=data["name"],
            description=data.get("description"),
            field=field,
            grid=grid,
            tracing=tracing,
            template=template,
            metadata=data.get("metadata"),
        )

        return config


class ExperimentSeriesConfigIO(YamlIO):
    """YAML I/O for experiment series configurations."""

    def validate(self, config: Any) -> bool:
        """Validate experiment series configuration."""
        try:
            # Basic validation - check required attributes
            required_attrs = ["name", "field", "grid", "tracing", "template"]
            for attr in required_attrs:
                if not hasattr(config, attr):
                    logger.error(f"Missing required attribute: {attr}")
                    return False

            # Validate that it's a Pydantic model
            if not hasattr(config, "model_dump"):
                logger.error(
                    "Invalid experiment series configuration - not a Pydantic model"
                )
                return False

            logger.debug("Experiment series configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def _prepare_for_save(self, config: Any) -> Dict[Any, Any]:
        """Prepare experiment series configuration for saving."""
        # Use Pydantic's model_dump for serialization
        config_dict = config.model_dump()

        # Remove coordinates from custom grid configs in grids list
        if "grid" in config_dict and "grids" in config_dict["grid"]:
            for grid_item in config_dict["grid"]["grids"]:
                if isinstance(grid_item, dict) and "custom" in grid_item:
                    if isinstance(grid_item["custom"], dict) and "coordinates" in grid_item["custom"]:
                        del grid_item["custom"]["coordinates"]

        # Convert FieldLine objects to dicts
        if "template" in config_dict and hasattr(
            config_dict["template"], "property_system"
        ):
            config_dict["template"] = config_dict[
                "template"
            ].property_system.get_flags_dict()

        # Convert tuples and numpy arrays to lists for YAML compatibility
        return cast(Dict[Any, Any], IOUtils.convert_for_yaml(config_dict))

    def _reconstruct_from_load(self, data: dict, **kwargs) -> Any:
        """Reconstruct experiment series configuration from loaded data."""
        from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
        from psr_magpy.line.field_line import FieldLine

        # Reconstruct nested objects
        reconstructed_data = reconstruct_from_yaml(data)

        # Reconstruct template if it's a dict
        if "template" in reconstructed_data and isinstance(
            reconstructed_data["template"], dict
        ):
            template_data = reconstructed_data["template"]

            # Ensure backwards compatibility for new properties
            default_flags = {
                "save_b_line": False,
                "save_curvature_line": False,
            }

            # Add missing properties with defaults
            for prop_name, default_value in default_flags.items():
                if prop_name not in template_data:
                    template_data[prop_name] = default_value
                    logger.debug(
                        f"Applied default {default_value} for missing property {prop_name} in series config"
                    )

            reconstructed_data["template"] = FieldLine(**template_data)

        # Create config object using model_validate to handle grid type information
        config = ExperimentSeriesConfig.model_validate(reconstructed_data)

        return config


def reconstruct_from_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct nested objects from YAML data.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary loaded from YAML.

    Returns
    -------
    Dict[str, Any]
        Dictionary with reconstructed objects.
    """
    if "factories" in data:
        factories_data = data["factories"]
        if isinstance(factories_data, list):
            # Multiple factories
            factories = []
            for factory_data in factories_data:
                factory_data = reconstruct_factory_data(factory_data)
                factories.append(factory_data)
            data["factories"] = factories
        else:
            # Single factory
            data["factories"] = reconstruct_factory_data(factories_data)

    return data


def reconstruct_factory_data(factory_data: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct individual factory data.

    Parameters
    ----------
    factory_data : Dict[str, Any]
        Factory data dictionary.

    Returns
    -------
    Dict[str, Any]
        Reconstructed factory data.
    """
    # Reconstruct template
    if "template" in factory_data and isinstance(factory_data["template"], dict):
        from psr_magpy.experiment.factory_configs import TemplateFactoryConfig

        factory_data["template"] = TemplateFactoryConfig.from_yaml_dict(
            factory_data["template"]
        )

    # Reconstruct field
    if "field" in factory_data and isinstance(factory_data["field"], dict):
        from psr_magpy.experiment.factory_configs import FieldFactoryConfig

        factory_data["field"] = FieldFactoryConfig(**factory_data["field"])

    # Reconstruct grid with nested factory configs
    if "grid" in factory_data and isinstance(factory_data["grid"], dict):
        grid_data = factory_data["grid"]
        if "grids" in grid_data and isinstance(grid_data["grids"], list):
            # Reconstruct nested grid factory configs
            from psr_magpy.experiment.factory_configs import (
                SphericalGridFactoryConfig,
                RandomGridFactoryConfig,
                HealpixGridFactoryConfig,
                CustomGridFactoryConfig,
            )

            reconstructed_grids: List[Union[
                SphericalGridFactoryConfig,
                RandomGridFactoryConfig,
                HealpixGridFactoryConfig,
                CustomGridFactoryConfig,
                Dict[Any, Any],
            ]] = []
            for grid_item in grid_data["grids"]:
                if isinstance(grid_item, dict):
                    # Determine grid type and reconstruct accordingly
                    if "nlon" in grid_item and "nlat" in grid_item:
                        reconstructed_grids.append(
                            SphericalGridFactoryConfig(**grid_item)
                        )
                    elif "nline_ori" in grid_item and "coordinates" not in grid_item:
                        reconstructed_grids.append(RandomGridFactoryConfig(**grid_item))
                    elif "nside" in grid_item:
                        reconstructed_grids.append(
                            HealpixGridFactoryConfig(**grid_item)
                        )
                    elif "nline_ori" in grid_item and "coordinates" in grid_item:
                        reconstructed_grids.append(CustomGridFactoryConfig(**grid_item))
                    else:
                        reconstructed_grids.append(grid_item)
                else:
                    reconstructed_grids.append(grid_item)
            grid_data["grids"] = reconstructed_grids
        from psr_magpy.experiment.factory_configs import GridFactoryConfig

        factory_data["grid"] = GridFactoryConfig(**grid_data)

    # Reconstruct tracing
    if "tracing" in factory_data and isinstance(factory_data["tracing"], dict):
        from psr_magpy.experiment.factory_configs import TracingFactoryConfig

        factory_data["tracing"] = TracingFactoryConfig(**factory_data["tracing"])

    return factory_data
