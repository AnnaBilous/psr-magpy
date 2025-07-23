"""YAML handling utilities for experiment series.

This module provides YAML serialization, deserialization, and reconstruction
utilities for experiment series configurations.
"""

import yaml
from typing import Dict, Any
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


# Custom YAML representer for FieldLine objects
def field_line_representer(dumper, data):
    """Custom YAML representer for FieldLine objects."""
    # Convert FieldLine to a dict representation
    flags_dict = data.property_system.get_flags_dict()
    return dumper.represent_dict({
        'property_system': flags_dict
    })


# Custom YAML representer for tuples
def tuple_representer(dumper, data):
    """Custom YAML representer for tuple objects."""
    return dumper.represent_list(list(data))


# Register the custom representers
yaml.add_representer(FieldLine, field_line_representer)
yaml.add_representer(tuple, tuple_representer)


def reconstruct_from_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct nested objects from YAML data.
    
    Args:
        data: Dictionary loaded from YAML
        
    Returns:
        Dictionary with reconstructed objects
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
    
    Args:
        factory_data: Factory data dictionary
        
    Returns:
        Reconstructed factory data
    """
    # Reconstruct template
    if "template" in factory_data and isinstance(factory_data["template"], dict):
        from .factory_configs import TemplateFactoryConfig
        factory_data["template"] = TemplateFactoryConfig.from_yaml_dict(factory_data["template"])
    
    # Reconstruct field
    if "field" in factory_data and isinstance(factory_data["field"], dict):
        from .factory_configs import FieldFactoryConfig
        factory_data["field"] = FieldFactoryConfig(**factory_data["field"])
    
    # Reconstruct grid with nested factory configs
    if "grid" in factory_data and isinstance(factory_data["grid"], dict):
        grid_data = factory_data["grid"]
        if "grids" in grid_data and isinstance(grid_data["grids"], list):
            # Reconstruct nested grid factory configs
            from .factory_configs import (
                SphericalGridFactoryConfig, RandomGridFactoryConfig,
                HealpixGridFactoryConfig, CustomGridFactoryConfig
            )
            reconstructed_grids = []
            for grid_item in grid_data["grids"]:
                if isinstance(grid_item, dict):
                    # Determine grid type and reconstruct accordingly
                    if "nlon" in grid_item and "nlat" in grid_item:
                        reconstructed_grids.append(SphericalGridFactoryConfig(**grid_item))
                    elif "nline_ori" in grid_item and "coordinates" not in grid_item:
                        reconstructed_grids.append(RandomGridFactoryConfig(**grid_item))
                    elif "nside" in grid_item:
                        reconstructed_grids.append(HealpixGridFactoryConfig(**grid_item))
                    elif "nline_ori" in grid_item and "coordinates" in grid_item:
                        reconstructed_grids.append(CustomGridFactoryConfig(**grid_item))
                    else:
                        reconstructed_grids.append(grid_item)
                else:
                    reconstructed_grids.append(grid_item)
            grid_data["grids"] = reconstructed_grids
        from .factory_configs import GridFactoryConfig
        factory_data["grid"] = GridFactoryConfig(**grid_data)
    
    # Reconstruct tracing
    if "tracing" in factory_data and isinstance(factory_data["tracing"], dict):
        from .factory_configs import TracingFactoryConfig
        factory_data["tracing"] = TracingFactoryConfig(**factory_data["tracing"])
    
    return factory_data 