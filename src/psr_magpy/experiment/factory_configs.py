"""Factory configurations for experiment series.

This module provides factory configuration classes that support flexible
parameter inputs for creating multiple experiment variations.
"""

from typing import List, Union, Dict, Any, Optional, Tuple
from pydantic import BaseModel

from .experiment_config import (
    FieldComponentConfig, FieldConfig, SphericalGridConfig, RandomGridConfig,
    HealpixGridConfig, CustomGridConfig, GridConfig, TracingConfig
)
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)

# Import parameter expansion function
from .parameter_expansion import get_parameter_combinations


class FieldFactoryConfig(BaseModel):
    """Factory configuration for field components with flexible parameter inputs."""
    
    components: List[Union[FieldComponentConfig, List[FieldComponentConfig], Dict[str, Any]]]
    
    model_config = {"arbitrary_types_allowed": True}


class SphericalGridFactoryConfig(BaseModel):
    """Factory configuration for spherical grids with flexible parameters."""
    
    nlon: Union[int, List[int]]
    nlat: Union[int, List[int]]
    pole_lat: Union[float, List[float]] = 0.0
    
    model_config = {"arbitrary_types_allowed": True}


class RandomGridFactoryConfig(BaseModel):
    """Factory configuration for random grids with flexible parameters."""
    
    nline_ori: Union[int, List[int]]
    
    model_config = {"arbitrary_types_allowed": True}


class HealpixGridFactoryConfig(BaseModel):
    """Factory configuration for HEALPix grids with flexible parameters."""
    
    nside: Union[int, List[int]]
    
    model_config = {"arbitrary_types_allowed": True}


class CustomGridFactoryConfig(BaseModel):
    """Factory configuration for custom grids with flexible parameters."""
    
    nline_ori: Union[int, List[int]]
    coordinates: Optional[Union[List[List[float]], List[List[List[float]]]]] = None
    
    model_config = {"arbitrary_types_allowed": True}


class GridFactoryConfig(BaseModel):
    """Factory configuration for grids with flexible parameter inputs."""
    
    grids: List[Union[SphericalGridConfig, RandomGridConfig, HealpixGridConfig, CustomGridConfig, 
                      SphericalGridFactoryConfig, RandomGridFactoryConfig, HealpixGridFactoryConfig, CustomGridFactoryConfig]]
    
    model_config = {"arbitrary_types_allowed": True}


class TracingFactoryConfig(BaseModel):
    """Factory configuration for tracing parameters with flexible inputs."""
    
    # Integration parameters
    max_integration_radius: Union[float, List[float]] = 10.0
    threshold_line_distance: Union[float, List[float]] = 1.0
    t_span: Union[Tuple[float, float], List[Tuple[float, float]]] = (0.0, 1e8)
    rtol: Union[float, List[float]] = 5.0e-9
    atol: Union[float, List[float]] = 5.0e-9
    surface_radius: Union[float, List[float]] = 1.0 - 1.e-8
    dt: Union[float, List[float]] = 0.01
    max_step: Union[float, List[float]] = 0.01
    trace_open_lines_backwards: Union[bool, List[bool]] = False
    
    # Performance parameters
    max_memory_mb: Union[float, List[float]] = 100.0
    safety_factor: Union[float, List[float]] = 0.9
    max_batch_size: Union[int, List[int]] = 100
    check_interval_sec: Union[float, List[float]] = 1.0
    n_cores: Union[Optional[int], List[Optional[int]]] = None
    n_trial: Union[int, List[int]] = 10
    progress_bar: Union[bool, List[bool]] = True
    verbose: Union[bool, List[bool]] = False
    skip_memory_estimation: Union[bool, List[bool]] = False
    
    # Output parameters
    save_results: Union[bool, List[bool]] = False
    filename: Union[Optional[str], List[Optional[str]]] = None
    output_format: Union[str, List[str]] = "fits"
    overwrite: Union[bool, List[bool]] = False
    
    model_config = {"arbitrary_types_allowed": True}


class TemplateFactoryConfig(BaseModel):
    """Factory configuration for templates with flexible parameter inputs."""
    
    templates: List[FieldLine]
    
    model_config = {"arbitrary_types_allowed": True}
    
    @classmethod
    def from_yaml_dict(cls, data: Dict[str, Any]) -> "TemplateFactoryConfig":
        """Create TemplateFactoryConfig from YAML dictionary with FieldLine reconstruction."""
        if "templates" in data:
            templates = []
            for template_data in data["templates"]:
                if isinstance(template_data, dict) and "property_system" in template_data:
                    # Reconstruct FieldLine from property system flags
                    flags = template_data["property_system"]
                    template = FieldLine()
                    for prop_name, enabled in flags.items():
                        if hasattr(template.property_system.save_flags, prop_name):
                            setattr(template.property_system.save_flags, prop_name, enabled)
                        else:
                            logger.warning(f"Property {prop_name} not found in FieldLine property_system.save_flags")
                    templates.append(template)
                else:
                    # Assume it's already a FieldLine object
                    templates.append(template_data)
            data["templates"] = templates
        
        return cls(**data)


def expand_field_configs(field_config: Union[FieldConfig, FieldFactoryConfig]) -> List[FieldConfig]:
    """Expand field configuration into all possible combinations."""
    if isinstance(field_config, FieldConfig):
        return [field_config]
    
    # Handle FieldFactoryConfig
    field_configs = []
    for component_group in field_config.components:
        if isinstance(component_group, FieldComponentConfig):
            # Single component
            field_configs.append(FieldConfig(components=[component_group]))
        elif isinstance(component_group, list):
            # List of components
            field_configs.append(FieldConfig(components=component_group))
        elif isinstance(component_group, dict):
            # Dictionary input with flexible parameters
            param_combinations = get_parameter_combinations(component_group)
            
            for params in param_combinations:
                config = FieldComponentConfig(**params)
                field_configs.append(FieldConfig(components=[config]))
    
    return field_configs


def expand_grid_configs(grid_config: Union[GridConfig, GridFactoryConfig]) -> List[GridConfig]:
    """Expand grid configuration into all possible combinations."""
    if isinstance(grid_config, GridConfig):
        return [grid_config]
    
    # Handle GridFactoryConfig
    grid_configs = []
    for grid in grid_config.grids:
        if isinstance(grid, SphericalGridFactoryConfig):
            # Expand spherical grid factory config
            param_combinations = get_parameter_combinations(grid.model_dump())
            for params in param_combinations:
                config = SphericalGridConfig(**params)
                grid_configs.append(GridConfig(spherical=config))
        elif isinstance(grid, RandomGridFactoryConfig):
            # Expand random grid factory config
            param_combinations = get_parameter_combinations(grid.model_dump())
            for params in param_combinations:
                config = RandomGridConfig(**params)
                grid_configs.append(GridConfig(random=config))
        elif isinstance(grid, HealpixGridFactoryConfig):
            # Expand HEALPix grid factory config
            param_combinations = get_parameter_combinations(grid.model_dump())
            for params in param_combinations:
                config = HealpixGridConfig(**params)
                grid_configs.append(GridConfig(healpix=config))
        elif isinstance(grid, CustomGridFactoryConfig):
            # Expand custom grid factory config
            param_combinations = get_parameter_combinations(grid.model_dump())
            for params in param_combinations:
                config = CustomGridConfig(**params)
                grid_configs.append(GridConfig(custom=config))
        elif isinstance(grid, dict):
            # Handle dictionary input with flexible parameters
            grid_type = grid.get("type", "spherical")
            # Expand parameter combinations
            param_combinations = get_parameter_combinations(grid)
            
            for params in param_combinations:
                # Remove type from params before creating config
                params.pop("type", None)
                
                if grid_type == "spherical":
                    config = SphericalGridConfig(**params)
                    grid_configs.append(GridConfig(spherical=config))
                elif grid_type == "random":
                    config = RandomGridConfig(**params)
                    grid_configs.append(GridConfig(random=config))
                elif grid_type == "healpix":
                    config = HealpixGridConfig(**params)
                    grid_configs.append(GridConfig(healpix=config))
                elif grid_type == "custom":
                    config = CustomGridConfig(**params)
                    grid_configs.append(GridConfig(custom=config))
                else:
                    error_msg = f"Unknown grid type: {grid_type}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        else:
            # Handle regular config object
            if isinstance(grid, SphericalGridConfig):
                grid_configs.append(GridConfig(spherical=grid))
            elif isinstance(grid, RandomGridConfig):
                grid_configs.append(GridConfig(random=grid))
            elif isinstance(grid, HealpixGridConfig):
                grid_configs.append(GridConfig(healpix=grid))
            elif isinstance(grid, CustomGridConfig):
                grid_configs.append(GridConfig(custom=grid))
    
    return grid_configs


def expand_tracing_configs(tracing_config: Union[TracingConfig, TracingFactoryConfig]) -> List[TracingConfig]:
    """Expand tracing configuration into all possible combinations."""
    if isinstance(tracing_config, TracingConfig):
        return [tracing_config]
    
    # Handle TracingFactoryConfig
    param_combinations = get_parameter_combinations(tracing_config.model_dump())
    
    tracing_configs = []
    for params in param_combinations:
        tracing_config = TracingConfig(**params)
        tracing_configs.append(tracing_config)
    
    return tracing_configs


def expand_template_configs(template_config: Union[FieldLine, TemplateFactoryConfig]) -> List[FieldLine]:
    """Expand template configuration into all possible combinations."""
    if isinstance(template_config, FieldLine):
        return [template_config]
    
    # Handle TemplateFactoryConfig
    return template_config.templates


 