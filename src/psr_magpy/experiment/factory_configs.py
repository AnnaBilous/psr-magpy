"""Factory configurations for experiment series.

This module provides factory configuration classes that support flexible
parameter inputs for creating multiple experiment variations.
"""

from typing import List, Union, Dict, Any, Optional, Tuple
from pydantic import BaseModel

from .field_config import FieldComponentConfig, FieldConfig
from .grid_config import (
    SphericalGridConfig,
    RandomGridConfig,
    HealpixGridConfig,
    CustomGridConfig,
    GridConfig,
)
from .tracing_config import TracingConfig
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyValidationError, PSRMagPyGridError
from psr_magpy.line.tracing_parameters import TracingDefaults

logger = get_logger(__name__, console_format_brief=True)

# Import parameter expansion function
from .parameter_expansion import get_parameter_combinations


class FieldComponentFactoryConfig(BaseModel):
    """Factory configuration for field components with flexible parameters."""

    moment_type: str
    B_magnitude: Union[float, List[float]] = 1.0
    moment_orientation_latitude_degree: Union[float, List[float]] = 0.0
    moment_orientation_longitude_degree: Union[float, List[float]] = 0.0
    offset_r_ns: Union[List[float], List[List[float]]] = [
        0.0,
        0.0,
        0.0,
    ]  # Can be expanded

    model_config = {"arbitrary_types_allowed": True}


class FieldFactoryConfig(BaseModel):
    """Factory configuration for field components with flexible parameter inputs."""

    components: List[
        Union[
            FieldComponentConfig,
            List[FieldComponentConfig],
            List[FieldComponentFactoryConfig],
            Dict[str, Any],
            FieldComponentFactoryConfig,
        ]
    ]

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
    grid_file_path: Optional[Union[str, List[str]]] = None

    model_config = {"arbitrary_types_allowed": True}


class HealpixGridFactoryConfig(BaseModel):
    """Factory configuration for HEALPix grids with flexible parameters."""

    nside: Union[int, List[int]]

    model_config = {"arbitrary_types_allowed": True}


class CustomGridFactoryConfig(BaseModel):
    """Factory configuration for custom grids with flexible parameters."""

    nline_ori: Union[int, List[int]]
    coordinates: Optional[Union[List[List[float]], List[List[List[float]]]]] = None
    grid_file_path: Optional[Union[str, List[str]]] = None

    model_config = {"arbitrary_types_allowed": True}


class GridFactoryConfig(BaseModel):
    """Factory configuration for grids with flexible parameter inputs."""

    grids: List[
        Union[
            SphericalGridConfig,
            RandomGridConfig,
            HealpixGridConfig,
            CustomGridConfig,
            SphericalGridFactoryConfig,
            RandomGridFactoryConfig,
            HealpixGridFactoryConfig,
            CustomGridFactoryConfig,
        ]
    ]

    model_config = {"arbitrary_types_allowed": True}

    def model_dump(self, **kwargs):
        """Dump model to dictionary, preserving grid type information."""
        data = super().model_dump(**kwargs)

        # Convert grid configs to preserve type information
        if "grids" in data:
            grids_data = []
            for grid in self.grids:
                if isinstance(grid, SphericalGridConfig):
                    grids_data.append({"spherical": grid.model_dump(**kwargs)})
                elif isinstance(grid, RandomGridConfig):
                    grids_data.append({"random": grid.model_dump(**kwargs)})
                elif isinstance(grid, HealpixGridConfig):
                    grids_data.append({"healpix": grid.model_dump(**kwargs)})
                elif isinstance(grid, CustomGridConfig):
                    grids_data.append({"custom": grid.model_dump(**kwargs)})
                elif isinstance(grid, SphericalGridFactoryConfig):
                    grids_data.append({"spherical": grid.model_dump(**kwargs)})
                elif isinstance(grid, RandomGridFactoryConfig):
                    grids_data.append({"random": grid.model_dump(**kwargs)})
                elif isinstance(grid, HealpixGridFactoryConfig):
                    grids_data.append({"healpix": grid.model_dump(**kwargs)})
                elif isinstance(grid, CustomGridFactoryConfig):
                    grids_data.append({"custom": grid.model_dump(**kwargs)})
                else:
                    # Fallback to original serialization
                    grids_data.append(grid.model_dump(**kwargs))

            data["grids"] = grids_data

        return data

    @classmethod
    def model_validate(cls, obj):
        """Validate model, handling grid type information in YAML."""
        if isinstance(obj, dict) and "grids" in obj:
            # Convert grid type dictionaries back to config objects
            grids_data = []
            for grid_dict in obj["grids"]:
                if isinstance(grid_dict, dict):
                    if "spherical" in grid_dict:
                        grids_data.append(SphericalGridConfig(**grid_dict["spherical"]))
                    elif "random" in grid_dict:
                        grids_data.append(RandomGridConfig(**grid_dict["random"]))
                    elif "healpix" in grid_dict:
                        grids_data.append(HealpixGridConfig(**grid_dict["healpix"]))
                    elif "custom" in grid_dict:
                        grids_data.append(CustomGridConfig(**grid_dict["custom"]))
                    else:
                        # Fallback to original validation
                        grids_data.append(grid_dict)
                else:
                    grids_data.append(grid_dict)

            obj["grids"] = grids_data

        return super().model_validate(obj)


class TracingFactoryConfig(BaseModel):
    """Factory configuration for tracing parameters with flexible inputs."""

    # Integration parameters
    max_integration_radius: Union[float, List[float]] = (
        TracingDefaults.MAX_INTEGRATION_RADIUS
    )
    threshold_line_distance: Union[float, List[float]] = (
        TracingDefaults.THRESHOLD_LINE_DISTANCE
    )
    t_span: Union[Tuple[float, float], List[Tuple[float, float]]] = (
        TracingDefaults.T_SPAN
    )
    rtol: Union[float, List[float]] = TracingDefaults.RTOL
    atol: Union[float, List[float]] = TracingDefaults.ATOL
    surface_radius: Union[float, List[float]] = TracingDefaults.SURFACE_RADIUS
    dt: Union[float, List[float]] = TracingDefaults.DT
    max_step: Union[float, List[float]] = TracingDefaults.MAX_STEP
    method: Union[str, List[str]] = TracingDefaults.METHOD
    integration_system: Union[str, List[str]] = TracingDefaults.INTEGRATION_SYSTEM
    field_backend: Union[str, List[str]] = TracingDefaults.FIELD_BACKEND
    solver_backend: Union[str, List[str]] = TracingDefaults.SOLVER_BACKEND
    trace_open_lines_backwards: Union[bool, List[bool]] = (
        TracingDefaults.TRACE_OPEN_LINES_BACKWARDS
    )
    verify_by_reversing: Union[bool, List[bool]] = TracingDefaults.VERIFY_BY_REVERSING
    reversed_error_threshold: Union[float, List[float]] = (
        TracingDefaults.REVERSED_ERROR_THRESHOLD
    )

    # Performance parameters
    max_memory_mb: Union[float, List[float]] = TracingDefaults.MAX_MEMORY_MB
    safety_factor: Union[float, List[float]] = TracingDefaults.SAFETY_FACTOR
    max_batch_size: Union[int, List[int]] = TracingDefaults.MAX_BATCH_SIZE
    batch_size_safety_factor: Union[int, List[int]] = (
        TracingDefaults.BATCH_SIZE_SAFETY_FACTOR
    )
    check_interval_sec: Union[float, List[float]] = TracingDefaults.CHECK_INTERVAL_SEC
    n_cores: Union[Optional[int], List[Optional[int]]] = TracingDefaults.N_CORES
    n_trial: Union[int, List[int]] = TracingDefaults.N_TRIAL
    progress_bar: Union[bool, List[bool]] = TracingDefaults.PROGRESS_BAR
    verbose: Union[bool, List[bool]] = TracingDefaults.VERBOSE
    skip_memory_estimation: Union[bool, List[bool]] = (
        TracingDefaults.SKIP_MEMORY_ESTIMATION
    )

    # Output parameters
    save_results_to_disk: Union[bool, List[bool]] = TracingDefaults.SAVE_RESULTS_TO_DISK
    keep_results_in_memory: Union[bool, List[bool]] = (
        TracingDefaults.KEEP_RESULTS_IN_MEMORY
    )
    filename: Union[Optional[str], List[Optional[str]]] = TracingDefaults.FILENAME
    output_format: Union[str, List[str]] = TracingDefaults.OUTPUT_FORMAT
    overwrite: Union[bool, List[bool]] = TracingDefaults.OVERWRITE

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
                if (
                    isinstance(template_data, dict)
                    and "property_system" in template_data
                ):
                    # Reconstruct FieldLine from property system flags
                    flags = template_data["property_system"]
                    template = FieldLine()

                    # Define default values for new properties (backwards compatibility)
                    default_flags = {
                        "save_b_line": False,
                        "save_curvature_line": False,
                        "save_tangent_theta": False,
                        "save_tangent_phi": False,
                    }

                    # Apply flags from YAML, with defaults for missing properties
                    for prop_name, enabled in flags.items():
                        if hasattr(template.property_system.save_flags, prop_name):
                            setattr(
                                template.property_system.save_flags, prop_name, enabled
                            )
                        else:
                            logger.warning(
                                f"Property {prop_name} not found in FieldLine property_system.save_flags"
                            )

                    # Apply defaults for missing new properties
                    for prop_name, default_value in default_flags.items():
                        if prop_name not in flags:
                            if hasattr(template.property_system.save_flags, prop_name):
                                setattr(
                                    template.property_system.save_flags,
                                    prop_name,
                                    default_value,
                                )
                                logger.debug(
                                    f"Applied default {default_value} for missing property {prop_name}"
                                )
                            else:
                                logger.warning(
                                    f"Default property {prop_name} not found in FieldLine property_system.save_flags"
                                )
                    templates.append(template)
                else:
                    # Assume it's already a FieldLine object
                    templates.append(template_data)
            data["templates"] = templates

        return cls(**data)


def expand_field_configs(
    field_config: Union[FieldConfig, FieldFactoryConfig]
) -> List[FieldConfig]:
    """Expand field configuration into all possible combinations."""
    if isinstance(field_config, FieldConfig):
        return [field_config]

    # Handle FieldFactoryConfig
    # First, expand each component group separately
    expanded_components = []

    for component_group in field_config.components:
        if isinstance(component_group, FieldComponentConfig):
            # Single component
            expanded_components.append([component_group])
        elif isinstance(component_group, list):
            # List of components - could be FieldComponentConfig or FieldComponentFactoryConfig
            if all(isinstance(comp, FieldComponentConfig) for comp in component_group):
                # List of FieldComponentConfig - no expansion needed
                expanded_components.append(component_group)
            elif all(
                isinstance(comp, FieldComponentFactoryConfig)
                for comp in component_group
            ):
                # List of FieldComponentFactoryConfig - expand each and combine within group
                expanded_components_in_group = []

                for comp in component_group:
                    factory_params = comp.model_dump()

                    # Handle special case for offset_r_ns - only expand if it's a list of lists
                    if "offset_r_ns" in factory_params:
                        offset_value = factory_params["offset_r_ns"]
                        if isinstance(offset_value, list) and len(offset_value) > 0:
                            # Check if it's a list of lists (should be expanded) or a single list (should not be expanded)
                            if isinstance(offset_value[0], list):
                                # List of lists - keep for expansion
                                pass
                            else:
                                # Single list - remove from expansion, treat as fixed
                                factory_params.pop("offset_r_ns")

                    param_combinations = get_parameter_combinations(factory_params)
                    comp_expanded = []

                    for params in param_combinations:
                        # Restore offset_r_ns if it was removed
                        if (
                            "offset_r_ns" not in params
                            and "offset_r_ns" in comp.model_dump()
                        ):
                            offset_value = comp.model_dump()["offset_r_ns"]
                            if (
                                isinstance(offset_value, list)
                                and len(offset_value) > 0
                                and not isinstance(offset_value[0], list)
                            ):
                                params["offset_r_ns"] = offset_value

                        config = FieldComponentConfig(**params)
                        comp_expanded.append(config)

                    expanded_components_in_group.append(comp_expanded)

                # Create all combinations within this group
                from itertools import product

                group_combinations = list(product(*expanded_components_in_group))
                expanded_components.append(group_combinations)
            else:
                # Mixed types - treat as regular list
                expanded_components.append(component_group)
        elif isinstance(component_group, dict):
            # Dictionary input with flexible parameters
            param_combinations = get_parameter_combinations(component_group)
            group_components = []
            for params in param_combinations:
                config = FieldComponentConfig(**params)
                group_components.append(config)
            expanded_components.append(group_components)
        elif isinstance(component_group, FieldComponentFactoryConfig):
            # FieldComponentFactoryConfig
            factory_params = component_group.model_dump()

            # Handle special case for offset_r_ns - only expand if it's a list of lists
            if "offset_r_ns" in factory_params:
                offset_value = factory_params["offset_r_ns"]
                if isinstance(offset_value, list) and len(offset_value) > 0:
                    # Check if it's a list of lists (should be expanded) or a single list (should not be expanded)
                    if isinstance(offset_value[0], list):
                        # List of lists - keep for expansion
                        pass
                    else:
                        # Single list - remove from expansion, treat as fixed
                        factory_params.pop("offset_r_ns")

            param_combinations = get_parameter_combinations(factory_params)
            group_components = []

            for params in param_combinations:
                # Restore offset_r_ns if it was removed
                if (
                    "offset_r_ns" not in params
                    and "offset_r_ns" in component_group.model_dump()
                ):
                    offset_value = component_group.model_dump()["offset_r_ns"]
                    if (
                        isinstance(offset_value, list)
                        and len(offset_value) > 0
                        and not isinstance(offset_value[0], list)
                    ):
                        params["offset_r_ns"] = offset_value

                config = FieldComponentConfig(**params)
                group_components.append(config)
            expanded_components.append(group_components)

    # Now create all combinations of the expanded components
    field_configs = []

    # Each group should be a separate field configuration
    for group in expanded_components:
        if isinstance(group, list):
            # Check if this is a list of FieldComponentConfig or a list of tuples
            if len(group) > 0 and isinstance(group[0], tuple):
                # List of tuples - each tuple is a combination
                for combo in group:
                    field_configs.append(FieldConfig(components=list(combo)))
            else:
                # Single group - create one field config
                field_configs.append(FieldConfig(components=group))
        else:
            # Group contains combinations - create field config for each combination
            for combo in group:
                if isinstance(combo, list):
                    field_configs.append(FieldConfig(components=combo))
                elif isinstance(combo, tuple):
                    # Convert tuple to list
                    field_configs.append(FieldConfig(components=list(combo)))
                else:
                    field_configs.append(FieldConfig(components=[combo]))

    # Sort field configs by B_magnitude in descending order
    field_configs.sort(key=lambda x: x.components[0].B_magnitude, reverse=True)

    return field_configs


def _expand_random_grid_factory_config(
    grid: RandomGridFactoryConfig,
) -> List[GridConfig]:
    """Expand RandomGridFactoryConfig with paired nline_ori and grid_file_path parameters."""
    grid_configs = []

    # Handle nline_ori parameter
    if isinstance(grid.nline_ori, list):
        nline_ori_values = grid.nline_ori
    else:
        nline_ori_values = [grid.nline_ori]

    # Handle grid_file_path parameter
    if grid.grid_file_path is None:
        # No file paths provided, create from scratch
        for nline_ori in nline_ori_values:
            config = RandomGridConfig(nline_ori=nline_ori)
            grid_configs.append(GridConfig(random=config))
    else:
        if isinstance(grid.grid_file_path, list):
            grid_file_paths = grid.grid_file_path
        else:
            grid_file_paths = [grid.grid_file_path]

        # Pair nline_ori with grid_file_path
        if len(nline_ori_values) != len(grid_file_paths):
            raise PSRMagPyValidationError(
                f"Number of nline_ori values ({len(nline_ori_values)}) must match number of grid_file_paths ({len(grid_file_paths)})"
            )

        for nline_ori, grid_file_path in zip(nline_ori_values, grid_file_paths):
            config = RandomGridConfig(
                nline_ori=nline_ori, grid_file_path=grid_file_path
            )
            grid_configs.append(GridConfig(random=config))

    return grid_configs


def _expand_custom_grid_factory_config(
    grid: CustomGridFactoryConfig,
) -> List[GridConfig]:
    """Expand CustomGridFactoryConfig with paired nline_ori and grid_file_path parameters."""
    grid_configs = []

    # Handle nline_ori parameter
    if isinstance(grid.nline_ori, list):
        nline_ori_values = grid.nline_ori
    else:
        nline_ori_values = [grid.nline_ori]

    # Handle coordinates parameter
    if grid.coordinates is None:
        coordinates_values = [None] * len(nline_ori_values)
    elif isinstance(grid.coordinates, list):
        coordinates_values = grid.coordinates
        if len(coordinates_values) != len(nline_ori_values):
            raise PSRMagPyValidationError(
                f"Number of coordinates ({len(coordinates_values)}) must match number of nline_ori values ({len(nline_ori_values)})"
            )
        # Ensure each coordinates value is a list of lists
        for i, coords in enumerate(coordinates_values):
            if coords is not None and not isinstance(coords[0], list):
                coordinates_values[i] = [list(coords)]
    else:
        coordinates_values = [grid.coordinates] * len(nline_ori_values)

    # Handle grid_file_path parameter
    if grid.grid_file_path is None:
        # No file paths provided, create from scratch
        for nline_ori, coordinates in zip(nline_ori_values, coordinates_values):
            config = CustomGridConfig(nline_ori=nline_ori, coordinates=coordinates)
            grid_configs.append(GridConfig(custom=config))
    else:
        if isinstance(grid.grid_file_path, list):
            grid_file_paths = grid.grid_file_path
        else:
            grid_file_paths = [grid.grid_file_path]

        # Pair nline_ori with grid_file_path
        if len(nline_ori_values) != len(grid_file_paths):
            raise PSRMagPyValidationError(
                f"Number of nline_ori values ({len(nline_ori_values)}) must match number of grid_file_paths ({len(grid_file_paths)})"
            )

        for nline_ori, coordinates, grid_file_path in zip(
            nline_ori_values, coordinates_values, grid_file_paths
        ):
            config = CustomGridConfig(
                nline_ori=nline_ori,
                coordinates=coordinates,
                grid_file_path=grid_file_path,
            )
            grid_configs.append(GridConfig(custom=config))

    return grid_configs


def expand_grid_configs(
    grid_config: Union[GridConfig, GridFactoryConfig]
) -> List[GridConfig]:
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
            # Expand random grid factory config with paired parameters
            grid_configs.extend(_expand_random_grid_factory_config(grid))
        elif isinstance(grid, HealpixGridFactoryConfig):
            # Expand HEALPix grid factory config
            param_combinations = get_parameter_combinations(grid.model_dump())
            for params in param_combinations:
                config = HealpixGridConfig(**params)
                grid_configs.append(GridConfig(healpix=config))
        elif isinstance(grid, CustomGridFactoryConfig):
            # Expand custom grid factory config with paired parameters
            grid_configs.extend(_expand_custom_grid_factory_config(grid))
        elif isinstance(grid, dict):
            # Handle dictionary input with flexible parameters
            # Check if it's the new format with grid type as key
            if "spherical" in grid:
                grid_data = grid["spherical"]
                param_combinations = get_parameter_combinations(grid_data)
                for params in param_combinations:
                    config = SphericalGridConfig(**params)
                    grid_configs.append(GridConfig(spherical=config))
            elif "random" in grid:
                grid_data = grid["random"]
                param_combinations = get_parameter_combinations(grid_data)
                for params in param_combinations:
                    config = RandomGridConfig(**params)
                    grid_configs.append(GridConfig(random=config))
            elif "healpix" in grid:
                grid_data = grid["healpix"]
                param_combinations = get_parameter_combinations(grid_data)
                for params in param_combinations:
                    config = HealpixGridConfig(**params)
                    grid_configs.append(GridConfig(healpix=config))
            elif "custom" in grid:
                grid_data = grid["custom"]
                param_combinations = get_parameter_combinations(grid_data)
                for params in param_combinations:
                    config = CustomGridConfig(**params)
                    grid_configs.append(GridConfig(custom=config))
            else:
                # Handle old format with "type" field
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
                        raise PSRMagPyGridError(f"Unknown grid type: {grid_type}")
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


def expand_tracing_configs(
    tracing_config: Union[TracingConfig, TracingFactoryConfig]
) -> List[TracingConfig]:
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


def expand_template_configs(
    template_config: Union[FieldLine, TemplateFactoryConfig]
) -> List[FieldLine]:
    """Expand template configuration into all possible combinations."""
    if isinstance(template_config, FieldLine):
        return [template_config]

    # Handle TemplateFactoryConfig
    return template_config.templates
