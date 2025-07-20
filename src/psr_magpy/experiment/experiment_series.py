"""Experiment series configuration and generation.

This module provides functionality to create multiple experiments from flexible
parameter inputs, including lists and dictionaries for parameter variation.
"""

from typing import List, Optional, Union, Dict, Any, Tuple
from pydantic import BaseModel
import yaml
from pathlib import Path

from .experiment_config import (
    ExperimentConfig, FieldConfig, FieldComponentConfig, 
    GridConfig, SphericalGridConfig, RandomGridConfig, HealpixGridConfig, CustomGridConfig,
    TracingConfig
)
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


class ExperimentSeriesConfigFactory(BaseModel):
    """Factory configuration for experiment series with flexible parameter inputs."""
    
    name: str
    field: Union[FieldConfig, FieldFactoryConfig]
    grid: Union[GridConfig, GridFactoryConfig]
    tracing: Union[TracingConfig, TracingFactoryConfig]
    template: Union[FieldLine, TemplateFactoryConfig]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = {"arbitrary_types_allowed": True}
    
    def create_experiment_series(self) -> List[ExperimentConfig]:
        """Create all possible experiment combinations from the configuration."""
        logger.debug(f"Creating experiment series: {self.name}")
        
        # Generate all parameter combinations
        field_configs = self._expand_field_configs()
        grid_configs = self._expand_grid_configs()
        tracing_configs = self._expand_tracing_configs()
        template_configs = self._expand_template_configs()
        
        # Create all combinations
        experiments = []
        experiment_count = 0
        
        for field_config in field_configs:
            for grid_config in grid_configs:
                for tracing_config in tracing_configs:
                    for template_config in template_configs:
                        experiment_count += 1
                        
                        # Create unique name and metadata
                        exp_name = f"{self.name}_{experiment_count:03d}"
                        exp_metadata = self._create_experiment_metadata(
                            experiment_count, field_config, grid_config, tracing_config, template_config
                        )
                        
                        experiment = ExperimentConfig(
                            name=exp_name,
                            description=self.description,
                            field=field_config,
                            grid=grid_config,
                            tracing=tracing_config,
                            template=template_config,
                            metadata=exp_metadata
                        )
                        experiments.append(experiment)
        
        logger.info(f"Created {len(experiments)} experiments in series")
        return experiments
    
    def _expand_field_configs(self) -> List[FieldConfig]:
        """Expand field configuration into all possible combinations."""
        if isinstance(self.field, FieldConfig):
            return [self.field]
        
        # Handle FieldFactoryConfig
        field_configs = []
        for component_group in self.field.components:
            if isinstance(component_group, FieldComponentConfig):
                # Single component
                field_configs.append(FieldConfig(components=[component_group]))
            elif isinstance(component_group, list):
                # List of components
                field_configs.append(FieldConfig(components=component_group))
            elif isinstance(component_group, dict):
                # Dictionary input with flexible parameters
                param_combinations = self._get_parameter_combinations(component_group)
                
                for params in param_combinations:
                    config = FieldComponentConfig(**params)
                    field_configs.append(FieldConfig(components=[config]))
        
        return field_configs
    
    def _expand_grid_configs(self) -> List[GridConfig]:
        """Expand grid configuration into all possible combinations."""
        if isinstance(self.grid, GridConfig):
            return [self.grid]
        
        # Handle GridFactoryConfig
        grid_configs = []
        for grid in self.grid.grids:
            if isinstance(grid, SphericalGridFactoryConfig):
                # Expand spherical grid factory config
                param_combinations = self._get_parameter_combinations(grid.model_dump())
                for params in param_combinations:
                    config = SphericalGridConfig(**params)
                    grid_configs.append(GridConfig(spherical=config))
            elif isinstance(grid, RandomGridFactoryConfig):
                # Expand random grid factory config
                param_combinations = self._get_parameter_combinations(grid.model_dump())
                for params in param_combinations:
                    config = RandomGridConfig(**params)
                    grid_configs.append(GridConfig(random=config))
            elif isinstance(grid, HealpixGridFactoryConfig):
                # Expand HEALPix grid factory config
                param_combinations = self._get_parameter_combinations(grid.model_dump())
                for params in param_combinations:
                    config = HealpixGridConfig(**params)
                    grid_configs.append(GridConfig(healpix=config))
            elif isinstance(grid, CustomGridFactoryConfig):
                # Expand custom grid factory config
                param_combinations = self._get_parameter_combinations(grid.model_dump())
                for params in param_combinations:
                    config = CustomGridConfig(**params)
                    grid_configs.append(GridConfig(custom=config))
            elif isinstance(grid, dict):
                # Handle dictionary input with flexible parameters
                grid_type = grid.get("type", "spherical")
                # Expand parameter combinations
                param_combinations = self._get_parameter_combinations(grid)
                
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
    
    def _expand_tracing_configs(self) -> List[TracingConfig]:
        """Expand tracing configuration into all possible combinations."""
        if isinstance(self.tracing, TracingConfig):
            return [self.tracing]
        
        # Handle TracingFactoryConfig
        param_combinations = self._get_parameter_combinations(self.tracing.model_dump())
        
        tracing_configs = []
        for params in param_combinations:
            # Remove branch info from params before creating TracingConfig
            branch_info = params.pop('branch', None)
            tracing_config = TracingConfig(**params)
            if branch_info:
                # Add branch info as a custom attribute
                setattr(tracing_config, 'branch', branch_info)
            tracing_configs.append(tracing_config)
        
        return tracing_configs
    
    def _expand_template_configs(self) -> List[FieldLine]:
        """Expand template configuration into all possible combinations."""
        if isinstance(self.template, FieldLine):
            return [self.template]
        
        # Handle TemplateFactoryConfig
        return self.template.templates
    
    def _get_parameter_combinations(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters that contain lists or dictionaries."""
        # Find parameters that need expansion
        expandable_params = {}
        fixed_params = {}
        
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                expandable_params[key] = value
            else:
                fixed_params[key] = value
        
        if not expandable_params:
            return [params]
        
        # Generate combinations
        combinations = [fixed_params]
        
        for key, value in expandable_params.items():
            new_combinations = []
            
            if isinstance(value, list):
                # List: create combinations with each value
                for item in value:
                    for combo in combinations:
                        new_combo = combo.copy()
                        new_combo[key] = item
                        new_combinations.append(new_combo)
            
            elif isinstance(value, dict):
                # Dictionary: create separate branches for each key-value pair
                for dict_key, dict_value in value.items():
                    for combo in combinations:
                        new_combo = combo.copy()
                        new_combo[key] = dict_value
                        # Add branch info to metadata
                        if 'branch' not in new_combo:
                            new_combo['branch'] = {}
                        new_combo['branch'][key] = dict_key
                        new_combinations.append(new_combo)
            
            combinations = new_combinations
        
        return combinations
    
    def _create_experiment_metadata(self, experiment_count: int, field_config: FieldConfig, 
                                  grid_config: GridConfig, tracing_config: TracingConfig, 
                                  template_config: FieldLine) -> Dict[str, Any]:
        """Create unique metadata for each experiment."""
        metadata = self.metadata.copy() if self.metadata else {}
        
        # Add experiment-specific metadata
        metadata.update({
            "experiment_number": experiment_count,
            "series_name": self.name,
            "field_components": len(field_config.components),
            "grid_type": grid_config.get_grid_type(),
            "template_properties": list(template_config.property_system.get_flags_dict().keys())
        })
        
        # Add branch information if present
        if hasattr(tracing_config, 'branch'):
            metadata["branch"] = getattr(tracing_config, 'branch')
        
        return metadata
    
    def _remove_duplicate_experiments(self, experiments: List["ExperimentConfig"]) -> List["ExperimentConfig"]:
        """Remove duplicate experiments based on parameter sets."""
        logger.debug(f"Removing duplicates from {len(experiments)} experiments")
        
        unique_experiments = []
        seen_parameter_sets = set()
        
        for exp in experiments:
            # Create a parameter signature for comparison
            param_signature = self._create_parameter_signature(exp)
            
            if param_signature not in seen_parameter_sets:
                seen_parameter_sets.add(param_signature)
                unique_experiments.append(exp)
            else:
                logger.debug(f"Removing duplicate experiment: {exp.name}")
        
        logger.info(f"Removed {len(experiments) - len(unique_experiments)} duplicate experiments")
        return unique_experiments
    
    def _create_parameter_signature(self, exp: "ExperimentConfig") -> str:
        """Create a unique signature for an experiment's parameters."""
        # Field signature
        field_sig = []
        for comp in exp.field.components:
            field_sig.append(f"{comp.moment_type}_{comp.B_magnitude}_{comp.moment_orientation_latitude_degree}_{comp.moment_orientation_longitude_degree}")
        field_signature = "|".join(field_sig)
        
        # Grid signature
        grid_type = exp.grid.get_grid_type()
        if grid_type == "spherical" and exp.grid.spherical is not None:
            grid_signature = f"spherical_{exp.grid.spherical.nlon}_{exp.grid.spherical.nlat}_{exp.grid.spherical.pole_lat}"
        elif grid_type == "random" and exp.grid.random is not None:
            grid_signature = f"random_{exp.grid.random.nline_ori}"
        elif grid_type == "healpix" and exp.grid.healpix is not None:
            grid_signature = f"healpix_{exp.grid.healpix.nside}"
        elif grid_type == "custom" and exp.grid.custom is not None:
            grid_signature = f"custom_{exp.grid.custom.nline_ori}"
        else:
            grid_signature = f"unknown_{grid_type}"
        
        # Tracing signature
        tracing_signature = f"{exp.tracing.max_integration_radius}_{exp.tracing.dt}_{exp.tracing.n_cores}"
        
        # Template signature
        template_flags = exp.template.property_system.get_flags_dict()
        enabled_properties = [name for name, enabled in template_flags.items() if enabled]
        template_signature = "|".join(sorted(enabled_properties))
        
        # Combine all signatures
        full_signature = f"{field_signature}|{grid_signature}|{tracing_signature}|{template_signature}"
        return full_signature
    
    def print_parameter_analysis(self, experiments: List["ExperimentConfig"]) -> None:
        """Print detailed parameter analysis with color highlighting."""
        from psr_magpy.utils.logging_config import get_logger
        logger = get_logger(__name__, console_format_brief=True)
        
        logger.info(f"Analyzing parameters for {len(experiments)} experiments")
        
        # Collect all parameter values to identify common vs changing
        all_params = []
        for exp in experiments:
            # Get template properties
            template_flags = exp.template.property_system.get_flags_dict()
            enabled_properties = [name for name, enabled in template_flags.items() if enabled]
            
            params = {
                'field_components': len(exp.field.components),
                'grid_type': exp.grid.get_grid_type(),
                'template_properties': enabled_properties,
            }
            
            # Add field component details
            for i, comp in enumerate(exp.field.components):
                params[f'field_{i}_type'] = comp.moment_type
                params[f'field_{i}_B'] = comp.B_magnitude
                params[f'field_{i}_lat'] = comp.moment_orientation_latitude_degree
                params[f'field_{i}_lon'] = comp.moment_orientation_longitude_degree
            
            # Add grid details
            grid_type = exp.grid.get_grid_type()
            if grid_type == "spherical" and exp.grid.spherical is not None:
                params['grid_nlon'] = exp.grid.spherical.nlon
                params['grid_nlat'] = exp.grid.spherical.nlat
                params['grid_pole_lat'] = exp.grid.spherical.pole_lat
            elif grid_type == "random" and exp.grid.random is not None:
                params['grid_nline_ori'] = exp.grid.random.nline_ori
            
            # Add tracing details
            params['tracing_max_radius'] = exp.tracing.max_integration_radius
            params['tracing_dt'] = exp.tracing.dt
            params['tracing_n_cores'] = exp.tracing.n_cores
            
            all_params.append(params)
        
        # Identify common parameters
        common_params = {}
        changing_params = {}
        
        if all_params:
            # Get all unique keys from all experiments
            all_keys = set()
            for params in all_params:
                all_keys.update(params.keys())
            
            for key in all_keys:
                # Check if all experiments have this parameter
                values = []
                for params in all_params:
                    if key in params:
                        values.append(params[key])
                    else:
                        values.append(None)  # Mark missing parameters
                
                # Handle different types for comparison
                if key == 'template_properties':
                    # For template properties, compare as strings
                    value_strings = [str(v) if v is not None else None for v in values]
                    if len(set(value_strings)) == 1 and value_strings[0] is not None:
                        common_params[key] = values[0]
                    else:
                        changing_params[key] = values
                else:
                    # For other parameters, check if they're constant when present
                    non_none_values = [v for v in values if v is not None]
                    if len(non_none_values) > 0 and len(set(non_none_values)) == 1:
                        # Parameter is constant when present
                        if len(non_none_values) == len(values):
                            # Parameter present in all experiments
                            common_params[key] = non_none_values[0]
                        else:
                            # Parameter only present in some experiments (grid-type specific)
                            # Add to common params if it's constant within its type
                            common_params[key] = non_none_values[0]
                    else:
                        # Parameter actually varies
                        changing_params[key] = values
        
        # Color codes
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        END = '\033[0m'
        
        print(f"{BOLD}Common parameters (same for all experiments):{END}")
        for key, value in common_params.items():
            if key == 'template_properties':
                # Special handling for template properties
                if len(value) == 8:  # All properties enabled
                    print(f"  {GREEN}✓{END} {key}: all")
                else:
                    print(f"  {GREEN}✓{END} {key}: {value}")
            else:
                print(f"  {GREEN}✓{END} {key}: {value}")
        print()
        
        print(f"{BOLD}Changing parameters:{END}")
        for key, values in changing_params.items():
            if key == 'template_properties':
                # Special handling for template properties
                unique_values = []
                for v in values:
                    if v is not None:
                        if len(v) == 8:  # All properties enabled
                            unique_values.append("all")
                        else:
                            unique_values.append(v)
                unique_values = list(set(str(v) for v in unique_values))
                if len(unique_values) <= 3:
                    print(f"  {YELLOW}↻{END} {key}: {unique_values}")
                else:
                    print(f"  {YELLOW}↻{END} {key}: {len(unique_values)} different configurations")
            else:
                unique_values = list(set(v for v in values if v is not None))
                if len(unique_values) <= 5:
                    print(f"  {YELLOW}↻{END} {key}: {unique_values}")
                else:
                    print(f"  {YELLOW}↻{END} {key}: {len(unique_values)} different values")
        print()
        
        print(f"{BOLD}Individual experiment details:{END}")
        for i, exp in enumerate(experiments):
            print(f"  {BOLD}{i+1:2d}. {exp.name}{END}")
            
            # Field details
            print(f"    {BLUE}Field:{END}")
            for j, comp in enumerate(exp.field.components):
                type_color = GREEN if all_params[i][f'field_{j}_type'] == common_params.get(f'field_{j}_type', 'DIFFERENT') else YELLOW
                B_color = GREEN if all_params[i][f'field_{j}_B'] == common_params.get(f'field_{j}_B', 'DIFFERENT') else YELLOW
                lat_color = GREEN if all_params[i][f'field_{j}_lat'] == common_params.get(f'field_{j}_lat', 'DIFFERENT') else YELLOW
                lon_color = GREEN if all_params[i][f'field_{j}_lon'] == common_params.get(f'field_{j}_lon', 'DIFFERENT') else YELLOW
                
                print(f"      Component {j+1}: {type_color}{comp.moment_type}{END} "
                      f"B={B_color}{comp.B_magnitude}{END} "
                      f"lat={lat_color}{comp.moment_orientation_latitude_degree}°{END} "
                      f"lon={lon_color}{comp.moment_orientation_longitude_degree}°{END}")
            
            # Grid details
            print(f"    {BLUE}Grid:{END}")
            grid_type = exp.grid.get_grid_type()
            grid_type_color = GREEN if all_params[i]['grid_type'] == common_params.get('grid_type', 'DIFFERENT') else YELLOW
            print(f"      Type: {grid_type_color}{grid_type}{END}")
            
            if grid_type == "spherical" and exp.grid.spherical is not None:
                nlon_color = GREEN if all_params[i]['grid_nlon'] == common_params.get('grid_nlon', 'DIFFERENT') else YELLOW
                nlat_color = GREEN if all_params[i]['grid_nlat'] == common_params.get('grid_nlat', 'DIFFERENT') else YELLOW
                pole_color = GREEN if all_params[i]['grid_pole_lat'] == common_params.get('grid_pole_lat', 'DIFFERENT') else YELLOW
                print(f"      nlon: {nlon_color}{exp.grid.spherical.nlon}{END} "
                      f"nlat: {nlat_color}{exp.grid.spherical.nlat}{END} "
                      f"pole_lat: {pole_color}{exp.grid.spherical.pole_lat}°{END}")
            elif grid_type == "random" and exp.grid.random is not None:
                nline_color = GREEN if all_params[i]['grid_nline_ori'] == common_params.get('grid_nline_ori', 'DIFFERENT') else YELLOW
                print(f"      nline_ori: {nline_color}{exp.grid.random.nline_ori}{END}")
            
            # Template details
            template_props = all_params[i]['template_properties']
            template_color = GREEN if template_props == common_params.get('template_properties', 'DIFFERENT') else YELLOW
            if len(template_props) == 8:  # All properties enabled
                print(f"    {BLUE}Template:{END} {template_color}all{END}")
            else:
                print(f"    {BLUE}Template:{END} {template_color}{template_props}{END}")
            
            # Tracing details
            print(f"    {BLUE}Tracing:{END}")
            radius_color = GREEN if all_params[i]['tracing_max_radius'] == common_params.get('tracing_max_radius', 'DIFFERENT') else YELLOW
            dt_color = GREEN if all_params[i]['tracing_dt'] == common_params.get('tracing_dt', 'DIFFERENT') else YELLOW
            cores_color = GREEN if all_params[i]['tracing_n_cores'] == common_params.get('tracing_n_cores', 'DIFFERENT') else YELLOW
            print(f"      max_radius: {radius_color}{exp.tracing.max_integration_radius}{END} "
                  f"dt: {dt_color}{exp.tracing.dt}{END} "
                  f"n_cores: {cores_color}{exp.tracing.n_cores}{END}")
            
            print() 


class ExperimentSeriesConfig(BaseModel):
    """Configuration for experiment series that can hold multiple factories with duplicate removal."""
    
    factories: Union[ExperimentSeriesConfigFactory, List[ExperimentSeriesConfigFactory]]
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = {"arbitrary_types_allowed": True}
    
    def create_experiment_series(self) -> List[ExperimentConfig]:
        """Create all experiments from factories with duplicate removal."""
        logger.info(f"Creating experiment series from {self._get_factory_count()} factories")
        
        # Convert to list if single factory
        if isinstance(self.factories, ExperimentSeriesConfigFactory):
            factory_list = [self.factories]
        else:
            factory_list = self.factories
        
        # Create experiments from all factories
        all_experiments = []
        for factory in factory_list:
            experiments = factory.create_experiment_series()
            all_experiments.extend(experiments)
        
        # Remove duplicates
        unique_experiments = self._remove_duplicate_experiments(all_experiments)
        
        logger.info(f"Created {len(unique_experiments)} unique experiments from {len(all_experiments)} total")
        return unique_experiments
    
    def _get_factory_count(self) -> int:
        """Get the number of factories."""
        if isinstance(self.factories, ExperimentSeriesConfigFactory):
            return 1
        else:
            return len(self.factories)
    
    def _remove_duplicate_experiments(self, experiments: List[ExperimentConfig]) -> List[ExperimentConfig]:
        """Remove duplicate experiments based on parameter sets."""
        logger.debug(f"Removing duplicates from {len(experiments)} experiments")
        
        # Print parameters for all experiments before duplicate removal
        self._print_duplicate_analysis(experiments)
        
        unique_experiments = []
        seen_parameter_sets = set()
        
        for exp in experiments:
            # Create a parameter signature for comparison
            param_signature = self._create_parameter_signature(exp)
            
            if param_signature not in seen_parameter_sets:
                seen_parameter_sets.add(param_signature)
                unique_experiments.append(exp)
            else:
                logger.debug(f"Removing duplicate experiment: {exp.name}")
        
        logger.info(f"Removed {len(experiments) - len(unique_experiments)} duplicate experiments")
        return unique_experiments
    
    def _print_duplicate_analysis(self, experiments: List[ExperimentConfig]) -> None:
        """Print detailed analysis of experiments to identify duplicates."""
        logger.info(f"Analyzing {len(experiments)} experiments for duplicates:")
        
        # Group experiments by parameter signature
        signature_groups = {}
        for exp in experiments:
            param_signature = self._create_parameter_signature(exp)
            if param_signature not in signature_groups:
                signature_groups[param_signature] = []
            signature_groups[param_signature].append(exp)
        
        # Print groups with multiple experiments (duplicates)
        duplicate_count = 0
        for signature, exp_group in signature_groups.items():
            if len(exp_group) > 1:
                duplicate_count += len(exp_group) - 1
                logger.info(f"Duplicate group ({len(exp_group)} experiments):")
                
                # Print signature components
                # The signature format is: field_signature|grid_signature|tracing_signature|template_signature
                # But field_signature can contain multiple components separated by |
                # So we need to parse it more carefully
                
                # Find the grid signature (starts with grid type)
                grid_start = -1
                for i, part in enumerate(signature.split('|')):
                    if part.startswith(('spherical_', 'random_', 'healpix_', 'custom_', 'unknown_')):
                        grid_start = i
                        break
                
                if grid_start >= 0:
                    # Field signature is everything before grid
                    field_signature = '|'.join(signature.split('|')[:grid_start])
                    # Grid signature
                    grid_signature = signature.split('|')[grid_start]
                    # Tracing signature (format: radius_dt_cores)
                    tracing_signature = signature.split('|')[grid_start + 1]
                    # Template signature is everything after tracing
                    template_signature = '|'.join(signature.split('|')[grid_start + 2:])
                    
                    logger.info(f"  Field: {field_signature}")
                    logger.info(f"  Grid: {grid_signature}")
                    logger.info(f"  Tracing: {tracing_signature}")
                    logger.info(f"  Template: {template_signature}")
                else:
                    # Fallback to simple split if grid type not found
                    signature_parts = signature.split('|')
                    logger.info(f"  Field: {signature_parts[0]}")
                    logger.info(f"  Grid: {signature_parts[1] if len(signature_parts) > 1 else 'unknown'}")
                    logger.info(f"  Tracing: {signature_parts[2] if len(signature_parts) > 2 else 'unknown'}")
                    logger.info(f"  Template: {signature_parts[3] if len(signature_parts) > 3 else 'unknown'}")
                
                # Print experiment names in this group
                for i, exp in enumerate(exp_group):
                    logger.info(f"    {i+1}. {exp.name}")
                logger.info("")
        
        if duplicate_count == 0:
            logger.info("No duplicates found.")
        else:
            logger.info(f"Found {duplicate_count} duplicate experiments in {len([g for g in signature_groups.values() if len(g) > 1])} groups.")
    
    def _create_parameter_signature(self, exp: ExperimentConfig) -> str:
        """Create a unique signature for an experiment's parameters."""
        # Field signature
        field_sig = []
        for comp in exp.field.components:
            field_sig.append(f"{comp.moment_type}_{comp.B_magnitude}_{comp.moment_orientation_latitude_degree}_{comp.moment_orientation_longitude_degree}")
        field_signature = "|".join(field_sig)
        
        # Grid signature
        grid_type = exp.grid.get_grid_type()
        if grid_type == "spherical" and exp.grid.spherical is not None:
            grid_signature = f"spherical_{exp.grid.spherical.nlon}_{exp.grid.spherical.nlat}_{exp.grid.spherical.pole_lat}"
        elif grid_type == "random" and exp.grid.random is not None:
            grid_signature = f"random_{exp.grid.random.nline_ori}"
        elif grid_type == "healpix" and exp.grid.healpix is not None:
            grid_signature = f"healpix_{exp.grid.healpix.nside}"
        elif grid_type == "custom" and exp.grid.custom is not None:
            grid_signature = f"custom_{exp.grid.custom.nline_ori}"
        else:
            grid_signature = f"unknown_{grid_type}"
        
        # Tracing signature
        tracing_signature = f"{exp.tracing.max_integration_radius}_{exp.tracing.dt}_{exp.tracing.n_cores}"
        
        # Template signature
        template_flags = exp.template.property_system.get_flags_dict()
        enabled_properties = [name for name, enabled in template_flags.items() if enabled]
        template_signature = "|".join(sorted(enabled_properties))
        
        # Combine all signatures
        full_signature = f"{field_signature}|{grid_signature}|{tracing_signature}|{template_signature}"
        return full_signature
    
    def print_parameter_analysis(self, experiments: List[ExperimentConfig]) -> None:
        """Print detailed parameter analysis with color highlighting."""
        from psr_magpy.utils.logging_config import get_logger
        logger = get_logger(__name__, console_format_brief=True)
        
        logger.info(f"Analyzing parameters for {len(experiments)} experiments")
        
        # Collect all parameter values to identify common vs changing
        all_params = []
        for exp in experiments:
            # Get template properties
            template_flags = exp.template.property_system.get_flags_dict()
            enabled_properties = [name for name, enabled in template_flags.items() if enabled]
            
            params = {
                'field_components': len(exp.field.components),
                'grid_type': exp.grid.get_grid_type(),
                'template_properties': enabled_properties,
            }
            
            # Add field component details
            for i, comp in enumerate(exp.field.components):
                params[f'field_{i}_type'] = comp.moment_type
                params[f'field_{i}_B'] = comp.B_magnitude
                params[f'field_{i}_lat'] = comp.moment_orientation_latitude_degree
                params[f'field_{i}_lon'] = comp.moment_orientation_longitude_degree
            
            # Add grid details
            grid_type = exp.grid.get_grid_type()
            if grid_type == "spherical" and exp.grid.spherical is not None:
                params['grid_nlon'] = exp.grid.spherical.nlon
                params['grid_nlat'] = exp.grid.spherical.nlat
                params['grid_pole_lat'] = exp.grid.spherical.pole_lat
            elif grid_type == "random" and exp.grid.random is not None:
                params['grid_nline_ori'] = exp.grid.random.nline_ori
            
            # Add tracing details
            params['tracing_max_radius'] = exp.tracing.max_integration_radius
            params['tracing_dt'] = exp.tracing.dt
            params['tracing_n_cores'] = exp.tracing.n_cores
            
            all_params.append(params)
        
        # Identify common parameters
        common_params = {}
        changing_params = {}
        
        if all_params:
            # Get all unique keys from all experiments
            all_keys = set()
            for params in all_params:
                all_keys.update(params.keys())
            
            for key in all_keys:
                # Check if all experiments have this parameter
                values = []
                for params in all_params:
                    if key in params:
                        values.append(params[key])
                    else:
                        values.append(None)  # Mark missing parameters
                
                # Handle different types for comparison
                if key == 'template_properties':
                    # For template properties, compare as strings
                    value_strings = [str(v) if v is not None else None for v in values]
                    if len(set(value_strings)) == 1 and value_strings[0] is not None:
                        common_params[key] = values[0]
                    else:
                        changing_params[key] = values
                else:
                    # For other parameters, check if they're constant when present
                    non_none_values = [v for v in values if v is not None]
                    if len(non_none_values) > 0 and len(set(non_none_values)) == 1:
                        # Parameter is constant when present
                        if len(non_none_values) == len(values):
                            # Parameter present in all experiments
                            common_params[key] = non_none_values[0]
                        else:
                            # Parameter only present in some experiments (grid-type specific)
                            # Add to common params if it's constant within its type
                            common_params[key] = non_none_values[0]
                    else:
                        # Parameter actually varies
                        changing_params[key] = values
        
        # Color codes
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        END = '\033[0m'
        
        print(f"{BOLD}Common parameters (same for all experiments):{END}")
        for key, value in common_params.items():
            if key == 'template_properties':
                # Special handling for template properties
                if len(value) == 8:  # All properties enabled
                    print(f"  {GREEN}✓{END} {key}: all")
                else:
                    print(f"  {GREEN}✓{END} {key}: {value}")
            else:
                print(f"  {GREEN}✓{END} {key}: {value}")
        print()
        
        print(f"{BOLD}Changing parameters:{END}")
        for key, values in changing_params.items():
            if key == 'template_properties':
                # Special handling for template properties
                unique_values = []
                for v in values:
                    if v is not None:
                        if len(v) == 8:  # All properties enabled
                            unique_values.append("all")
                        else:
                            unique_values.append(v)
                unique_values = list(set(str(v) for v in unique_values))
                if len(unique_values) <= 3:
                    print(f"  {YELLOW}↻{END} {key}: {unique_values}")
                else:
                    print(f"  {YELLOW}↻{END} {key}: {len(unique_values)} different configurations")
            else:
                unique_values = list(set(v for v in values if v is not None))
                if len(unique_values) <= 5:
                    print(f"  {YELLOW}↻{END} {key}: {unique_values}")
                else:
                    print(f"  {YELLOW}↻{END} {key}: {len(unique_values)} different values")
        print()
        
        print(f"{BOLD}Individual experiment details:{END}")
        for i, exp in enumerate(experiments):
            print(f"  {BOLD}{i+1:2d}. {exp.name}{END}")
            
            # Field details
            print(f"    {BLUE}Field:{END}")
            for j, comp in enumerate(exp.field.components):
                type_color = GREEN if all_params[i][f'field_{j}_type'] == common_params.get(f'field_{j}_type', 'DIFFERENT') else YELLOW
                B_color = GREEN if all_params[i][f'field_{j}_B'] == common_params.get(f'field_{j}_B', 'DIFFERENT') else YELLOW
                lat_color = GREEN if all_params[i][f'field_{j}_lat'] == common_params.get(f'field_{j}_lat', 'DIFFERENT') else YELLOW
                lon_color = GREEN if all_params[i][f'field_{j}_lon'] == common_params.get(f'field_{j}_lon', 'DIFFERENT') else YELLOW
                
                print(f"      Component {j+1}: {type_color}{comp.moment_type}{END} "
                      f"B={B_color}{comp.B_magnitude}{END} "
                      f"lat={lat_color}{comp.moment_orientation_latitude_degree}°{END} "
                      f"lon={lon_color}{comp.moment_orientation_longitude_degree}°{END}")
            
            # Grid details
            print(f"    {BLUE}Grid:{END}")
            grid_type = exp.grid.get_grid_type()
            grid_type_color = GREEN if all_params[i]['grid_type'] == common_params.get('grid_type', 'DIFFERENT') else YELLOW
            print(f"      Type: {grid_type_color}{grid_type}{END}")
            
            if grid_type == "spherical" and exp.grid.spherical is not None:
                nlon_color = GREEN if all_params[i]['grid_nlon'] == common_params.get('grid_nlon', 'DIFFERENT') else YELLOW
                nlat_color = GREEN if all_params[i]['grid_nlat'] == common_params.get('grid_nlat', 'DIFFERENT') else YELLOW
                pole_color = GREEN if all_params[i]['grid_pole_lat'] == common_params.get('grid_pole_lat', 'DIFFERENT') else YELLOW
                print(f"      nlon: {nlon_color}{exp.grid.spherical.nlon}{END} "
                      f"nlat: {nlat_color}{exp.grid.spherical.nlat}{END} "
                      f"pole_lat: {pole_color}{exp.grid.spherical.pole_lat}°{END}")
            elif grid_type == "random" and exp.grid.random is not None:
                nline_color = GREEN if all_params[i]['grid_nline_ori'] == common_params.get('grid_nline_ori', 'DIFFERENT') else YELLOW
                print(f"      nline_ori: {nline_color}{exp.grid.random.nline_ori}{END}")
            
            # Template details
            template_props = all_params[i]['template_properties']
            template_color = GREEN if template_props == common_params.get('template_properties', 'DIFFERENT') else YELLOW
            if len(template_props) == 8:  # All properties enabled
                print(f"    {BLUE}Template:{END} {template_color}all{END}")
            else:
                print(f"    {BLUE}Template:{END} {template_color}{template_props}{END}")
            
            # Tracing details
            print(f"    {BLUE}Tracing:{END}")
            radius_color = GREEN if all_params[i]['tracing_max_radius'] == common_params.get('tracing_max_radius', 'DIFFERENT') else YELLOW
            dt_color = GREEN if all_params[i]['tracing_dt'] == common_params.get('tracing_dt', 'DIFFERENT') else YELLOW
            cores_color = GREEN if all_params[i]['tracing_n_cores'] == common_params.get('tracing_n_cores', 'DIFFERENT') else YELLOW
            print(f"      max_radius: {radius_color}{exp.tracing.max_integration_radius}{END} "
                  f"dt: {dt_color}{exp.tracing.dt}{END} "
                  f"n_cores: {cores_color}{exp.tracing.n_cores}{END}")
            
            print()
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save the experiment series configuration to a YAML file."""
        filepath = Path(filepath)
        logger.info(f"Saving experiment series config to {filepath}")
        
        # Convert to dict for YAML serialization
        config_dict = self.model_dump()
        
        # Write to YAML file
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentSeriesConfig":
        """Load an experiment series configuration from a YAML file."""
        filepath = Path(filepath)
        logger.info(f"Loading experiment series config from {filepath}")
        
        # Read YAML file
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested objects if needed
        config_dict = cls._reconstruct_from_yaml(config_dict)
        
        # Create config object
        config = cls(**config_dict)
        logger.info(f"Config loaded from {filepath}")
        return config
    
    @classmethod
    def _reconstruct_from_yaml(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct nested objects from YAML data."""
        if "factories" in data:
            factories_data = data["factories"]
            if isinstance(factories_data, list):
                # Multiple factories
                factories = []
                for factory_data in factories_data:
                    factory_data = cls._reconstruct_factory_data(factory_data)
                    factories.append(factory_data)
                data["factories"] = factories
            else:
                # Single factory
                data["factories"] = cls._reconstruct_factory_data(factories_data)
        
        return data
    
    @classmethod
    def _reconstruct_factory_data(cls, factory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct individual factory data."""
        # Reconstruct template
        if "template" in factory_data and isinstance(factory_data["template"], dict):
            factory_data["template"] = TemplateFactoryConfig.from_yaml_dict(factory_data["template"])
        
        # Reconstruct field
        if "field" in factory_data and isinstance(factory_data["field"], dict):
            factory_data["field"] = FieldFactoryConfig(**factory_data["field"])
        
        # Reconstruct grid with nested factory configs
        if "grid" in factory_data and isinstance(factory_data["grid"], dict):
            grid_data = factory_data["grid"]
            if "grids" in grid_data and isinstance(grid_data["grids"], list):
                # Reconstruct nested grid factory configs
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
            factory_data["grid"] = GridFactoryConfig(**grid_data)
        
        # Reconstruct tracing
        if "tracing" in factory_data and isinstance(factory_data["tracing"], dict):
            factory_data["tracing"] = TracingFactoryConfig(**factory_data["tracing"])
        
        return factory_data