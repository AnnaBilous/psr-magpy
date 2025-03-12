# %% Imports
from dataclasses import dataclass, asdict, replace, field
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import itertools
import copy
import json
import numpy as np
from copy import deepcopy
# %% Class definitions
@dataclass
class FieldComponentConfig:
    """Configuration for a single magnetic field component."""
    moment_type: Union[str, List, Dict]  # 'dipole', 'quadrupole', etc.
    B_magnitude: Union[float, List, Dict]
    lat_degree: Union[float, List, Dict] = 0
    lon_degree: Union[float, List, Dict] = 0
    offset_r_ns: Union[np.ndarray, List, Dict] = field(default_factory=lambda: np.zeros(3))


@dataclass
class MultipoleFieldConfig:
    """Configuration for magnetic field with multiple components.
    
    Initialize with one or more sets of components, where each set is a list:
    - One set: MultipoleFieldConfig([dipole_component, quadrupole_component])
    - Multiple sets: MultipoleFieldConfig([set1_components], [set2_components])
    
    Examples
    --------
    # Single component in one set
    config1 = MultipoleFieldConfig([dipole_component])
    
    # Multiple components in one set
    config2 = MultipoleFieldConfig([dipole_component, quadrupole_component])
    
    # Multiple sets of components
    config3 = MultipoleFieldConfig(
        [dipole1, quadrupole1],  # First set
        [dipole2, quadrupole2]   # Second set
    )
    """
    components: List[List[FieldComponentConfig]]
    
    def __init__(self, *args: List[FieldComponentConfig]):
        if len(args) == 0:
            self.components = [{}]
            return
            
        # Validate that each argument is a list
        for arg in args:
            if not isinstance(arg, list):
                raise ValueError(
                    f"Each argument must be a list of components, got {type(arg)}. "
                    "Use: MultipoleFieldConfig([components]) for single set"
                )
        self.components = args
          
        

    def _expand_parameters(self) -> 'MultipoleFieldConfig':
        """Expand parameters into all possible combinations.
        
        For each component in a set:
        1. Collect all parameter ranges (lists) and parameter sets (dicts)
        2. Generate all possible combinations of values
        3. Create new component configs with specific values
        
        Returns
        -------
        MultipoleFieldConfig
            New config with expanded parameters, where each set contains
            components with specific parameter values
        """
        
        component_dicts = [
            {f"component_{i}": comp for i, comp in enumerate(c)}
            for c in self.components
        ]
        
        whole_expanded_component_sets = []
        # Process each set of components separately
        for component_set in component_dicts:
            print(component_set)
            
            expanded_components = self._expand_single_component_set(component_set)

            whole_expanded_components = []
            
            for component in expanded_components:
                modified_component = deepcopy(component_set)
                for i, comp_params in enumerate(component):
                    modified_component["component_" + str(i)] = replace(modified_component["component_" + str(i)], **comp_params)
                modified_component_set = [v for v in modified_component.values()]
                whole_expanded_component_sets.append(modified_component_set)
          
 
        return whole_expanded_component_sets


    def _expand_single_component_set(self, component_set: Dict[str, FieldComponentConfig]) -> List[FieldComponentConfig]:
        """Expand a single set of components into all possible combinations.
        
        Parameters
        ----------
        component_set : List[FieldComponentConfig]
            A single set of components to expand
        """
    
        # Process each component in the set
        param_ranges, param_sets = self._collect_parameters(component_set)
        
        number_of_components = len(component_set)
        
        if param_ranges:
            expanded_ranges = self._expand_parameter_ranges(param_ranges)
        if param_sets:
            expanded_sets = self._expand_parameter_sets(param_sets)
            
        if param_ranges and param_sets:
            expanded_components = [
                {**range_combo, **set_combo}
                for range_combo in expanded_ranges
                for set_combo in expanded_sets
            ]
        elif param_ranges and not param_sets:
            expanded_components = expanded_ranges
        elif not param_ranges and param_sets:
            expanded_components = expanded_sets
        else:
            expanded_components = [component_set]
            
        expanded_components_formatted = []    
        for expanded_component in expanded_components:
            result = []
            for i in range(number_of_components):
                component_dict = {}
                # Extract parameters for this component
                prefix = f'component_{i}_'
                for full_key in expanded_component:
                    if full_key.startswith(prefix):
                        # Remove the component prefix to get the parameter name
                        param_name = '_'.join(full_key.split('_')[2:])
                        component_dict[param_name] = expanded_component[full_key]
                result.append(component_dict)
            expanded_components_formatted.append(result)
        
                
        return expanded_components_formatted
        

    def _collect_parameters(self, component_dict: Dict[str, FieldComponentConfig]) -> Tuple[Dict, Dict]:
        """Collect parameter ranges and sets from a component.
        
        Parameters
        ----------
        component : FieldComponentConfig
            Component configuration to collect parameters from
            
        Returns
        -------
        Tuple[Dict, Dict]
            Parameter ranges and sets
        """ 
        param_ranges = {}  # name -> list of values
        param_sets = {}    # name -> dict of values
        
        for component_name, component in component_dict.items():
            for param_name, param_value in asdict(component).items():
                expanded_param_name = component_name + "_" + param_name
                if isinstance(param_value, list) and not isinstance(param_value, np.ndarray):
                    param_ranges[expanded_param_name] = param_value
                elif isinstance(param_value, dict):
                    param_sets[expanded_param_name] = param_value

        return param_ranges, param_sets
    
    def _expand_parameter_sets(self, param_sets: Dict) -> List[Dict]:
        """Expand parameter sets into all possible combinations.
        
        Parameters
        ----------
        param_sets : Dict
        """
        set_keys = set.intersection(
                *[set(param_dict.keys()) for param_dict in param_sets.values()]
            )
        
        set_names = list(param_sets.keys())

   
        set_combinations = [{name: param_sets[name][key] for name in set_names} for key in set_keys]

        return set_combinations
        
    def _expand_parameter_ranges(self, param_ranges: Dict) -> List[Dict]:
        """Expand parameter ranges into all possible combinations.
        
        Parameters
        ----------
        param_ranges : Dict
        """
        # Prepare all parameter combinations
        range_names = list(param_ranges.keys())
        range_values = list(param_ranges.values())
        # Generate all combinations

        combinations = []
        for values in itertools.product(*range_values):
            combination = {
                name: value 
                for name, value in zip(range_names, values)
            }
            combinations.append(combination)
            
        return combinations

    

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    field_config: MultipoleFieldConfig
    output_dir: Path
    save_vars: List[str]
    experiment_name: str
    filename_template: str = ""
    

class ExperimentSetup:
    """Manages experiment parameter combinations and file naming."""
    
    def __init__(self, base_config: ExperimentConfig):
        """Initialize experiment setup.
        
        Parameters
        ----------
        base_config : ExperimentConfig
            Base configuration with default values
        """
        self.base_config = base_config
    

    def generate_configs(self) -> List[Tuple[ExperimentConfig, str]]:
        """Generate experiment configurations by expanding parameter sets and ranges.
        
        Returns
        -------
        List[Tuple[ExperimentConfig, str]]
            List of (config, filename) tuples for each experiment
        """
        # First, expand parameter sets
        field_configs = self._expand_parameter_sets(self.base_config.field_config)
        
        # Then, expand parameter ranges
        field_configs = self._expand_parameter_ranges(field_configs)
        
        # Create experiment configs with each expanded field config
        experiment_configs = []
        for field_config in field_configs:
            # Create a copy of the base config with the new field config
            config_dict = asdict(self.base_config)
            config_dict['field_config'] = field_config
            experiment_config = ExperimentConfig.from_dict(config_dict)
            
            # Generate filename
            filename = self._generate_filename(experiment_config)
            
            experiment_configs.append((experiment_config, filename))
        
        return experiment_configs
    
    def _generate_filename(self, config: ExperimentConfig) -> str:
        """Generate a filename for an experiment configuration.
        
        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration
            
        Returns
        -------
        str
            Generated filename
        """
        if config.filename_template:
            # Use the template with format string substitution
            field_dict = asdict(config.field_config)
            filename = config.filename_template.format(**field_dict)
        else:
            # Generate a default filename
            field_dict = asdict(config.field_config)
            parts = []
            
            for component in field_dict['components']:
                for key, value in sorted(asdict(component).items()):
                    if not isinstance(value, (dict, list, np.ndarray)):
                        parts.append(f"{key}_{value}")
            
            if parts:
                filename = f"{config.experiment_name}_{'_'.join(parts)}.npz"
            else:
                filename = f"{config.experiment_name}.npz"
        
        return filename
  
  
# %% Test data setup
dipole = FieldComponentConfig(
    moment_type="dipole",
    B_magnitude={"1": 1.0, "2": 2.0},  # parameter set
    lat_degree=0,  # parameter range
    lon_degree=0
)

quadrupole = FieldComponentConfig(
    moment_type="quadrupole",
    B_magnitude=1.0, #{"1": 0.5, "2": 1.0},  # parameter set
    lat_degree=0,  # parameter range
    lon_degree=0
)

# %% Create and test configuration
# Create config with different component sets
field_config = MultipoleFieldConfig([dipole, quadrupole])

# Expand parameters
expanded = field_config._expand_parameters()
for s in expanded:
    print(s)


# %% Additional tests as needed
# Test specific parameter combinations
# Add more test cells as needed