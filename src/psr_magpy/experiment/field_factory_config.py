from dataclasses import dataclass, asdict, replace, field
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import itertools
import numpy as np
from copy import deepcopy
import json
import hashlib
from datetime import datetime

from src.psr_magpy.field.components import MultipoleType, Parameters
@dataclass
class FieldComponentFactory:
    """Configuration for a single magnetic field component."""
    moment_type: Union[str, List, Dict]  
    B_magnitude: Union[float, List, Dict]
    moment_orientation_latitude_degree: Union[float, List, Dict] = 0
    moment_orientation_longitude_degree: Union[float, List, Dict] = 0
    offset_r_ns: Union[np.ndarray, List, Dict] = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        """Convert moment_type to MultipoleType if it's a string."""
        if isinstance(self.moment_type, str):
            self.moment_type = MultipoleType(self.moment_type.lower())
        elif isinstance(self.moment_type, list):
            self.moment_type = [MultipoleType(mt.lower()) if isinstance(mt, str) else mt for mt in self.moment_type]
        elif isinstance(self.moment_type, dict):
            self.moment_type = {k: MultipoleType(v.lower()) if isinstance(v, str) else v 
                              for k, v in self.moment_type.items()}

    # Move to json serialization
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        d = asdict(self)
        if isinstance(d['moment_type'], MultipoleType):
            d['moment_type'] = d['moment_type'].value
        elif isinstance(d['moment_type'], list):
            d['moment_type'] = [mt.value if isinstance(mt, MultipoleType) else mt for mt in d['moment_type']]
        elif isinstance(d['moment_type'], dict):
            d['moment_type'] = {k: v.value if isinstance(v, MultipoleType) else v for k, v in d['moment_type'].items()}
        
        if isinstance(d['offset_r_ns'], np.ndarray):
            d['offset_r_ns'] = d['offset_r_ns'].tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldComponentFactory':
        """Create a FieldComponentFactory instance from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the field component configuration
            
        Returns
        -------
        FieldComponentFactory
            A new FieldComponentFactory instance
        """
        # Handle offset_r_ns if it's a list
        if 'offset_r_ns' in data and isinstance(data['offset_r_ns'], list):
            data['offset_r_ns'] = np.array(data['offset_r_ns'])
            
        # Create a new instance
        return cls(**data)

class MultipoleFieldFactory:
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
    components: List[List[FieldComponentFactory]]
    
    def __init__(self, *args: List[FieldComponentFactory]):
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
          
   

    def _convert_to_parameters(self, component: FieldComponentFactory) -> Parameters:

        return Parameters(
            B_magnitude=component.B_magnitude,
            moment_orientation_latitude_degree=component.moment_orientation_latitude_degree,
            moment_orientation_longitude_degree=component.moment_orientation_longitude_degree,
            offset_r_ns=component.offset_r_ns,
            moment_type=component.moment_type
        )

    def _expand_parameters(self) -> List[List[Parameters]]:
        """Expand parameters into all possible combinations.
        
        Returns
        -------
        List[List[Parameters]]
            List of component sets, where each set is a list of Parameters objects
        """
       
        component_dicts = [
            {f"component_{i}": comp for i, comp in enumerate(c)}
            for c in self.components
        ]
        
        whole_expanded_component_sets = []
        # Process each set of components separately
        for component_set in component_dicts:
                      
            expanded_components = self._expand_single_component_set(component_set)
            
            for component in expanded_components:
                modified_component = deepcopy(component_set)
                for i, comp_params in enumerate(component):
                    modified_component["component_" + str(i)] = replace(modified_component["component_" + str(i)], **comp_params)
                modified_component_set = [v for v in modified_component.values()]
                whole_expanded_component_sets.append(modified_component_set)
          
         
        return whole_expanded_component_sets


    def _expand_single_component_set(self, component_set: Dict[str, FieldComponentFactory]) -> List[FieldComponentFactory]:
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
        

    def _collect_parameters(self, component_dict: Dict[str, FieldComponentFactory]) -> Tuple[Dict, Dict]:
        """Collect parameter ranges and sets from a component.
        
        Parameters
        ----------
        component : FieldComponentFactory
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

        set_keys_union = set.union(*[set(param_dict.keys()) for param_dict in param_sets.values()])
        set_keys_intersection = set.intersection(*[set(param_dict.keys()) for param_dict in param_sets.values()])
        if set_keys_intersection != set_keys_union:
            raise ValueError("Parameter sets have different keys. Perhaps use ranges instead?")
        
        set_names = list(param_sets.keys())
     
   
        set_combinations = [{name: param_sets[name][key] for name in set_names} for key in set_keys_union]

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        components_dict = []
        for component_set in self.components:
            component_set_dict = []
            for comp in component_set:
                if isinstance(comp, FieldComponentFactory):
                    component_set_dict.append(comp.to_dict())
                elif isinstance(comp, dict):
                    component_set_dict.append(comp)
                elif isinstance(comp, list):
                    # Handle the case where comp is a list
                    component_set_dict.append([c.to_dict() if hasattr(c, 'to_dict') else c for c in comp])
                else:
                    # For other types, just use the value as is
                    component_set_dict.append(comp)
            components_dict.append(component_set_dict)
        
        return {
            'components': components_dict
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultipoleFieldFactory':
        """Create config from dictionary."""
        components = [[FieldComponentFactory.from_dict(comp) for comp in component_set] 
                     for component_set in data['components']]
        return cls(*components)

    def get_config_id(self, component_set_idx: int) -> str:
        """Generate unique identifier for a specific component set configuration.
        
        Parameters
        ----------
        component_set_idx : int
            Index of the component set to generate ID for
            
        Returns
        -------
        str
            Unique identifier for the configuration
        """
        config_dict = self.components[component_set_idx][0].to_dict()
        # Convert to string and hash
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
