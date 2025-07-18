import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Any, Literal, TypedDict
from pathlib import Path
import numpy as np

from psr_magpy.field.components import DipoleFieldComponent, QuadrupoleFieldComponent, MultipoleType
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.grid.grid import SphericalGrid, RandomGrid, HealpixGrid, GridType
from psr_magpy.line.tracing_parameters import TracingParameters, IntegrationParameters, OutputParameters, PerformanceParameters
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.json_utils import NumpyEncoder

logger = get_logger(__name__, console_format_brief=True)


# Type definitions for better type safety


def validate_field_component(comp: Dict[str, Any]) -> Dict[str, Any]:
    """Validate field component configuration."""
    required_fields = ["moment_type", "B_magnitude", "moment_orientation_latitude_degree", "moment_orientation_longitude_degree"]
    
    for field in required_fields:
        if field not in comp:
            error_msg = f"Missing required field: {field}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    valid_types = [mt.value for mt in MultipoleType]
    if comp["moment_type"] not in valid_types:
        error_msg = f"Invalid moment_type: {comp['moment_type']}. Must be one of {valid_types}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate numeric fields
    for field in ["B_magnitude", "moment_orientation_latitude_degree", "moment_orientation_longitude_degree"]:
        if not isinstance(comp[field], (int, float)):
            error_msg = f"Field {field} must be numeric, got {type(comp[field])}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    return comp


@dataclass
class FieldConfig:
    """Configuration for magnetic field components."""
    
    components: List[Dict[str, Any]]
    
    def __post_init__(self):
        """Validate all field components."""
        logger.debug(f"Validating {len(self.components)} field components")
        for i, comp in enumerate(self.components):
            try:
                validate_field_component(comp)
                logger.debug(f"Component {i+1} validated successfully")
            except ValueError as e:
                error_msg = f"Validation failed for component {i+1}: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
    
    def create_field(self) -> MultipoleField:
        """Create MultipoleField from configuration."""
        logger.debug(f"Creating field with {len(self.components)} components")
        
        field_components = []
        for i, comp in enumerate(self.components):
            comp_type = comp["moment_type"]
            logger.debug(f"Creating {comp_type} component {i+1}")
            
            if comp_type == MultipoleType.DIPOLE.value:
                component = DipoleFieldComponent(**{k: v for k, v in comp.items() if k != "moment_type"})
            elif comp_type == MultipoleType.QUADRUPOLE.value:
                component = QuadrupoleFieldComponent(**{k: v for k, v in comp.items() if k != "moment_type"})
            else:
                error_msg = f"Unknown field component type: {comp_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            field_components.append(component)
        
        return MultipoleField(field_components)
    
    def get_complete_components(self) -> List[Dict[str, Any]]:
        """Get complete component configurations including all default parameters."""
        logger.debug("Getting complete component configurations")
        
        complete_components = []
        for comp in self.components:
            comp_type = comp["moment_type"]
            
            # Create component to get all parameters (including defaults)
            if comp_type == MultipoleType.DIPOLE.value:
                component = DipoleFieldComponent(**{k: v for k, v in comp.items() if k != "moment_type"})
            elif comp_type == MultipoleType.QUADRUPOLE.value:
                component = QuadrupoleFieldComponent(**{k: v for k, v in comp.items() if k != "moment_type"})
            else:
                error_msg = f"Unknown field component type: {comp_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Get complete parameters including defaults
            complete_params = component.params.model_dump()
            # Convert enum to string for JSON serialization
            complete_params["moment_type"] = complete_params["moment_type"].value
            
            complete_components.append(complete_params)
            logger.debug(f"Complete params for {comp_type}: {complete_params}")
        
        return complete_components


@dataclass
class GridConfig:
    """Configuration for field line grid."""
    
    type: str  # "spherical", "random", "healpix"
    params: Dict[str, Any]
    
    def __post_init__(self):
        """Validate grid configuration."""
        logger.debug(f"Validating grid configuration: {self.type}")
        
        valid_types = [gt.value for gt in GridType]
        if self.type not in valid_types:
            error_msg = f"Invalid grid type: {self.type}. Must be one of {valid_types}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate required parameters for each grid type
        if self.type == GridType.SPHERICAL.value:
            required = ["nlon", "nlat"]
            for param in required:
                if param not in self.params:
                    error_msg = f"Missing required parameter for spherical grid: {param}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        elif self.type == GridType.RANDOM.value:
            if "nline_ori" not in self.params:
                error_msg = "Missing required parameter for random grid: nline_ori"
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif self.type == GridType.HEALPIX.value:
            if "nside" not in self.params:
                error_msg = "Missing required parameter for healpix grid: nside"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        logger.debug(f"Grid configuration validated successfully")
    
    def create_grid(self):
        """Create grid from configuration."""
        logger.debug(f"Creating {self.type} grid")
        
        if self.type == GridType.SPHERICAL.value:
            return SphericalGrid(**self.params)
        elif self.type == GridType.RANDOM.value:
            return RandomGrid(**self.params)
        elif self.type == GridType.HEALPIX.value:
            return HealpixGrid(**self.params)
        else:
            error_msg = f"Unknown grid type: {self.type}"
            logger.error(error_msg)
            raise ValueError(error_msg)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    name: str
    field: FieldConfig
    grid: GridConfig
    tracing: TracingParameters
    template: FieldLine
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        logger.debug(f"Initializing experiment: {self.name}")
        if self.description:
            logger.debug(f"Description: {self.description}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        logger.info(f"Saving experiment config to {filepath}")
        
        # Convert to dict, handling nested objects
        config_dict = {
            "name": self.name,
            "description": self.description,
            "field": {
                "components": self.field.get_complete_components()
            },
            "grid": asdict(self.grid),
            "tracing": {
                "integration": self.tracing.integration.model_dump(),
                "performance": self.tracing.performance.model_dump(),
                "output": self.tracing.output.model_dump()
            },
            "template": self.template.property_system.get_flags_dict(),
            "metadata": self.metadata or {}
        }
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with numpy support
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        logger.info(f"Loading experiment config from {filepath}")
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Reconstruct TracingParameters
        tracing_data = data["tracing"]
        tracing = TracingParameters(
            integration=IntegrationParameters.model_validate(tracing_data["integration"]),
            performance=PerformanceParameters.model_validate(tracing_data["performance"]),
            output=OutputParameters.model_validate(tracing_data["output"])
        )
        
        # Reconstruct template
        template_data = data.get("template", {})
        template = FieldLine(**template_data)
        
        # Create config objects
        config = cls(
            name=data["name"],
            description=data.get("description"),
            field=FieldConfig(**data["field"]),
            grid=GridConfig(**data["grid"]),
            tracing=tracing,
            template=template,
            metadata=data.get("metadata")
        )
        
        logger.info(f"Loaded experiment: {config.name}")
        return config
    
    @classmethod
    def create_dipole_example(cls) -> "ExperimentConfig":
        """Create a simple dipole field example."""
        field = FieldConfig(components=[{
            "moment_type": "dipole",
            "B_magnitude": 1.0,
            "moment_orientation_latitude_degree": 80.0,
            "moment_orientation_longitude_degree": 0.0
        }])
        
        grid = GridConfig(
            type="spherical",
            params={"nlon": 2, "nlat": 5, "pole_lat": 0.0}
        )
        
        tracing = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                dt=5,
                trace_open_lines_backwards=True
            ),
            performance=PerformanceParameters(
                max_memory_mb=100.0,
                n_cores=2,
                skip_memory_estimation=True
            ),
            output=OutputParameters(
                save_results=False,
                filename=None
            )
        )
        
        template = FieldLine.template_all()
        
        return cls(
            name="dipole_example",
            description="Simple dipole field experiment",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template
        ) 