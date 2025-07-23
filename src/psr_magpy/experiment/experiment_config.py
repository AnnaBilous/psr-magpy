import yaml
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Any, Literal, TypedDict, Tuple, TYPE_CHECKING


from pathlib import Path
import numpy as np

from psr_magpy.field.components import DipoleFieldComponent, QuadrupoleFieldComponent, MultipoleType
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.grid.grid import SphericalGrid, RandomGrid, HealpixGrid, CustomGrid, GridType
from psr_magpy.line.tracing_parameters import TracingParameters, IntegrationParameters, OutputParameters, PerformanceParameters
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


# Type definitions for better type safety


from pydantic import BaseModel, Field


class FieldComponentConfig(BaseModel):
    """Configuration for a single field component."""
    
    moment_type: str
    B_magnitude: float = 1.0
    moment_orientation_latitude_degree: float = 0.0
    moment_orientation_longitude_degree: float = 0.0
    offset_r_ns: List[float] = [0.0, 0.0, 0.0]


class FieldConfig(BaseModel):
    """Configuration for magnetic field components."""
    
    components: List[FieldComponentConfig] = Field(description="List of field components")
    
    def create_field(self) -> MultipoleField:
        """Create MultipoleField from configuration."""
        logger.debug(f"Creating field with {len(self.components)} components")
        
        field_components = []
        for i, comp in enumerate(self.components):
            comp_type = comp.moment_type
            logger.debug(f"Creating {comp_type} component {i+1}")
            
            if comp_type == MultipoleType.DIPOLE.value:
                component = DipoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns
                )
            elif comp_type == MultipoleType.QUADRUPOLE.value:
                component = QuadrupoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns
                )
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
            comp_type = comp.moment_type
            
            # Create component to get all parameters (including defaults)
            if comp_type == MultipoleType.DIPOLE.value:
                component = DipoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns
                )
            elif comp_type == MultipoleType.QUADRUPOLE.value:
                component = QuadrupoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns
                )
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


class SphericalGridConfig(BaseModel):
    """Configuration for spherical grid."""
    
    nlon: int
    nlat: int
    pole_lat: float = 0.0


class RandomGridConfig(BaseModel):
    """Configuration for random grid."""
    
    nline_ori: int


class HealpixGridConfig(BaseModel):
    """Configuration for HEALPix grid."""
    
    nside: int


class CustomGridConfig(BaseModel):
    """Configuration for custom grid."""
    
    nline_ori: int
    coordinates: Optional[List[List[float]]] = None


class GridConfig(BaseModel):
    """Configuration for field line grid."""
    
    spherical: Optional[SphericalGridConfig] = Field(default=None, description="Spherical grid parameters")
    random: Optional[RandomGridConfig] = Field(default=None, description="Random grid parameters")
    healpix: Optional[HealpixGridConfig] = Field(default=None, description="HEALPix grid parameters")
    custom: Optional[CustomGridConfig] = Field(default=None, description="Custom grid parameters")
    
    def model_post_init(self, __context) -> None:
        """Validate that exactly one grid type is specified."""
        configs = [self.spherical, self.random, self.healpix, self.custom]
        non_none_configs = [c for c in configs if c is not None]
        
        if len(non_none_configs) != 1:
            error_msg = f"Exactly one grid type must be specified, got {len(non_none_configs)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def get_grid_type(self) -> str:
        """Get the grid type from the configuration."""
        if self.spherical is not None:
            return GridType.SPHERICAL.value
        elif self.random is not None:
            return GridType.RANDOM.value
        elif self.healpix is not None:
            return GridType.HEALPIX.value
        elif self.custom is not None:
            return GridType.CUSTOM.value
        else:
            error_msg = "No grid type specified"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def create_grid(self):
        """Create grid from configuration."""
        grid_type = self.get_grid_type()
        logger.debug(f"Creating {grid_type} grid")
        
        if self.spherical is not None:
            return SphericalGrid(**self.spherical.model_dump())
        elif self.random is not None:
            return RandomGrid(**self.random.model_dump())
        elif self.healpix is not None:
            return HealpixGrid(**self.healpix.model_dump())
        elif self.custom is not None:
            return CustomGrid(**self.custom.model_dump())
        else:
            error_msg = "No grid type specified"
            logger.error(error_msg)
            raise ValueError(error_msg)


class TracingConfig(BaseModel):
    """Configuration for field line tracing parameters."""
    
    # Integration parameters
    max_integration_radius: float = 10.0
    threshold_line_distance: float = 1.0
    t_span: Tuple[float, float] = (0.0, 1e8)
    rtol: float = 5.0e-9
    atol: float = 5.0e-9
    surface_radius: float = 1.0 - 1.e-8
    dt: float = 0.01
    max_step: float = 0.01
    trace_open_lines_backwards: bool = False
    
    # Performance parameters
    max_memory_mb: float = 100.0
    safety_factor: float = 0.9
    max_batch_size: int = 100
    check_interval_sec: float = 1.0
    n_cores: Optional[int] = None
    n_trial: int = 10
    progress_bar: bool = True
    verbose: bool = False
    skip_memory_estimation: bool = False
    
    # Output parameters
    save_results: bool = False
    filename: Optional[str] = None
    output_format: str = "fits"
    overwrite: bool = False
    
    def create_tracing_parameters(self) -> TracingParameters:
        """Create TracingParameters from configuration."""
        logger.debug("Creating tracing parameters from configuration")
        
        integration = IntegrationParameters(
            max_integration_radius=self.max_integration_radius,
            threshold_line_distance=self.threshold_line_distance,
            t_span=self.t_span,
            rtol=self.rtol,
            atol=self.atol,
            surface_radius=self.surface_radius,
            dt=self.dt,
            max_step=self.max_step,
            trace_open_lines_backwards=self.trace_open_lines_backwards
        )
        
        performance = PerformanceParameters(
            max_memory_mb=self.max_memory_mb,
            safety_factor=self.safety_factor,
            max_batch_size=self.max_batch_size,
            check_interval_sec=self.check_interval_sec,
            n_cores=self.n_cores,
            n_trial=self.n_trial,
            progress_bar=self.progress_bar,
            verbose=self.verbose,
            skip_memory_estimation=self.skip_memory_estimation
        )
        
        output = OutputParameters(
            save_results=self.save_results,
            filename=self.filename,
            output_format=self.output_format,
            overwrite=self.overwrite
        )
        
        return TracingParameters(
            integration=integration,
            performance=performance,
            output=output
        )
    
    def get_complete_config(self) -> Dict[str, Any]:
        """Get complete tracing configuration including all default parameters."""
        logger.debug("Getting complete tracing configuration")
        
        # Return flat configuration for backward compatibility
        return self.model_dump()


class FieldFactoryConfig(BaseModel):
    """Factory configuration for field components with flexible parameter inputs."""
    
    components: List[Union[FieldComponentConfig, List[FieldComponentConfig]]]


class GridFactoryConfig(BaseModel):
    """Factory configuration for grids with flexible parameter inputs."""
    
    grids: List[Union[SphericalGridConfig, RandomGridConfig, HealpixGridConfig, CustomGridConfig]]


class TemplateFactoryConfig(BaseModel):
    """Factory configuration for templates with flexible parameter inputs."""
    
    templates: List[FieldLine]
    
    model_config = {"arbitrary_types_allowed": True}


class ExperimentSeriesConfig(BaseModel):
    """Configuration for experiment series with flexible parameter inputs."""
    
    name: str
    field: Union[FieldConfig, FieldFactoryConfig]
    grid: Union[GridConfig, GridFactoryConfig]
    tracing: TracingConfig
    template: Union[FieldLine, TemplateFactoryConfig]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = {"arbitrary_types_allowed": True}
    
    def create_experiment_series(self) -> List["ExperimentConfig"]:
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
            else:
                # List of components
                field_configs.append(FieldConfig(components=component_group))
        
        return field_configs
    
    def _expand_grid_configs(self) -> List[GridConfig]:
        """Expand grid configuration into all possible combinations."""
        if isinstance(self.grid, GridConfig):
            return [self.grid]
        
        # Handle GridFactoryConfig
        grid_configs = []
        for grid in self.grid.grids:
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
        # For now, just return the single tracing config
        # Parameter expansion is handled in experiment_series.py
        return [self.tracing]
    
    def _expand_template_configs(self) -> List[FieldLine]:
        """Expand template configuration into all possible combinations."""
        if isinstance(self.template, FieldLine):
            return [self.template]
        
        # Handle TemplateFactoryConfig
        return self.template.templates
    

    
    def _create_experiment_metadata(self, experiment_count: int, field_config: FieldConfig, 
                                  grid_config: GridConfig, tracing_config: TracingConfig, 
                                  template_config: FieldLine) -> Dict[str, Any]:
        """Create unique metadata for each experiment."""
        metadata = self.metadata.copy() if self.metadata else {}
        
        # Add experiment-specific metadata
        metadata.update({
            "experiment_number": experiment_count,
            "series_name": self.name,
        })
        
        return metadata


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    name: str
    field: FieldConfig
    grid: GridConfig
    tracing: TracingConfig
    template: FieldLine
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        logger.debug(f"Initializing experiment: {self.name}")
        if self.description:
            logger.debug(f"Description: {self.description}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        # Ensure I/O module is imported for registration
        import psr_magpy.io
        from psr_magpy.io import IOFactory, ExperimentConfigIO
        
        # Ensure registration
        if type(self) not in IOFactory._config_registry:
            IOFactory.register_config_io(type(self), ExperimentConfigIO)
        
        IOFactory.save_config(self, filepath)
    
    def print_yaml(self):
        """Print configuration as pretty YAML to screen."""
        # Ensure I/O module is imported for registration
        import psr_magpy.io
        from psr_magpy.io import IOFactory, ExperimentConfigIO
        
        # Ensure registration
        if type(self) not in IOFactory._config_registry:
            IOFactory.register_config_io(type(self), ExperimentConfigIO)
        
        io_instance = IOFactory.create_config_io(type(self))
        io_instance.print_config(self)
    

    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        # Ensure I/O module is imported for registration
        import psr_magpy.io
        from psr_magpy.io import IOFactory, ExperimentConfigIO
        
        # Ensure registration
        if cls not in IOFactory._config_registry:
            IOFactory.register_config_io(cls, ExperimentConfigIO)
        
        return IOFactory.load_config(cls, filepath)
    
    @classmethod
    def create_dipole_example(cls) -> "ExperimentConfig":
        """Create a simple dipole field example."""
        field = FieldConfig(components=[FieldComponentConfig(
            moment_type="dipole",
            B_magnitude=1.0,
            moment_orientation_latitude_degree=0.0,
            moment_orientation_longitude_degree=0.0
        )])
        
        grid = GridConfig(
            spherical=SphericalGridConfig(nlon=2, nlat=5, pole_lat=0.0)
        )
        
        tracing = TracingConfig(
            max_integration_radius=10.0,
            threshold_line_distance=1.0,
            t_span=(0.0, 1e8),
            rtol=5.0e-9,
            atol=5.0e-9,
            surface_radius=1.0 - 1.e-8,
            dt=5,
            max_step=0.01,
            trace_open_lines_backwards=True,
            max_memory_mb=100.0,
            safety_factor=0.9,
            max_batch_size=100,
            check_interval_sec=1.0,
            n_cores=2,
            n_trial=100,
            progress_bar=True,
            verbose=False,
            skip_memory_estimation=True,
            save_results=False,
            filename=None,
            output_format="fits",
            overwrite=False
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