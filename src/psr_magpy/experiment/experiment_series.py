"""Experiment series configuration and generation.

This module provides functionality to create multiple experiments from flexible
parameter inputs, including lists for parameter variation.
"""

from typing import List, Optional, Union, Dict, Any, ClassVar
from pydantic import BaseModel
import yaml
from pathlib import Path

from .experiment_config import ExperimentConfig, FieldConfig, GridConfig, TracingConfig, SphericalGridConfig
from .factory_configs import (
    FieldFactoryConfig, GridFactoryConfig, TemplateFactoryConfig, TracingFactoryConfig,
    expand_field_configs, expand_grid_configs, expand_tracing_configs, expand_template_configs
)
from .experiment_metadata import create_experiment_metadata, generate_experiment_name
from psr_magpy.io.config_io import reconstruct_from_yaml
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class ExperimentSeriesConfig(BaseModel):
    """Configuration for experiment series with flexible parameter inputs."""
    
    # Standard subdirectory names
    CONFIGS_DIR: ClassVar[str] = "configs"
    GRIDS_DIR: ClassVar[str] = "grids"
    LINES_DIR: ClassVar[str] = "lines"
    LOGS_DIR: ClassVar[str] = "logs"
    
    name: str
    field: Union[FieldConfig, FieldFactoryConfig]
    grid: Union[GridConfig, GridFactoryConfig]
    tracing: Union[TracingConfig, TracingFactoryConfig]
    template: Union[FieldLine, TemplateFactoryConfig]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = "."  
    
    model_config = {"arbitrary_types_allowed": True}
    
    def create_experiment_series(self) -> List[ExperimentConfig]:
        """Create all possible experiment combinations from the configuration.
        
        Creates output directory structure under self.output_dir:
        {output_dir}/configs, {output_dir}/grids, {output_dir}/lines, {output_dir}/logs
        """
        logger.debug(f"Creating experiment series: {self.name}")
        
        # Create output directory structure
        if self.output_dir:
            self._create_output_directories(self.output_dir)
        
        # Generate all parameter combinations
        field_configs = expand_field_configs(self.field)
        grid_configs = expand_grid_configs(self.grid)
        tracing_configs = expand_tracing_configs(self.tracing)
        template_configs = expand_template_configs(self.template)
        
        # Create all combinations
        experiments = []
        experiment_count = 0
        
        for field_config in field_configs:
            for grid_config in grid_configs:
                for tracing_config in tracing_configs:
                    for template_config in template_configs:
                        experiment_count += 1
                        
                        # Create unique name and metadata
                        exp_name = generate_experiment_name(self.name, experiment_count)
                        exp_metadata = create_experiment_metadata(
                            self.name, self.metadata or {}, experiment_count,
                            field_config, grid_config, tracing_config, template_config
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
    
    def _create_output_directories(self, output_dir: str):
        """Create the standard output directory structure for experiment series."""
        from pathlib import Path
        
        base_dir = Path(output_dir) / self.name
        directories = [self.CONFIGS_DIR, self.GRIDS_DIR, self.LINES_DIR, self.LOGS_DIR]
        
        for dir_name in directories:
            dir_path = base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        
        logger.info(f"Created output directory structure under: {base_dir}")
    
    def get_configs_dir(self) -> Path:
        """Get the full path to the configs directory."""
        from pathlib import Path
        output_dir = self.output_dir or "."
        return Path(output_dir) / self.name / self.CONFIGS_DIR
    
    def get_grids_dir(self) -> Path:
        """Get the full path to the grids directory."""
        from pathlib import Path
        output_dir = self.output_dir or "."
        return Path(output_dir) / self.name / self.GRIDS_DIR
    
    def get_lines_dir(self) -> Path:
        """Get the full path to the lines directory."""
        from pathlib import Path
        output_dir = self.output_dir or "."
        return Path(output_dir) / self.name / self.LINES_DIR
    
    def get_logs_dir(self) -> Path:
        """Get the full path to the logs directory."""
        from pathlib import Path
        output_dir = self.output_dir or "."
        return Path(output_dir) / self.name / self.LOGS_DIR
    
    def save(self, filename: Optional[str] = None) -> None:
        """Save configuration to YAML file in the configs directory.
        
        Args:
            filename: Optional filename. If not provided, uses "{series_name}_config.yaml"
        """
        from psr_magpy.io import IOFactory
        
        if filename is None:
            filename = f"{self.name}_config.yaml"
        
        filepath = self.get_configs_dir() / filename
        
        # Create I/O instance without overwrite parameter
        io_instance = IOFactory.create_config_io(type(self))
        # Save with overwrite=True
        io_instance.save(self, filepath, overwrite=True)
    
    @classmethod
    def load(cls, output_dir: str, experiment_name: str, filename: Optional[str] = None) -> "ExperimentSeriesConfig":
        """Load configuration from YAML file in the configs directory.
        
        Args:
            output_dir: Base output directory
            experiment_name: Name of the experiment series
            filename: Optional filename. If not provided, uses "{experiment_name}_config.yaml"
        """
        from psr_magpy.io import IOFactory
        
        if filename is None:
            filename = f"{experiment_name}_config.yaml"
        
        # Create a temporary instance to get the configs directory path
        temp_instance = cls(
            name=experiment_name,
            field=FieldConfig(components=[]),  # Placeholder
            grid=GridConfig(spherical=SphericalGridConfig(nlon=1, nlat=1)),  # Placeholder
            tracing=TracingConfig(),  # Placeholder
            template=FieldLine(),  # Placeholder
            output_dir=output_dir
        )
        
        filepath = temp_instance.get_configs_dir() / filename
        return IOFactory.load_config(cls, filepath)
    
    def print_yaml(self):
        """Print configuration as pretty YAML to screen."""
        from psr_magpy.io import IOFactory
        io_instance = IOFactory.create_config_io(type(self))
        io_instance.print_config(self)
    
    def print_experiment_summary(self, show_changing_only: bool = True, max_experiments: int = 20):
        """Print a summary of experiments with changing parameters highlighted.
        
        Args:
            show_changing_only: If True, only show parameters that vary between experiments
            max_experiments: Maximum number of experiments to display (to avoid overwhelming output)
        """
        experiments = self.create_experiment_series()
        
        if len(experiments) == 0:
            print("No experiments created.")
            return
        
        print(f"\n=== Experiment Series Summary: {self.name} ===")
        print(f"Total experiments: {len(experiments)}")
        print(f"Description: {self.description or 'No description'}")
        
        if len(experiments) > max_experiments:
            print(f"\nShowing first {max_experiments} experiments (use max_experiments parameter to see more):")
            experiments = experiments[:max_experiments]
        
        # Find changing parameters
        changing_params = self._identify_changing_parameters(experiments)
        
        print(f"\nChanging parameters: {', '.join(changing_params) if changing_params else 'None'}")
        print("\nExperiments:")
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n  Experiment {i}: {exp.name}")
            
            if show_changing_only and changing_params:
                # Show only changing parameters
                for param in changing_params:
                    value = self._get_parameter_value(exp, param)
                    if value is not None:
                        print(f"    {param}: {value}")
            else:
                # Show all key parameters
                self._print_experiment_parameters(exp)
    
    def _identify_changing_parameters(self, experiments: List[ExperimentConfig]) -> List[str]:
        """Identify which parameters change between experiments."""
        if len(experiments) <= 1:
            return []
        
        # Get all parameter values for each experiment
        all_params = {}
        for exp in experiments:
            params = self._extract_all_parameters(exp)
            for key, value in params.items():
                if key not in all_params:
                    all_params[key] = set()
                all_params[key].add(str(value))
        
        # Find parameters that have more than one unique value
        changing_params = []
        for param, values in all_params.items():
            if len(values) > 1:
                changing_params.append(param)
        
        return sorted(changing_params)
    
    def _extract_all_parameters(self, exp: ExperimentConfig) -> Dict[str, Any]:
        """Extract all relevant parameters from an experiment."""
        params = {}
        
        # Field parameters - use moment_type-based identification only
        for i, comp in enumerate(exp.field.components):
            # Use moment_type to identify components more reliably
            comp_id = f"field.{comp.moment_type}_{i+1}"
            params[f"{comp_id}.moment_type"] = comp.moment_type
            params[f"{comp_id}.B_magnitude"] = comp.B_magnitude
            params[f"{comp_id}.lat"] = comp.moment_orientation_latitude_degree
            params[f"{comp_id}.lon"] = comp.moment_orientation_longitude_degree
        
        # Grid parameters
        if exp.grid.spherical:
            params["grid.spherical.nlon"] = exp.grid.spherical.nlon
            params["grid.spherical.nlat"] = exp.grid.spherical.nlat
            params["grid.spherical.pole_lat"] = exp.grid.spherical.pole_lat
        elif exp.grid.random:
            params["grid.random.nline_ori"] = exp.grid.random.nline_ori
        elif exp.grid.healpix:
            params["grid.healpix.nside"] = exp.grid.healpix.nside
        elif exp.grid.custom:
            params["grid.custom.nline_ori"] = exp.grid.custom.nline_ori
        
        # Tracing parameters
        params["tracing.max_integration_radius"] = exp.tracing.max_integration_radius
        params["tracing.threshold_line_distance"] = exp.tracing.threshold_line_distance
        params["tracing.surface_radius"] = exp.tracing.surface_radius
        params["tracing.max_memory_mb"] = exp.tracing.max_memory_mb
        
        return params
    
    def _get_parameter_value(self, exp: ExperimentConfig, param: str) -> Any:
        """Get the value of a specific parameter from an experiment."""
        params = self._extract_all_parameters(exp)
        return params.get(param)
    
    def _print_experiment_parameters(self, exp: ExperimentConfig):
        """Print all key parameters for an experiment."""
        # Field components
        print(f"    Field components: {len(exp.field.components)}")
        for i, comp in enumerate(exp.field.components):
            print(f"      Component {i+1}: {comp.moment_type}")
            print(f"        B_magnitude: {comp.B_magnitude}")
            print(f"        lat: {comp.moment_orientation_latitude_degree}°")
            print(f"        lon: {comp.moment_orientation_longitude_degree}°")
        
        # Grid
        if exp.grid.spherical:
            print(f"    Grid: Spherical (nlon={exp.grid.spherical.nlon}, nlat={exp.grid.spherical.nlat})")
        elif exp.grid.random:
            print(f"    Grid: Random (nline_ori={exp.grid.random.nline_ori})")
        elif exp.grid.healpix:
            print(f"    Grid: HEALPix (nside={exp.grid.healpix.nside})")
        elif exp.grid.custom:
            print(f"    Grid: Custom (nline_ori={exp.grid.custom.nline_ori})")
        
        # Tracing
        print(f"    Tracing: max_radius={exp.tracing.max_integration_radius}, memory={exp.tracing.max_memory_mb}MB")