"""Experiment series configuration and generation.

This module provides functionality to create multiple experiments from flexible
parameter inputs, including lists for parameter variation.
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel
import yaml
from pathlib import Path

from .experiment_config import ExperimentConfig, FieldConfig, GridConfig, TracingConfig
from .factory_configs import (
    FieldFactoryConfig, GridFactoryConfig, TemplateFactoryConfig, TracingFactoryConfig,
    expand_field_configs, expand_grid_configs, expand_tracing_configs, expand_template_configs
)
from .experiment_metadata import create_experiment_metadata, generate_experiment_name
from .yaml_handling import reconstruct_from_yaml
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class ExperimentSeriesConfig(BaseModel):
    """Configuration for experiment series with flexible parameter inputs."""
    
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
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        logger.info(f"Saving experiment series config to {filepath}")
        
        # Convert to dict
        config_dict = self.model_dump()
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with YAML
        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentSeriesConfig":
        """Load configuration from YAML file."""
        filepath = Path(filepath)
        logger.info(f"Loading experiment series config from {filepath}")
        
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        
        # Reconstruct nested objects
        data = reconstruct_from_yaml(data)
        
        # Create config object
        config = cls(**data)
        
        logger.info(f"Loaded experiment series: {config.name}")
        return config
    
    def print_yaml(self):
        """Print configuration as pretty YAML to screen."""
        config_dict = self.model_dump()
        yaml_string = yaml.dump(config_dict, default_flow_style=False, indent=2, sort_keys=False)
        print("=== Experiment Series Configuration ===")
        print(yaml_string)
        print("=======================================")