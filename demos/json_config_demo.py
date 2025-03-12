#!/usr/bin/env python
"""
Demo script for JSON-based experiment configuration.

This script demonstrates how to use the ExperimentSetup class to generate
experiment configurations from a JSON specification.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from utils import setup_demo

# Setup directories and logging
fits_dir, plot_dir, log_dir, logger = setup_demo(__file__)

from src.experiment.config import ExperimentSetup, ExperimentConfig, FieldConfig

def create_example_json():
    """Create an example JSON configuration file."""
    config = {
        "field_config": {
            "moment_type": {"1": "dipole", "2": "quadrupole"},
            "B_magnitude": {"1": 2.0, "2": 3.0},
            "lat_degree": [0.0, 30.0, 90.0],
            "lon_degree": 0.0,
            "offset_r_ns": [0.0, 0.0, 0.0]
        },
        "output_dir": str(fits_dir),
        "save_vars": ["footpoints", "field_lines", "open_fraction"],
        "experiment_name": "multipole_study",
        "filename_template": "{moment_type}_{B_magnitude}_lat{lat_degree}.npz"
    }
    
    json_path = fits_dir / "example_config.json"
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created example JSON configuration at {json_path}")
    return json_path

def create_programmatic_config():
    """Create a configuration programmatically."""
    # Create a field configuration with parameter sets and ranges
    field_config = FieldConfig(
        moment_type={"A": "dipole", "B": "quadrupole"},
        B_magnitude={"A": 1.0, "B": 2.0},
        lat_degree=[0.0, 45.0, 90.0],
        lon_degree=0.0
    )
    
    dip_comp = FieldComponentConfig(
        moment_type="dipole",
        B_magnitude=1.,
        lat_degree=[0.0, 45.0, 90.0],
        lon_degree=0.0
    )
    quad_comp = FieldComponentConfig(
        moment_type="quadrupole",
        B_magnitude=1.,
        lat_degree=[0.0, 45.0, 90.0],
        lon_degree=0.0
    )
    
    field_config = FieldConfig([dip_comp, quad_comp])
    
    field_config = FieldConfig(
        moment_type={"A": "dipole", "B": "quadrupole"},
        B_magnitude={"A": 1.0, "B": 2.0},
        lat_degree=[0.0, 45.0, 90.0],
        lon_degree=0.0
    )
    
    
    
    # Create the experiment configuration
    experiment_config = ExperimentConfig(
        field_config=field_config,
        output_dir=fits_dir,
        save_vars=["footpoints", "field_lines"],
        experiment_name="programmatic_example",
        filename_template="prog_{moment_type}_{B_magnitude}_lat{lat_degree}.npz"
    )
    
    return experiment_config

def visualize_configs(configs, title):
    """Visualize the parameter space of configurations."""
    # Extract parameters
    moment_types = []
    field_strengths = []
    latitudes = []
    
    for config, filename in configs:
        moment_types.append(config.field_config.moment_type)
        field_strengths.append(config.field_config.B_magnitude)
        latitudes.append(config.field_config.lat_degree)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a scatter plot
    for i, moment_type in enumerate(set(moment_types)):
        indices = [j for j, mt in enumerate(moment_types) if mt == moment_type]
        ax.scatter(
            [latitudes[j] for j in indices],
            [field_strengths[j] for j in indices],
            label=moment_type,
            s=100,
            alpha=0.7
        )
    
    ax.set_xlabel('Latitude (degrees)')
    ax.set_ylabel('Field Strength (B)')
    ax.set_title(f'Parameter Space: {title}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig(plot_dir / f"{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {plot_dir / f'{title.lower().replace(' ', '_')}.png'}")
    plt.close()

def main():
    """Run the demo."""
    logger.info("Starting JSON-based experiment configuration demo...")
    
    # Example 1: Load from JSON
    logger.info("\n=== Example 1: Configuration from JSON ===")
    json_path = create_example_json()
    experiment_setup = ExperimentSetup.from_json(json_path)
    json_configs = experiment_setup.generate_configs()
    
    logger.info(f"Generated {len(json_configs)} configurations from JSON")
    for i, (config, filename) in enumerate(json_configs):
        logger.info(f"Configuration {i+1}:")
        logger.info(f"  Filename: {filename}")
        logger.info(f"  Moment type: {config.field_config.moment_type}")
        logger.info(f"  Field strength: {config.field_config.B_magnitude}")
        logger.info(f"  Latitude: {config.field_config.lat_degree} degrees")
    
    # Example 2: Create programmatically
    logger.info("\n=== Example 2: Programmatic Configuration ===")
    experiment_config = create_programmatic_config()
    experiment_setup = ExperimentSetup(experiment_config)
    prog_configs = experiment_setup.generate_configs()
    
    logger.info(f"Generated {len(prog_configs)} configurations programmatically")
    for i, (config, filename) in enumerate(prog_configs):
        logger.info(f"Configuration {i+1}:")
        logger.info(f"  Filename: {filename}")
        logger.info(f"  Moment type: {config.field_config.moment_type}")
        logger.info(f"  Field strength: {config.field_config.B_magnitude}")
        logger.info(f"  Latitude: {config.field_config.lat_degree} degrees")
    
    # Example 3: Modify an existing configuration
    logger.info("\n=== Example 3: Modified Configuration ===")
    # Start with the first configuration from Example 2
    base_config = prog_configs[0][0]
    
    # Create a new field configuration with different parameters
    modified_field = FieldConfig(
        moment_type="octupole",
        B_magnitude=4.0,
        lat_degree=[15.0, 60.0],  # Parameter range
        lon_degree={"X": 0.0, "Y": 45.0}  # Parameter set
    )
    
    # Create a new experiment configuration
    modified_experiment = ExperimentConfig(
        field_config=modified_field,
        output_dir=base_config.output_dir,
        save_vars=base_config.save_vars + ["additional_data"],
        experiment_name="modified_example",
        filename_template="mod_{moment_type}_B{B_magnitude}_lat{lat_degree}_lon{lon_degree}.npz"
    )
    
    # Generate configurations
    modified_setup = ExperimentSetup(modified_experiment)
    mod_configs = modified_setup.generate_configs()
    
    logger.info(f"Generated {len(mod_configs)} configurations from modified config")
    for i, (config, filename) in enumerate(mod_configs):
        logger.info(f"Configuration {i+1}:")
        logger.info(f"  Filename: {filename}")
        logger.info(f"  Moment type: {config.field_config.moment_type}")
        logger.info(f"  Field strength: {config.field_config.B_magnitude}")
        logger.info(f"  Latitude: {config.field_config.lat_degree} degrees")
        logger.info(f"  Longitude: {config.field_config.lon_degree} degrees")
        logger.info(f"  Save variables: {config.save_vars}")
    
    # Visualize the parameter spaces
    visualize_configs(json_configs, "JSON Configuration")
    visualize_configs(prog_configs, "Programmatic Configuration")
    visualize_configs(mod_configs, "Modified Configuration")
    
    # Example 4: Practical use case - Dipole tilt angle study
    logger.info("\n=== Example 4: Practical Use Case - Dipole Tilt Study ===")
    
    # Create a configuration for studying dipole tilt angles
    tilt_field = FieldConfig(
        moment_type="dipole",
        B_magnitude=1.0,
        lat_degree=np.linspace(0, 90, 10).tolist(),  # 10 tilt angles from 0 to 90 degrees
        lon_degree=0.0
    )
    
    tilt_experiment = ExperimentConfig(
        field_config=tilt_field,
        output_dir=fits_dir / "tilt_study",
        save_vars=["footpoints", "field_lines", "open_fraction"],
        experiment_name="dipole_tilt_study",
        filename_template="dipole_tilt_{lat_degree:.1f}.npz"
    )
    
    # Generate configurations
    tilt_setup = ExperimentSetup(tilt_experiment)
    tilt_configs = tilt_setup.generate_configs()
    
    logger.info(f"Generated {len(tilt_configs)} configurations for tilt study")
    logger.info("Tilt angles:")
    for i, (config, filename) in enumerate(tilt_configs):
        if i < 3 or i >= len(tilt_configs) - 3:
            logger.info(f"  {config.field_config.lat_degree:.1f} degrees -> {filename}")
        elif i == 3:
            logger.info("  ...")
    
    logger.info("\nDemo completed successfully!")
    
    # Return all configurations for potential further use
    return {
        "json": json_configs,
        "programmatic": prog_configs,
        "modified": mod_configs,
        "tilt_study": tilt_configs
    }

if __name__ == "__main__":
    configs = main() 