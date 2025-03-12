#!/usr/bin/env python
"""
Demo script for ExperimentSetup class from config.py.

This script demonstrates how to use the ExperimentSetup class to generate
experiment configurations with various parameter combinations.
"""

import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from utils import setup_demo
fits_dir, plot_dir, log_dir, logger = setup_demo(__file__)


from src.experiment.config import (
    ExperimentConfig, 
    FieldConfig, 
    GridConfig, 
    TracerConfig, 
    ExperimentSetup
)



def create_base_config():
    """Create a base configuration for the experiments."""
    # Create a dipole field configuration
    field_config = FieldConfig.create_dipole(
        B=1.0,  # Field strength
        lat=90.0,  # Aligned with rotation axis
        lon=0.0  # Longitude
    )
    
    # Create a grid configuration using HEALPix
    grid_config = GridConfig(
        grid_type='healpix',
        nside=8  # Resolution parameter for HEALPix
    )
    
    # Create a tracer configuration
    tracer_config = TracerConfig(
        R_max=10.0,  # Maximum radius for tracing
        R_min=1.0,   # Minimum radius (neutron star surface)
        dt=0.01,     # Time step for integration
        n_cores=4    # Number of cores for parallel processing
    )
    

    
    experiment_config = ExperimentConfig(
        field_config=field_config,
        grid_config=grid_config,
        tracer_config=tracer_config,
        output_dir=fits_dir,
        save_vars=["footpoints", "field_lines", "open_fraction"],
        experiment_name="dipole_demo"
    )
    
    return experiment_config

def demo_parameter_ranges():
    """Demonstrate using parameter ranges to generate configurations."""
    logger.info("Demonstrating parameter ranges...")
    
    base_config = create_base_config()
    
    # Define parameter ranges for a grid search
    parameter_ranges = {
        'field_config.components[0].lat_degree': [0.0, 45.0, 90.0],  # Dipole inclination
        'field_config.components[0].B_magnitude': [0.5, 1.0, 2.0],   # Field strength
        'tracer_config.R_max': [5.0, 10.0]                           # Maximum tracing radius
    }
    
    # Create the experiment setup with parameter ranges
    experiment_setup = ExperimentSetup(
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        filename_template="dipole_{field_config.components[0].lat_degree}deg_{field_config.components[0].B_magnitude}B_Rmax{tracer_config.R_max}.npz"
    )
    
    # Generate configurations
    configs = experiment_setup.generate_configs()
    
    logger.info(f"Generated {len(configs)} configurations from parameter ranges")
    
    # Print details of the first few configurations
    for i, (config, filename) in enumerate(configs[:3]):
        logger.info(f"Configuration {i+1}:")
        logger.info(f"  Filename: {filename}")
        logger.info(f"  Dipole latitude: {config.field_config.components[0].lat_degree} degrees")
        logger.info(f"  Field strength: {config.field_config.components[0].B_magnitude}")
        logger.info(f"  Maximum radius: {config.tracer_config.R_max}")
    
    if len(configs) > 3:
        logger.info(f"... and {len(configs) - 3} more configurations")
    
    return configs

def demo_parameter_sets():
    """Demonstrate using explicit parameter sets to generate configurations."""
    logger.info("\nDemonstrating explicit parameter sets...")
    
    base_config = create_base_config()
    
    # Define explicit parameter sets for specific configurations
    parameter_sets = [
        {
            'field_config.components[0].lat_degree': 30.0,
            'field_config.components[0].B_magnitude': 1.5,
            'grid_config.nside': 16
        },
        {
            'field_config.components[0].lat_degree': 60.0,
            'field_config.components[0].B_magnitude': 2.0,
            'tracer_config.R_max': 15.0
        },
        # Add a dipole + quadrupole configuration
        {
            'field_config': FieldConfig.create_dipole_quadrupole(
                dipole_B=1.0, dipole_lat=45.0,
                quad_B=0.5, quad_lat=90.0
            )
        }
    ]
    
    # Create the experiment setup with parameter sets
    experiment_setup = ExperimentSetup(
        base_config=base_config,
        parameter_sets=parameter_sets,
        filename_template="custom_config_{field_config.components[0].lat_degree}.npz"
    )
    
    # Generate configurations
    configs = experiment_setup.generate_configs()
    
    logger.info(f"Generated {len(configs)} configurations from parameter sets")
    
    # Print details of each configuration
    for i, (config, filename) in enumerate(configs):
        logger.info(f"Configuration {i+1}:")
        logger.info(f"  Filename: {filename}")
        logger.info(f"  Number of field components: {len(config.field_config.components)}")
        for j, component in enumerate(config.field_config.components):
            logger.info(f"  Component {j+1}: {component.moment_type}, B={component.B_magnitude}, lat={component.lat_degree}°")
        logger.info(f"  Grid nside: {config.grid_config.nside}")
        logger.info(f"  Maximum radius: {config.tracer_config.R_max}")
    
    return configs

def demo_combined():
    """Demonstrate combining parameter ranges and explicit sets."""
    logger.info("\nDemonstrating combined parameter ranges and sets...")
    
    base_config = create_base_config()
    
    # Define parameter ranges
    parameter_ranges = {
        'field_config.components[0].lat_degree': [0.0, 90.0],
        'tracer_config.dt': [0.01, 0.005]
    }
    
    # Define explicit parameter sets
    parameter_sets = [
        {
            'field_config.components[0].lat_degree': 45.0,
            'field_config.components[0].B_magnitude': 3.0,
            'tracer_config.R_max': 20.0
        }
    ]
    
    # Create the experiment setup with both parameter ranges and sets
    experiment_setup = ExperimentSetup(
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        parameter_sets=parameter_sets,
        filename_template="combined_{field_config.components[0].lat_degree}deg_dt{tracer_config.dt}.npz"
    )
    
    # Generate configurations
    configs = experiment_setup.generate_configs()
    
    logger.info(f"Generated {len(configs)} configurations from combined approach")
    
    # Print details of each configuration
    for i, (config, filename) in enumerate(configs):
        logger.info(f"Configuration {i+1}:")
        logger.info(f"  Filename: {filename}")
        logger.info(f"  Dipole latitude: {config.field_config.components[0].lat_degree} degrees")
        logger.info(f"  Field strength: {config.field_config.components[0].B_magnitude}")
        logger.info(f"  Time step: {config.tracer_config.dt}")
        logger.info(f"  Maximum radius: {config.tracer_config.R_max}")
    
    return configs

def visualize_parameter_space(configs):
    """Visualize the parameter space of the generated configurations."""
    logger.info("\nVisualizing parameter space...")
    
    # Extract parameters for visualization
    latitudes = []
    field_strengths = []
    max_radii = []
    
    for config, _ in configs:
        latitudes.append(config.field_config.components[0].lat_degree)
        field_strengths.append(config.field_config.components[0].B_magnitude)
        max_radii.append(config.tracer_config.R_max)
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        latitudes, 
        field_strengths, 
        max_radii,
        c=max_radii,  # Color by max radius
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    ax.set_xlabel('Dipole Latitude (degrees)')
    ax.set_ylabel('Field Strength (B)')
    ax.set_zlabel('Maximum Radius (R)')
    
    plt.colorbar(scatter, ax=ax, label='Maximum Radius (R)')
    plt.title('Parameter Space of Generated Configurations')
    
    # Save the figure
    output_dir = Path("./output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "parameter_space.png", dpi=300, bbox_inches='tight')
    logger.info(f"Parameter space visualization saved to {output_dir / 'parameter_space.png'}")
    
    plt.close()

def main():
    """Run the demo."""
    logger.info("Starting ExperimentSetup demo...")
    
    # Demonstrate parameter ranges
    range_configs = demo_parameter_ranges()
    
    # Demonstrate parameter sets
    set_configs = demo_parameter_sets()
    
    # Demonstrate combined approach
    combined_configs = demo_combined()
    
    # Visualize the parameter space
    visualize_parameter_space(range_configs)
    
    logger.info("Demo completed successfully!")

if __name__ == "__main__":
    main() 