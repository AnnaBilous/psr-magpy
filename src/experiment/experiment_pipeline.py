"""
Modular pipeline implementation for running field line experiments.

This module provides a pipeline-based approach to running field line experiments,
with distinct stages that can be composed, tested independently, and reused.
"""

import os
import logging
import time
import traceback
from typing import Dict, Optional

from src.field.multipole import MultipoleField
from src.footpoint_grid.grid import FootpointGrid
from src.line.field_line import FieldLine
from src.line.calculator import FieldLineCalculator

from .experiment_config import ExperimentConfig, TracingConfig
from .run_experiment import log_experiment_status, get_file_size

logger = logging.getLogger(__name__)

# ---- Pipeline Stage Functions ----

def setup_logging(config: ExperimentConfig, experiment_number: int, log_file: Optional[str] = None) -> str:
    """Set up logging for the experiment.
    
    Args:
        config: Experiment configuration
        experiment_number: Number of the experiment in the series
        log_file: Path to the log file (default: None)
        
    Returns:
        Path to the log file
    """
    # Set default log file if not provided
    if log_file is None or not log_file:
        log_dir = config.output.output_dir
        log_file = os.path.join(log_dir, f"{config.name}_log.txt")
    
    # Log experiment start
    log_experiment_status(
        log_file, 
        experiment_number, 
        config.name, 
        "Started"
    )
    
    logger.info(f"Starting experiment: {config.name}")
    return log_file

def create_output_filename(config: ExperimentConfig) -> str:
    """Create output filename for the experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Output filename
    """
    return str(config.output.get_output_filename(config.name))

def create_field(config: ExperimentConfig) -> MultipoleField:
    """Create magnetic field from configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        MultipoleField object
    """
    field_components = [
        comp_config.create_component() 
        for comp_config in config.field_components
    ]
    return MultipoleField(field_components)

def create_grid(config: ExperimentConfig) -> FootpointGrid:
    """Create grid from configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        FootpointGrid object
    """
    grid = config.grid.create_grid()
    logger.info(f"Created {config.grid.grid_type} grid with {grid.nline} points")
    return grid

def create_template(config: ExperimentConfig) -> FieldLine:
    """Create field line template from configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        FieldLine template
    """
    template = FieldLine(**config.output.get_save_flags())
    logger.info(f"Successfully created template: {template}")
    return template

def create_calculator(field: MultipoleField, grid: FootpointGrid) -> FieldLineCalculator:
    """Create field line calculator.
    
    Args:
        field: MultipoleField object
        grid: FootpointGrid object
        
    Returns:
        FieldLineCalculator object
    """
    calculator = FieldLineCalculator(field, grid)
    logger.info(f"Successfully created calculator: {repr(calculator)}")
    return calculator

def calculate_field_lines(calculator: FieldLineCalculator, template: FieldLine, 
                         config: TracingConfig, output_filename: str):
    """Calculate field lines.
    
    Args:
        calculator: FieldLineCalculator object
        template: FieldLine template
        config: TracingConfig object
        output_filename: Output filename
        
    Returns:
        Field line calculation results
    """
    # Update the filename in the tracing config
    config.filename = output_filename
    
    # Calculate field lines
    logger.info("Starting field line calculations...")
    return calculator.get_field_lines(
        template=template,
        config=config
    )

def save_configuration(config: ExperimentConfig, output_filename: str) -> str:
    """Save experiment configuration to file.
    
    Args:
        config: Experiment configuration
        output_filename: Base output filename
        
    Returns:
        Path to the saved configuration file
    """
    config_path = output_filename + "_config.json"
    config.save(config_path)
    logger.info(f"Configuration saved to: {config_path}")
    return config_path

def calculate_file_sizes(output_filename: str) -> Dict[str, float]:
    """Calculate sizes of output files.
    
    Args:
        output_filename: Base output filename
        
    Returns:
        Dictionary of file sizes in MB
    """
    return {
        "Data": get_file_size(output_filename + "*.fits", use_wildcard=True)
    }

def log_completion(log_file: str, experiment_number: int, experiment_name: str, 
                  start_time: float, file_sizes: Dict[str, float]) -> None:
    """Log experiment completion.
    
    Args:
        log_file: Path to the log file
        experiment_number: Number of the experiment in the series
        experiment_name: Name of the experiment
        start_time: Start time of the experiment
        file_sizes: Dictionary of file sizes
    """
    duration = time.time() - start_time
    log_experiment_status(
        log_file, 
        experiment_number, 
        experiment_name, 
        "Completed", 
        duration=duration,
        file_sizes=file_sizes
    )

def log_failure(log_file: str, experiment_number: int, experiment_name: str, 
               start_time: float, exception: Exception) -> None:
    """Log experiment failure.
    
    Args:
        log_file: Path to the log file
        experiment_number: Number of the experiment in the series
        experiment_name: Name of the experiment
        start_time: Start time of the experiment
        exception: Exception that caused the failure
    """
    duration = time.time() - start_time
    error_message = f"{type(exception).__name__}: {str(exception)}\n{traceback.format_exc()}"
    log_experiment_status(
        log_file, 
        experiment_number, 
        experiment_name, 
        "Failed", 
        duration=duration,
        error=error_message
    )
    
    logger.error(f"Error during experiment: {str(exception)}", exc_info=True)

# ---- Pipeline Execution ----

def run_experiment_pipeline(config: ExperimentConfig, experiment_number: int = 1, 
                           log_file: Optional[str] = None):
    """Run the experiment pipeline.
    
    Args:
        config: Experiment configuration
        experiment_number: Number of the experiment in the series
        log_file: Path to the log file
        
    Returns:
        Results of the experiment
    """
    start_time = time.time()
    
    try:
        # Setup stage
        log_file = setup_logging(config, experiment_number, log_file)
        output_filename = create_output_filename(config)
        
        # Preparation stage
        field = create_field(config)
        grid = create_grid(config)
        template = create_template(config)
        calculator = create_calculator(field, grid)
        
        # Calculation stage
        results = calculate_field_lines(calculator, template, config.tracing, output_filename)
        
        # Output stage
        save_configuration(config, output_filename)
        file_sizes = calculate_file_sizes(output_filename)
        
        # Logging stage
        log_completion(log_file, experiment_number, config.name, start_time, file_sizes)
        
        return results
        
    except Exception as e:
        # Error handling stage
        log_failure(log_file, experiment_number, config.name, start_time, e)
        raise 