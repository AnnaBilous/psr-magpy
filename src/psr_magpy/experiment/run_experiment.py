import glob
import logging
import os
import time
import traceback
from datetime import datetime

from psr_magpy.field.multipole import MultipoleField
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.calculator import FieldLineCalculator

from .experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


def get_file_size(filepath, use_wildcard=False):
    """Get file size in MB.

    Args:
        filepath: Path to the file or pattern
        use_wildcard: If True, treat filepath as a pattern and sum sizes of all matching files

    Returns:
        File size in MB
    """
    if use_wildcard:
        # Get the directory and filename pattern
        directory = os.path.dirname(filepath)
        if not directory:
            directory = "."

        # Create the pattern for glob
        pattern = os.path.basename(filepath)
        full_pattern = os.path.join(directory, pattern)

        # Find all matching files
        matching_files = glob.glob(full_pattern)

        # Sum the sizes
        total_size = 0
        for file_path in matching_files:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)

        return total_size / (1024 * 1024)  # Convert bytes to MB
    else:
        if not os.path.exists(filepath):
            return 0
        return os.path.getsize(filepath) / (1024 * 1024)  # Convert bytes to MB


def log_experiment_status(
    log_file,
    experiment_number,
    experiment_name,
    status,
    duration=None,
    error=None,
    file_sizes=None,
):
    """Log experiment status to a text file.

    Args:
        log_file: Path to the log file
        experiment_number: Number of the experiment in the series
        experiment_name: Name of the experiment
        status: Status of the experiment (e.g., "Started", "Completed", "Failed")
        duration: Duration of the experiment in seconds (optional)
        error: Error message if the experiment failed (optional)
        file_sizes: Dictionary of file sizes in MB (optional)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Format the log message
    log_message = (
        f"[{timestamp}] Experiment #{experiment_number}: {experiment_name} - {status}"
    )

    if duration is not None:
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        log_message += f" (Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s)"

    if file_sizes:
        log_message += "\nFile sizes:"
        for file_type, size in file_sizes.items():
            log_message += f"\n  - {file_type}: {size:.5f} MB"

    if error:
        log_message += f"\nError: {error}"

    # Write to log file
    with open(log_file, "a") as f:
        f.write(log_message + "\n\n")

    # Also log to console
    logger.info(log_message)


def run_experiment(config: ExperimentConfig, experiment_number=1, log_file=None):
    """
    Run field line calculation experiment using the provided configuration.
    
    This function sets up and executes a pulsar magnetosphere simulation based on
    the provided configuration. It handles field component creation, grid setup,
    and field line calculations.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration object containing all settings for the experiment.
    experiment_number : int, optional
        Number of the experiment in the series. Default is 1.
    log_file : str, optional
        Path to the log file. If None, a default log file is created in the output directory.
        
    Returns
    -------
    object
        Results of the experiment, typically field line data.
        
    Raises
    ------
    TypeError
        If field components are not properly configured.
    ValueError
        If configuration parameters are invalid.
    IOError
        If output files cannot be written.
        
    Notes
    -----
    The function logs experiment progress and results to both a log file and the
    console using the logging module.
    
    Examples
    --------
    >>> from psr_magpy.experiment.experiment_config import ExperimentConfig
    >>> from psr_magpy.field.components import DipoleFieldComponent
    >>> from psr_magpy.grid.grid_config import GridConfig
    >>> from psr_magpy.experiment.output_config import OutputConfig
    >>> from psr_magpy.experiment.tracing_config import TracingConfig
    
    Basic experiment with a dipole field:
    
    >>> # Create field component configuration
    >>> field_config = FieldComponentConfig(
    ...     component_type="dipole",
    ...     parameters={"strength": 1.0}
    ... )
    
    >>> # Create grid configuration
    >>> grid_config = GridConfig(
    ...     grid_type="polar",
    ...     parameters={"r_max": 5.0, "n_r": 10, "n_theta": 20}
    ... )
    
    >>> # Create output configuration
    >>> output_config = OutputConfig(
    ...     output_dir="./results",
    ...     save_coordinates=True,
    ...     save_field=True
    ... )
    
    >>> # Create tracing configuration
    >>> tracing_config = TracingConfig(
    ...     step_size=0.01,
    ...     max_steps=1000,
    ...     min_radius=1.0
    ... )
    
    >>> # Create experiment configuration
    >>> config = ExperimentConfig(
    ...     name="dipole_test",
    ...     field_components=[field_config],
    ...     grid=grid_config,
    ...     output=output_config,
    ...     tracing=tracing_config
    ... )
    
    >>> # Run the experiment
    >>> results = run_experiment(config)
    
    Running an experiment with a custom log file:
    
    >>> results = run_experiment(config, experiment_number=2, log_file="custom_log.txt")
    
    See Also
    --------
    ExperimentConfig : Configuration class for experiments
    FieldLineCalculator : Core calculator for field line tracing
    """
    start_time = time.time()

    # Set default log file if not provided
    if log_file is None or not log_file:
        log_dir = config.output.output_dir
        log_file = os.path.join(log_dir, f"{config.name}_log.txt")

    # Log experiment start
    log_experiment_status(log_file, experiment_number, config.name, "Started")

    logger.info(f"Starting experiment: {config.name}")

    # Create output filename
    output_filename = str(config.output.get_output_filename(config.name))

    try:
        # Create field components
        field_components = [
            comp_config.create_component() for comp_config in config.field_components
        ]
                
        field = MultipoleField(field_components)

        # Create grid
        grid = config.grid.create_grid()
        logger.info(f"Created {config.grid.grid_type} grid with {grid.nline} points")

        # Configure template using save flags from OutputConfig
        template = FieldLine(**config.output.get_save_flags())
        logger.info(f"Successfully created template: {template}")

        # Initialize calculator
        calculator = FieldLineCalculator(field, grid)
        logger.info(f"Successfully created calculator: {repr(calculator)}")

        config.tracing.filename = output_filename

        # Calculate field lines
        logger.info("Starting field line calculations...")
        results = calculator.get_field_lines(template=template, config=config.tracing)

        # Save configuration
        config_path = output_filename + "_config.json"
        config.save(config_path)
        logger.info(f"Configuration saved to: {config_path}")

        # Calculate file sizes - use wildcard to find all data files
        file_sizes = {
            "Data": get_file_size(output_filename + "*.fits", use_wildcard=True)
        }

        # Log experiment completion
        duration = time.time() - start_time
        log_experiment_status(
            log_file,
            experiment_number,
            config.name,
            "Completed",
            duration=duration,
            file_sizes=file_sizes,
        )

        return results

    except Exception as e:
        # Log experiment failure
        duration = time.time() - start_time
        error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        log_experiment_status(
            log_file,
            experiment_number,
            config.name,
            "Failed",
            duration=duration,
            error=error_message,
        )

        logger.error(f"Error during experiment: {str(e)}", exc_info=True)
        raise
