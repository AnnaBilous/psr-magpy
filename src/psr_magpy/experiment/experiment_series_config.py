import json
import os
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Union
from pathlib import Path
import numpy as np

from psr_magpy.grid.grid import GridType
from psr_magpy.line.calculator import TracingConfig
from psr_magpy.experiment.field_factory_config import MultipoleFieldFactory
from psr_magpy.experiment.experiment_config import (
    ExperimentConfig,
    GridConfig,
    OutputConfig,
    FieldComponentConfig,
)
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.json_utils import NumpyEncoder

logger = get_logger(__name__, console_format_brief=True)


@dataclass
class ExperimentSeriesConfig:
    """Configuration for a series of related experiments."""

    # Series metadata
    name: str
    description: Optional[str] = None

    # Base configurations
    grid: GridConfig = None
    tracing: TracingConfig = None
    output: OutputConfig = OutputConfig()

    # Field component factories for generating multiple experiments
    field_factories: MultipoleFieldFactory = None
    
    def __post_init__(self):
        """Log the creation of the experiment series configuration."""
        logger.debug(f"Created experiment series configuration: {self.name}")
        if self.description:
            logger.debug(f"Description: {self.description}")
        
        # Log component details
        if self.field_factories:
            logger.debug(f"Field factories defined: {self.field_factories}")
        if self.grid:
            logger.debug(f"Grid type: {self.grid.grid_type}")
        if self.tracing:
            logger.debug(f"Tracing config: R_max={self.tracing.R_max}, R_min={self.tracing.R_min}, dt={self.tracing.dt}")

    def expand_to_experiments(self) -> List[ExperimentConfig]:
        """Expand field factories into a list of individual experiment configs.

        Returns:
            List[ExperimentConfig]: A list of experiment configurations, each with
                                   specific field component settings.
        """
        logger.info(f"Expanding experiment series '{self.name}' to individual experiments")
        
        if not self.field_factories:
            error_msg = "No field factories defined for experiment series"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Expand the field factories into all possible component combinations
        logger.debug("Expanding field factories into component combinations")
        expanded_component_sets = self.field_factories._expand_parameters()
        logger.debug(f"Generated {len(expanded_component_sets)} component combinations")

        # Create an experiment config for each expanded component set
        experiment_configs = []

        for i, component_set in enumerate(expanded_component_sets):
            logger.debug(f"Creating experiment configuration for experiment {i}")
            
            # Convert Parameters objects to FieldComponentConfig objects
            field_components = []
            for params in component_set:
                logger.debug(f"Creating field component from parameters: {params}")
                field_component = FieldComponentConfig(
                    type=params.moment_type.value,
                    B_magnitude=params.B_magnitude,
                    moment_orientation_latitude_degree=params.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=params.moment_orientation_longitude_degree,
                )
                field_components.append(field_component)

            # Create a unique name for this experiment
            experiment_name = f"{self.name}_experiment_{i:04d}"

            # Create a copy of the tracing config and update it
            tracing_config = deepcopy(self.tracing)
            tracing_config.save_results = True

            # Set filename based on output file_prefix and experiment number
            if hasattr(self.output, "file_prefix") and self.output.file_prefix:
                tracing_config.filename = f"{self.output.file_prefix}_{experiment_name}"
                logger.debug(f"Setting filename with prefix: {tracing_config.filename}")
            else:
                tracing_config.filename = experiment_name
                logger.debug(f"Setting filename without prefix: {tracing_config.filename}")

            # Create the experiment config
            experiment_config = ExperimentConfig(
                name=experiment_name,
                description=(
                    f"{self.description} (Variant {i})"
                    if self.description
                    else f"Variant {i}"
                ),
                grid=deepcopy(self.grid),
                tracing=tracing_config,
                field_components=field_components,
                output=deepcopy(self.output),
            )

            experiment_configs.append(experiment_config)
            logger.debug(f"Created experiment configuration: {experiment_name}")

        logger.debug(f"Created {len(experiment_configs)} experiment configurations in total")
        return experiment_configs

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        filepath = Path(filepath)
        logger.debug(f"Saving experiment series configuration to {filepath}")
        
        # Convert to dictionary
        config_dict = asdict(self)
        logger.debug("Converted configuration to dictionary")

        # Convert grid_type to string if needed
        if isinstance(config_dict["grid"]["grid_type"], GridType):
            logger.debug("Converting grid_type enum to string")
            config_dict["grid"]["grid_type"] = config_dict["grid"]["grid_type"].value

        # Handle field factories serialization
        if self.field_factories:
            logger.debug("Serializing field factories")
            config_dict["field_factories"] = self.field_factories.to_dict()

        # Handle OutputConfig serialization
        logger.debug("Serializing output configuration")
        config_dict["output"] = self.output.to_dict()

        # Convert output_dir Path to string if needed
        if isinstance(config_dict["output"]["output_dir"], Path):
            logger.debug("Converting output_dir Path to string")
            config_dict["output"]["output_dir"] = str(
                config_dict["output"]["output_dir"]
            )

        # Ensure directory exists
        try:
            logger.debug(f"Creating directory if needed: {filepath.parent}")
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Save to file with the custom encoder
            with open(filepath, "w") as f:
                json.dump(config_dict, f, indent=4, cls=NumpyEncoder)
                logger.debug(f"Experiment series configuration successfully saved to {filepath}")
        except PermissionError:
            error_msg = f"Permission denied when writing to {filepath}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
        except OSError as e:
            error_msg = f"Error saving configuration to {filepath}: {str(e)}"
            logger.error(error_msg)
            raise

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentSeriesConfig":
        """Load configuration from a JSON file."""
        filepath = Path(filepath)
        logger.info(f"Loading experiment series configuration from {filepath}")
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                logger.debug(f"Successfully read JSON data from {filepath}")
        except FileNotFoundError:
            error_msg = f"Configuration file not found: {filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file {filepath}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Convert nested configs
            logger.debug("Converting grid configuration")
            data["grid"] = GridConfig(**data["grid"])
            
            logger.debug("Converting tracing configuration")
            data["tracing"] = TracingConfig(**data["tracing"])
            
            logger.debug("Converting output configuration")
            data["output"] = OutputConfig.from_dict(data["output"])

            # Handle field factories
            if "field_factories" in data:
                logger.debug("Converting field factories")
                data["field_factories"] = MultipoleFieldFactory.from_dict(
                    data["field_factories"]
                )

            config = cls(**data)
            logger.info(f"Successfully loaded configuration for experiment series: {config.name}")
            return config
            
        except KeyError as e:
            error_msg = f"Missing required field in configuration: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except TypeError as e:
            error_msg = f"Invalid configuration structure: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


def run_experiment_series(
    config: ExperimentSeriesConfig,
    output_dir: Optional[str] = None,
    clear_log: bool = False,
    save_individual_configs: bool = True,
):
    """Run a series of experiments based on the provided configuration.

    Args:
        config: The experiment series configuration
        output_dir: Optional directory to save results (overrides config.output.output_dir)
        clear_log: Whether to clear the existing log file before starting (default: False)
        save_individual_configs: Whether to save individual experiment configs (default: True)

    Returns:
        List of experiment results
    """
    from psr_magpy.experiment.run_experiment import run_experiment
    import logging
    import os

    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment series: {config.name}")

    # Override output directory if provided
    if output_dir:
        logger.info(f"Overriding output directory to: {output_dir}")
        config.output.output_dir = output_dir

    # Create log file path using the series name
    log_file = os.path.join(config.output.output_dir, f"{config.name}_log.txt")
    logger.debug(f"Log file path: {log_file}")

    # Clear the log file if requested
    if clear_log and os.path.exists(log_file):
        logger.info(f"Clearing existing log file: {log_file}")
        try:
            with open(log_file, "w") as f:
                f.write(f"# Experiment series log: {config.name}\n")
                f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except IOError as e:
            logger.error(f"Failed to clear log file: {e}")
            raise

    # Expand to individual experiments
    try:
        experiments = config.expand_to_experiments()
        for exp in experiments:
            logger.debug(f"Experiment {exp.name}")
            
            # Save individual experiment configs if requested
            if save_individual_configs:
                config_dir = os.path.join(config.output.output_dir, "configs")
                os.makedirs(config_dir, exist_ok=True)
                config_path = os.path.join(config_dir, f"{exp.name}_config.json")
                try:
                    exp.save(config_path)
                    logger.debug(f"Saved individual experiment config to {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to save individual experiment config: {e}")
                    
        logger.info(f"Generated {len(experiments)} experiment configurations")
    except Exception as e:
        logger.error(f"Failed to expand experiment configurations: {e}")
        raise

    # Run each experiment
    results = []
    for i, experiment in enumerate(experiments):
        logger.debug(f"Running experiment {i+1}/{len(experiments)}: {experiment.name}")
        try:
            result = run_experiment(experiment, experiment_number=i + 1, log_file=log_file)
            results.append(result)
            logger.debug(f"Experiment {i+1} completed successfully")
        except Exception as e:
            logger.error(f"Experiment {i+1} failed: {e}")
            # Append None to maintain index correspondence
            results.append(None)

    logger.info(f"Completed experiment series: {config.name}")
    logger.debug(f"Results count: {len(results)}")
    return results


def resume_experiment_series(
    config: ExperimentSeriesConfig,
    output_dir: Optional[str] = None,
    clear_log: bool = False,
):
    """Resume a series of experiments that was interrupted.

    This function reads the log file to determine which experiments have been completed
    successfully, and continues the series from where it left off.

    Args:
        config: The experiment series configuration
        output_dir: Optional directory to save results (overrides config.output.output_dir)
        clear_log: Whether to clear the existing log file before starting (default: False)
                  Note: If this is True, all experiments will be run again.

    Returns:
        List of experiment results
    """
    from psr_magpy.experiment.run_experiment import run_experiment
    import logging
    import re

    logger = logging.getLogger(__name__)
    logger.info(f"Resuming experiment series: {config.name}")

    # Override output directory if provided
    if output_dir:
        logger.info(f"Overriding output directory to: {output_dir}")
        config.output.output_dir = output_dir

    # Create log file path using the series name
    log_file = os.path.join(config.output.output_dir, f"{config.name}_log.txt")
    logger.debug(f"Log file path: {log_file}")

    # Clear the log file if requested
    if clear_log and os.path.exists(log_file):
        logger.warning(
            f"Clearing existing log file: {log_file} - all experiments will be run again"
        )
        try:
            with open(log_file, "w") as f:
                f.write(f"# Experiment series log: {config.name}\n")
                f.write(f"# Resumed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except IOError as e:
            logger.error(f"Failed to clear log file: {e}")
            raise

    # Expand to individual experiments
    try:
        experiments = config.expand_to_experiments()
        logger.info(f"Generated {len(experiments)} experiment configurations")
    except Exception as e:
        logger.error(f"Failed to expand experiment configurations: {e}")
        raise

    # Check which experiments have been completed (if not clearing the log)
    completed_experiments = set()
    if not clear_log and os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                log_content = f.read()

            # Find all completed experiments using regex
            completed_pattern = re.compile(r"Experiment #(\d+): (.*?) - Completed")
            for match in completed_pattern.finditer(log_content):
                experiment_number = int(match.group(1))
                experiment_name = match.group(2)
                completed_experiments.add(experiment_name)
                logger.info(
                    f"Found completed experiment #{experiment_number}: {experiment_name}"
                )
            
            logger.info(f"Found {len(completed_experiments)} completed experiments")
        except IOError as e:
            logger.error(f"Failed to read log file: {e}")
            # Continue with empty completed_experiments set
            pass

    # Run remaining experiments
    results = []
    for i, experiment in enumerate(experiments):
        if not clear_log and experiment.name in completed_experiments:
            logger.info(
                f"Skipping experiment {i+1}/{len(experiments)}: {experiment.name} (already completed)"
            )
            results.append(None)  # Placeholder for completed experiment
        else:
            logger.info(
                f"Running experiment {i+1}/{len(experiments)}: {experiment.name}"
            )
            try:
                result = run_experiment(
                    experiment, experiment_number=i + 1, log_file=log_file
                )
                results.append(result)
                logger.debug(f"Experiment {i+1} completed successfully")
            except Exception as e:
                logger.error(f"Experiment {i+1} failed: {e}")
                # Append None to maintain index correspondence
                results.append(None)

    logger.info(f"Completed experiment series: {config.name}")
    logger.debug(f"Results count: {len(results)}")
    return results


def analyze_experiment_log(
    config: ExperimentSeriesConfig, output_dir: Optional[str] = None
):
    """Analyze the experiment log file and provide a summary.

    Args:
        config: The experiment series configuration
        output_dir: Optional directory where results are saved (overrides config.output.output_dir)

    Returns:
        Dict containing summary statistics
    """
    import re
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing experiment log for series: {config.name}")

    # Override output directory if provided
    if output_dir:
        output_dir_path = output_dir
        logger.debug(f"Using provided output directory: {output_dir_path}")
    else:
        output_dir_path = config.output.output_dir
        logger.debug(f"Using config output directory: {output_dir_path}")

    # Create log file path using the series name
    log_file = os.path.join(output_dir_path, f"{config.name}_log.txt")
    logger.debug(f"Log file path: {log_file}")

    if not os.path.exists(log_file):
        logger.warning(f"Log file not found: {log_file}")
        return {
            "total_experiments": 0,
            "completed": 0,
            "failed": 0,
            "pending": 0,
            "total_duration": 0,
            "total_data_size": 0,
            "experiments": [],
        }

    try:
        with open(log_file, "r") as f:
            log_content = f.read()
            logger.debug(f"Successfully read log file: {log_file}")
    except IOError as e:
        logger.error(f"Failed to read log file: {e}")
        raise

    # Find all experiment entries
    experiment_data = {}
    logger.debug("Parsing log file for experiment data")

    # Find started experiments
    started_pattern = re.compile(r"\[([^\]]+)\] Experiment #(\d+): (.*?) - Started")
    for match in started_pattern.finditer(log_content):
        timestamp = match.group(1)
        experiment_number = int(match.group(2))
        experiment_name = match.group(3)

        if experiment_name not in experiment_data:
            experiment_data[experiment_name] = {
                "number": experiment_number,
                "name": experiment_name,
                "started": timestamp,
                "status": "Started",
            }
            logger.debug(f"Found started experiment: {experiment_name}")

    # Find completed experiments
    completed_pattern = re.compile(
        r"\[([^\]]+)\] Experiment #(\d+): (.*?) - Completed \(Duration: (\d+)h (\d+)m (\d+)s\)"
    )
    for match in completed_pattern.finditer(log_content):
        timestamp = match.group(1)
        experiment_number = int(match.group(2))
        experiment_name = match.group(3)
        hours = int(match.group(4))
        minutes = int(match.group(5))
        seconds = int(match.group(6))
        duration = hours * 3600 + minutes * 60 + seconds

        if experiment_name in experiment_data:
            experiment_data[experiment_name].update(
                {"completed": timestamp, "duration": duration, "status": "Completed"}
            )
            logger.debug(f"Found completed experiment: {experiment_name} (Duration: {duration}s)")

    # Find file sizes
    size_pattern = re.compile(
        r"Experiment #(\d+): (.*?) - Completed.*?File sizes:\s+- Data: ([\d.]+) MB",
        re.DOTALL,
    )
    for match in size_pattern.finditer(log_content):
        experiment_number = int(match.group(1))
        experiment_name = match.group(2)
        data_size = float(match.group(3))

        if experiment_name in experiment_data:
            experiment_data[experiment_name]["data_size"] = data_size
            logger.debug(f"Found file size for experiment {experiment_name}: {data_size} MB")

    # Find failed experiments
    failed_pattern = re.compile(r"\[([^\]]+)\] Experiment #(\d+): (.*?) - Failed")
    for match in failed_pattern.finditer(log_content):
        timestamp = match.group(1)
        experiment_number = int(match.group(2))
        experiment_name = match.group(3)

        if experiment_name in experiment_data:
            experiment_data[experiment_name].update(
                {"failed": timestamp, "status": "Failed"}
            )
            logger.debug(f"Found failed experiment: {experiment_name}")

    # Calculate summary statistics
    total_experiments = len(experiment_data)
    completed = sum(
        1 for data in experiment_data.values() if data.get("status") == "Completed"
    )
    failed = sum(
        1 for data in experiment_data.values() if data.get("status") == "Failed"
    )
    pending = total_experiments - completed - failed

    total_duration = sum(data.get("duration", 0) for data in experiment_data.values())
    total_data_size = sum(data.get("data_size", 0) for data in experiment_data.values())

    # Format the summary
    summary = {
        "total_experiments": total_experiments,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "total_duration": total_duration,
        "total_data_size": total_data_size,
        "experiments": list(experiment_data.values()),
    }

    # Log summary
    logger.info(f"Experiment series: {config.name}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Pending: {pending}")

    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Total data size: {total_data_size:.5f} MB")

    return summary
