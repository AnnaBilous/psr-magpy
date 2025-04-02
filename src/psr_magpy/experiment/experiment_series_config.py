
import json
import os
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Union
from pathlib import Path

from psr_magpy.grid.grid import GridType
from psr_magpy.line.calculator import TracingConfig
from psr_magpy.experiment.field_factory_config import MultipoleFieldFactory
from psr_magpy.experiment.experiment_config import (
    ExperimentConfig,
    GridConfig,
    OutputConfig,
    FieldComponentConfig,
)


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

    def expand_to_experiments(self) -> List[ExperimentConfig]:
        """Expand field factories into a list of individual experiment configs.

        Returns:
            List[ExperimentConfig]: A list of experiment configurations, each with
                                   specific field component settings.
        """
        if not self.field_factories:
            raise ValueError("No field factories defined for experiment series")

        # Expand the field factories into all possible component combinations
        expanded_component_sets = self.field_factories._expand_parameters()

        # Create an experiment config for each expanded component set
        experiment_configs = []

        for i, component_set in enumerate(expanded_component_sets):
            # Convert Parameters objects to FieldComponentConfig objects
            field_components = []
            for params in component_set:
                field_component = FieldComponentConfig(
                    type=params.moment_type.value,
                    B_magnitude=params.B_magnitude,
                    moment_orientation_latitude_degree=params.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=params.moment_orientation_longitude_degree,
                )
                field_components.append(field_component)

            # Create a unique name for this experiment
            experiment_name = f"{self.name}_variant_{i}"

            # Create a copy of the tracing config and update it
            tracing_config = deepcopy(self.tracing)
            tracing_config.save_results = True

            # Set filename based on output file_prefix and experiment number
            if hasattr(self.output, "file_prefix") and self.output.file_prefix:
                tracing_config.filename = f"{self.output.file_prefix}_{experiment_name}"
            else:
                tracing_config.filename = experiment_name

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

        return experiment_configs

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        # Convert to dictionary
        config_dict = asdict(self)

        # Convert grid_type to string if needed
        if isinstance(config_dict["grid"]["grid_type"], GridType):
            config_dict["grid"]["grid_type"] = config_dict["grid"]["grid_type"].value

        # Handle field factories serialization
        if self.field_factories:
            config_dict["field_factories"] = self.field_factories.to_dict()

        # Handle OutputConfig serialization
        config_dict["output"] = self.output.to_dict()

        # Convert output_dir Path to string if needed
        if isinstance(config_dict["output"]["output_dir"], Path):
            config_dict["output"]["output_dir"] = str(
                config_dict["output"]["output_dir"]
            )

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentSeriesConfig":
        """Load configuration from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert nested configs
        data["grid"] = GridConfig(**data["grid"])
        data["tracing"] = TracingConfig(**data["tracing"])
        data["output"] = OutputConfig.from_dict(data["output"])

        # Handle field factories
        if "field_factories" in data:
            data["field_factories"] = MultipoleFieldFactory.from_dict(
                data["field_factories"]
            )

        return cls(**data)


def run_experiment_series(
    config: ExperimentSeriesConfig,
    output_dir: Optional[str] = None,
    clear_log: bool = False,
):
    """Run a series of experiments based on the provided configuration.

    Args:
        config: The experiment series configuration
        output_dir: Optional directory to save results (overrides config.output.output_dir)
        clear_log: Whether to clear the existing log file before starting (default: False)

    Returns:
        List of experiment results
    """
    from psr_magpy.experiment.run_experiment import run_experiment
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment series: {config.name}")

    # Override output directory if provided
    if output_dir:
        config.output.output_dir = output_dir

    # Create log file path using the series name
    log_file = os.path.join(config.output.output_dir, f"{config.name}_log.txt")

    # Clear the log file if requested
    if clear_log and os.path.exists(log_file):
        logger.info(f"Clearing existing log file: {log_file}")
        with open(log_file, "w") as f:
            f.write(f"# Experiment series log: {config.name}\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Expand to individual experiments
    experiments = config.expand_to_experiments()
    logger.info(f"Generated {len(experiments)} experiment configurations")

    # Run each experiment
    results = []
    for i, experiment in enumerate(experiments):
        logger.info(f"Running experiment {i+1}/{len(experiments)}: {experiment.name}")
        result = run_experiment(experiment, experiment_number=i + 1, log_file=log_file)
        results.append(result)

    logger.info(f"Completed experiment series: {config.name}")
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
    from src.experiment.run_experiment import run_experiment
    import logging
    import re

    logger = logging.getLogger(__name__)
    logger.info(f"Resuming experiment series: {config.name}")

    # Override output directory if provided
    if output_dir:
        config.output.output_dir = output_dir

    # Create log file path using the series name
    log_file = os.path.join(config.output.output_dir, f"{config.name}_log.txt")

    # Clear the log file if requested
    if clear_log and os.path.exists(log_file):
        logger.warning(
            f"Clearing existing log file: {log_file} - all experiments will be run again"
        )
        with open(log_file, "w") as f:
            f.write(f"# Experiment series log: {config.name}\n")
            f.write(f"# Resumed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Expand to individual experiments
    experiments = config.expand_to_experiments()
    logger.info(f"Generated {len(experiments)} experiment configurations")

    # Check which experiments have been completed (if not clearing the log)
    completed_experiments = set()
    if not clear_log and os.path.exists(log_file):
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
            result = run_experiment(
                experiment, experiment_number=i + 1, log_file=log_file
            )
            results.append(result)

    logger.info(f"Completed experiment series: {config.name}")
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

    # Override output directory if provided
    if output_dir:
        output_dir_path = output_dir
    else:
        output_dir_path = config.output.output_dir

    # Create log file path using the series name
    log_file = os.path.join(output_dir_path, f"{config.name}_log.txt")

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

    with open(log_file, "r") as f:
        log_content = f.read()

    # Find all experiment entries
    experiment_data = {}

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
