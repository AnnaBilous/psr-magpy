"""Experiment series configuration and generation.

This module provides functionality to create multiple experiments from flexible
parameter inputs, including lists for parameter variation.
"""

from typing import List, Optional, Union, Dict, Any, ClassVar
from pydantic import BaseModel
import yaml
from pathlib import Path
import shutil

from .experiment_config import (
    ExperimentConfig,
    FieldConfig,
    GridConfig,
    TracingConfig,
    SphericalGridConfig,
)
from .factory_configs import (
    FieldFactoryConfig,
    GridFactoryConfig,
    TemplateFactoryConfig,
    TracingFactoryConfig,
    expand_field_configs,
    expand_grid_configs,
    expand_tracing_configs,
    expand_template_configs,
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

    def model_dump(self, **kwargs):
        """Custom model_dump to ensure grid type information is preserved."""
        data = super().model_dump(**kwargs)
        
        # Ensure grid type information is preserved
        if "grid" in data and hasattr(self.grid, 'model_dump'):
            data["grid"] = self.grid.model_dump(**kwargs)
        
        return data

    def create_experiment_series(
        self,
        save_configs: bool = True,
        delete_existing_configs: bool = False,
    ) -> List[ExperimentConfig]:
        """Create experiment series from configuration.

        Args:
            save_configs: Whether to save individual experiment configs
            delete_existing_configs: Whether to delete existing configs

        Returns:
            List of experiment configurations
        """
        logger.info("Creating experiment series: %s", self.name)

        # Create output directory structure
        self.create_directory_structure(delete_existing_configs)

        # Create experiments
        experiments = self._create_experiments()
        logger.info("Created %d experiments in series", len(experiments))

        # Save series config
        if save_configs:
            config_file = self.get_configs_dir() / f"{self.name}_config.yaml"
            if delete_existing_configs or not config_file.exists():
                self.save(str(config_file))
                logger.debug("Saved series config: %s", config_file)

        # Save individual experiment configs
        if save_configs:
            configs_dir = self.get_configs_dir()
            for experiment in experiments:
                config_file = configs_dir / f"{experiment.name}.yaml"
                if delete_existing_configs or not config_file.exists():
                    experiment.save(str(config_file))
                    logger.debug("Saved experiment config: %s", config_file)

        return experiments

    def _save_grid_and_update_config(self, experiment: ExperimentConfig) -> None:
        """Save grid for custom/random grids and update config with file path."""
        grid_type = experiment.grid.get_grid_type()

        # Only save grids for custom and random grid types
        if grid_type not in ["custom", "random"]:
            return

        # Create grid instance
        grid = experiment.grid.create_grid()

        # Save grid to file
        grids_dir = self.get_grids_dir()
        grids_dir.mkdir(parents=True, exist_ok=True)

        grid_path = grids_dir / f"{experiment.name}.fits"
        try:
            if grid is not None:
                grid.save_to_fits(str(grid_path))
                logger.debug(f"Saved grid for {experiment.name} to {grid_path}")

                # Update the experiment config with the grid file path
                if grid_type == "random" and experiment.grid.random:
                    experiment.grid.random.grid_file_path = str(grid_path)
                    logger.debug(f"Updated RandomGrid config with file path: {grid_path}")
                elif grid_type == "custom" and experiment.grid.custom:
                    experiment.grid.custom.grid_file_path = str(grid_path)
                    logger.debug(f"Updated CustomGrid config with file path: {grid_path}")
            else:
                logger.warning(f"Grid is None for {experiment.name}, skipping save")

        except Exception as e:
            logger.warning(f"Failed to save grid for {experiment.name}: {e}")

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
        """Get path to configs directory.

        Returns:
            Path to configs directory
        """
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
    
    def get_subdirectories(self) -> List[str]:
        """Get list of all subdirectory names."""
        return [self.LINES_DIR, self.GRIDS_DIR, self.CONFIGS_DIR, self.LOGS_DIR]
    
    def get_cleanup_directories(self) -> List[str]:
        """Get list of directories to clean up (only configs)."""
        return [self.CONFIGS_DIR]

    def save(self, filename: Optional[str] = None) -> None:
        """Save configuration to YAML file in the configs directory.

        Args:
            filename: Optional filename. If not provided, uses "{series_name}_config.yaml"
        """
        from psr_magpy.io import get_io_factory
        from psr_magpy.io.config_io import ExperimentSeriesConfigIO

        if filename is None:
            filename = str(self.get_configs_dir() / f"{self.name}_config.yaml")

        # Ensure registration
        if type(self) not in get_io_factory()._config_registry:
            get_io_factory().register_config_io(type(self), ExperimentSeriesConfigIO)

        # Save with overwrite=True
        get_io_factory().save_config(self, filename)

    @classmethod
    def load(cls, config_file: Union[str, Path]) -> "ExperimentSeriesConfig":
        """Load configuration from YAML file.

        Args:
            config_file: Path to configuration file

        Returns:
            Loaded configuration
        """
        from psr_magpy.io import get_io_factory
        from psr_magpy.io.config_io import ExperimentSeriesConfigIO

        # Ensure registration
        if cls not in get_io_factory()._config_registry:
            get_io_factory().register_config_io(cls, ExperimentSeriesConfigIO)

        return get_io_factory().load_config(cls, config_file)

    def print_yaml(self):
        """Print configuration as pretty YAML to screen."""
        from psr_magpy.io import get_io_factory
        from psr_magpy.io.config_io import ExperimentSeriesConfigIO

        # Ensure registration
        if type(self) not in get_io_factory()._config_registry:
            get_io_factory().register_config_io(type(self), ExperimentSeriesConfigIO)

        io_instance = get_io_factory().create_config_io(type(self))
        io_instance.print_config(self)

    def print_experiment_summary(
        self, show_changing_only: bool = True, max_experiments: int = 20
    ):
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
            print(
                f"\nShowing first {max_experiments} experiments (use max_experiments parameter to see more):"
            )
            experiments = experiments[:max_experiments]

        # Find changing parameters
        changing_params = self._identify_changing_parameters(experiments)

        print(
            f"\nChanging parameters: {', '.join(changing_params) if changing_params else 'None'}"
        )
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

    def _identify_changing_parameters(
        self, experiments: List[ExperimentConfig]
    ) -> List[str]:
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
            print(
                f"    Grid: Spherical (nlon={exp.grid.spherical.nlon}, nlat={exp.grid.spherical.nlat})"
            )
        elif exp.grid.random:
            print(f"    Grid: Random (nline_ori={exp.grid.random.nline_ori})")
        elif exp.grid.healpix:
            print(f"    Grid: HEALPix (nside={exp.grid.healpix.nside})")
        elif exp.grid.custom:
            print(f"    Grid: Custom (nline_ori={exp.grid.custom.nline_ori})")

        # Tracing
        print(
            f"    Tracing: max_radius={exp.tracing.max_integration_radius}, memory={exp.tracing.max_memory_mb}MB"
        )

    def load_experiments(self) -> List[ExperimentConfig]:
        """Load experiment configurations from saved config files.

        Returns:
            List of experiment configurations

        Raises:
            ValueError: If configs directory doesn't exist or no configs found
        """
        configs_dir = self.get_configs_dir()
        if not configs_dir.exists():
            error_msg = f"Configs directory does not exist: {configs_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Load all experiment configs (skip series config)
        experiment_files = sorted(
            [
                f
                for f in configs_dir.glob("*.yaml")
                if not f.name.endswith("_config.yaml")
            ]
        )

        if not experiment_files:
            error_msg = f"No experiment configs found in: {configs_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        experiments = []
        for config_file in experiment_files:
            experiment = ExperimentConfig.load(config_file)
            experiments.append(experiment)

        logger.info(f"Loaded {len(experiments)} experiments from {configs_dir}")
        return experiments

    def create_directory_structure(self, delete_existing_configs: bool = False) -> None:
        """Create output directory structure.
        
        Args:
            delete_existing_configs: Whether to delete existing configs directory
        """
        if self.output_dir is None:
            # Create series directory
            series_dir = Path(self.name)
            if not series_dir.exists():
                series_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created output directory structure under: %s", series_dir)

                # Create subdirectories
                for subdir in self.get_subdirectories():
                    subdir_path = series_dir / subdir
                    subdir_path.mkdir(parents=True, exist_ok=True)
            else:
                # Clean up directories if delete_existing_configs is True
                if delete_existing_configs:
                    for subdir in self.get_cleanup_directories():
                        subdir_path = series_dir / subdir
                        if subdir_path.exists():
                            shutil.rmtree(subdir_path)
                            logger.info("Removed existing %s directory: %s", subdir, subdir_path)
                    
                    # Recreate cleaned directories
                    for subdir in self.get_subdirectories():
                        subdir_path = series_dir / subdir
                        subdir_path.mkdir(parents=True, exist_ok=True)
                    logger.info("Recreated output directories for: %s", series_dir)
                else:
                    logger.info("Using existing directory structure: %s", series_dir)
        else:
            # Create series directory
            series_dir = Path(self.output_dir) / self.name
            if not series_dir.exists():
                series_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created output directory structure under: %s", series_dir)

                # Create subdirectories
                for subdir in self.get_subdirectories():
                    subdir_path = series_dir / subdir
                    subdir_path.mkdir(parents=True, exist_ok=True)
            else:
                # Clean up directories if delete_existing_configs is True
                if delete_existing_configs:
                    for subdir in self.get_cleanup_directories():
                        subdir_path = series_dir / subdir
                        if subdir_path.exists():
                            shutil.rmtree(subdir_path)
                            logger.info("Removed existing %s directory: %s", subdir, subdir_path)
                    
                    # Recreate cleaned directories
                    for subdir in self.get_subdirectories():
                        subdir_path = series_dir / subdir
                        subdir_path.mkdir(parents=True, exist_ok=True)
                    logger.info("Recreated output directories for: %s", series_dir)
                else:
                    logger.info("Using existing directory structure: %s", series_dir)

    def _create_experiments(self) -> List[ExperimentConfig]:
        """Create all experiment configurations from factory configs.

        Returns:
            List of experiment configurations
        """
        # Expand all factory configs
        field_configs = expand_field_configs(self.field)
        grid_configs = expand_grid_configs(self.grid)
        tracing_configs = expand_tracing_configs(self.tracing)
        template_configs = expand_template_configs(self.template)

        # Create all combinations
        experiments = []
        experiment_count = -1  # Start from -1 so first increment gives 0

        for field_config in field_configs:
            for grid_config in grid_configs:
                for tracing_config in tracing_configs:
                    for template_config in template_configs:
                        experiment_count += 1

                        # Create metadata for this experiment
                        metadata = create_experiment_metadata(
                            self.name,
                            self.metadata or {},
                            experiment_count,
                            field_config,
                            grid_config,
                            tracing_config,
                            template_config,
                        )

                        # Create experiment name (zero-based numbering)
                        name = f"{self.name}_{experiment_count:03d}"

                        # Create experiment config
                        experiment = ExperimentConfig(
                            name=name,
                            field=field_config,
                            grid=grid_config,
                            tracing=tracing_config,
                            template=template_config,
                            description=self.description,
                            metadata=metadata,
                        )
                        experiments.append(experiment)

        return experiments
