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
from psr_magpy.utils.exceptions import PSRMagPyExperimentError, PSRMagPyIOError

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
        if "grid" in data and hasattr(self.grid, "model_dump"):
            data["grid"] = self.grid.model_dump(**kwargs)

        return data

    def create_experiment_series(
        self,
        save_configs: bool = True,
        delete_existing_configs: bool = False,
        create_directories: bool = True,
    ) -> List[ExperimentConfig]:
        """Create experiment series from configuration.

        Args:
            save_configs: Whether to save individual experiment configs
            delete_existing_configs: Whether to delete existing configs

        Returns:
            List of experiment configurations
        """
        logger.info("Creating experiment series: %s", self.name)

        # Create output directory structure if requested
        if create_directories:
            self.create_directory_structure(delete_existing_configs)
        else:
            logger.info("Skipping directory creation (create_directories=False)")

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
                    logger.debug(
                        f"Updated RandomGrid config with file path: {grid_path}"
                    )
                elif grid_type == "custom" and experiment.grid.custom:
                    experiment.grid.custom.grid_file_path = str(grid_path)
                    logger.debug(
                        f"Updated CustomGrid config with file path: {grid_path}"
                    )
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

    def verify_directories(
        self, create_if_missing: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Verify existence of standard directories and optionally create them.

        Args:
            create_if_missing: If True, create directories that don't exist

        Returns:
            Dictionary with status for each directory type:
            {
                'configs': {'exists': bool, 'path': Path, 'created': bool},
                'grids': {'exists': bool, 'path': Path, 'created': bool},
                'lines': {'exists': bool, 'path': Path, 'created': bool},
                'logs': {'exists': bool, 'path': Path, 'created': bool}
            }
        """
        status = {}
        directories = {
            "configs": self.get_configs_dir(),
            "grids": self.get_grids_dir(),
            "lines": self.get_lines_dir(),
            "logs": self.get_logs_dir(),
        }

        for dir_name, dir_path in directories.items():
            exists = dir_path.exists()
            created = False

            if not exists and create_if_missing:
                dir_path.mkdir(parents=True, exist_ok=True)
                created = True
                logger.debug(f"Created directory: {dir_path}")

            status[dir_name] = {
                "exists": exists or created,
                "path": dir_path,
                "created": created,
            }

        return status

    def get_subdirectories(self) -> List[str]:
        """Get list of all subdirectory names."""
        return [self.LINES_DIR, self.GRIDS_DIR, self.CONFIGS_DIR, self.LOGS_DIR]

    def get_cleanup_directories(self) -> List[str]:
        """Get list of directories to clean up (only configs)."""
        return [self.CONFIGS_DIR]

    def get_experiment_config_path(self, experiment_number: int) -> Path:
        """Get the full path to a specific experiment config file.

        Args:
            experiment_number: The experiment number (0-based)

        Returns:
            Path to the experiment config file
        """
        # Try regular config first, then subset config
        regular_path = self.get_configs_dir() / f"{self.name}_{experiment_number:03d}.yaml"
        subset_path = self.get_configs_dir() / f"{self.name}_{experiment_number:03d}_subset.yaml"
        
        if regular_path.exists():
            return regular_path
        elif subset_path.exists():
            return subset_path
        else:
            # If neither exists, return the regular path (will raise error when loading)
            return regular_path

    def get_experiment_by_index(
        self, index: int, experiments: Optional[List["ExperimentConfig"]] = None
    ) -> Optional["ExperimentConfig"]:
        """Safely get experiment by index with validation.

        Args:
            index: Experiment index (0-based)
            experiments: Optional list of experiments. If None, creates series.

        Returns:
            ExperimentConfig if index is valid, None otherwise
        """
        if experiments is None:
            experiments = self.create_experiment_series(save_configs=False)

        if index < 0 or index >= len(experiments):
            logger.warning(
                f"Experiment index {index} out of range (0-{len(experiments)-1})"
            )
            return None

        return experiments[index]

    def load_experiment_config(self, experiment_number: int) -> "ExperimentConfig":
        """Load a specific experiment config by number.

        Args:
            experiment_number: The experiment number (0-based)

        Returns:
            Loaded experiment configuration
        """
        from psr_magpy.experiment.experiment_config import ExperimentConfig

        config_path = self.get_experiment_config_path(experiment_number)
        return ExperimentConfig.load(config_path)

    def get_total_experiments_from_summary(self) -> int:
        """Get the total number of experiments from the summary YAML file.

        Returns:
            Total number of experiments from summary

        Raises:
            FileNotFoundError: If summary.yaml does not exist
            KeyError: If 'total_experiments' key is missing from summary
        """
        output_dir = self.output_dir or "."
        summary_path = Path(output_dir) / self.name / f"{self.name}_results_summary.yaml"
        
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            summary = yaml.safe_load(f)
        
        if 'total_experiments' not in summary:
            raise KeyError(f"Key 'total_experiments' not found in summary file: {summary_path}")
        
        return summary['total_experiments']

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
            raise PSRMagPyIOError(f"Configs directory does not exist: {configs_dir}")

        # Load all experiment configs (skip series config)
        experiment_files = sorted(
            [
                f
                for f in configs_dir.glob("*.yaml")
                if not f.name.endswith("_config.yaml")
            ]
        )

        if not experiment_files:
            raise PSRMagPyIOError(f"No experiment configs found in: {configs_dir}")

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
                            logger.info(
                                "Removed existing %s directory: %s", subdir, subdir_path
                            )

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
                            logger.info(
                                "Removed existing %s directory: %s", subdir, subdir_path
                            )

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

    def create_subset_series_from_OFL(
        self,
        coarse_series_config: "ExperimentSeriesConfig",
        R_LC: float,
        h_LC_coeff: float = 3**1.5 / 2.0,
        experiment_numbers: Optional[List[int]] = None,
        save_configs: bool = True,
    ) -> "ExperimentSeriesConfig":
        """Create subset series from coarse experiment OFL footpoints.

        Processes all experiments in the coarse series, extracts OFL footpoints,
        creates fine resolution subset grids, and updates fine experiment configs.

        Args:
            coarse_series_config: Coarse resolution series config
            R_LC: Light cylinder radius for OFL calculation
            h_LC_coeff: Height coefficient for OFL calculation
            experiment_numbers: Optional list of experiment numbers to process.
                               If None, processes all experiments.
            save_configs: If True, save updated experiment configs

        Returns:
            Updated fine series config with subset grids

        Raises:
            ValueError: If directories don't exist or experiments can't be processed
        """
        from psr_magpy.grid.grid import create_fine_resolution_subset_from_coarse_OFL
        from psr_magpy.experiment.experiment_config import GridConfig, HealpixGridConfig

        # Verify coarse directories
        coarse_status = coarse_series_config.verify_directories(create_if_missing=False)
        if not coarse_status["configs"]["exists"] or not coarse_status["lines"]["exists"]:
            raise PSRMagPyExperimentError("Coarse series directories (configs, lines) must exist")

        # Verify/create fine directories
        fine_status = self.verify_directories(create_if_missing=True)

        # Create fine experiments
        fine_experiments = self.create_experiment_series(save_configs=False)

        # Get experiment numbers to process
        if experiment_numbers is None:
            coarse_experiments = coarse_series_config.create_experiment_series(
                save_configs=False
            )
            experiment_numbers = list(range(len(coarse_experiments)))

        logger.info(
            f"Processing {len(experiment_numbers)} experiments for OFL subset creation"
        )

        successful_count = 0
        failed_count = 0

        for exp_num in experiment_numbers:
            logger.info(f"Processing experiment {exp_num}...")

            try:
                # Load coarse experiment
                coarse_experiment = coarse_series_config.load_experiment_config(exp_num)
                logger.debug(f"Loaded coarse experiment: {coarse_experiment.name}")

                # Get corresponding fine experiment
                fine_experiment = self.get_experiment_by_index(exp_num, fine_experiments)
                if fine_experiment is None:
                    logger.warning(
                        f"No corresponding fine experiment found for {exp_num}, skipping..."
                    )
                    failed_count += 1
                    continue

                logger.debug(f"Using fine experiment: {fine_experiment.name}")

                # Create subset grid using helper function
                coarse_lines_dir = coarse_series_config.get_lines_dir()
                fine_grids_dir = self.get_grids_dir()

                grid_filepath, updated_grid_config = (
                    create_fine_resolution_subset_from_coarse_OFL(
                        coarse_experiment=coarse_experiment,
                        fine_experiment=fine_experiment,
                        R_LC=R_LC,
                        coarse_lines_dir=coarse_lines_dir,
                        fine_grids_dir=fine_grids_dir,
                        h_LC_coeff=h_LC_coeff,
                    )
                )

                # Update fine experiment config
                fine_experiment.grid = updated_grid_config

                # Add subset metadata
                subset_metadata = {
                    "coarse_experiment": coarse_experiment.name,
                    "subset_grid_type": "healpix",
                    "is_subset_grid": True,
                    "subset_grid_file": grid_filepath.name,
                    "R_LC": R_LC,
                    "h_LC_coeff": h_LC_coeff,
                }

                if fine_experiment.metadata is None:
                    fine_experiment.metadata = {}
                fine_experiment.metadata.update(subset_metadata)

                # Save subset config
                if save_configs:
                    fine_config_file = (
                        self.get_configs_dir() / f"{fine_experiment.name}_subset.yaml"
                    )
                    fine_experiment.save(str(fine_config_file), overwrite=True)
                    logger.info(f"Saved subset experiment config: {fine_config_file}")

                successful_count += 1

            except Exception as e:
                error_msg = f"Failed to process experiment {exp_num}: {e}"
                logger.error(error_msg)
                failed_count += 1
                continue

        logger.info(
            f"OFL subset creation completed: {successful_count} successful, "
            f"{failed_count} failed"
        )

        # Update series metadata
        if self.metadata is None:
            self.metadata = {}
        self.metadata.update(
            {
                "original_series": self.name,
                "subset_experiments": experiment_numbers,
                "subset_grid_type": "healpix",
                "has_subset_grids": True,
                "coarse_series": coarse_series_config.name,
                "R_LC": R_LC,
                "h_LC_coeff": h_LC_coeff,
            }
        )

        return self


def create_simple_series(
    name: str,
    field: Union[FieldConfig, FieldFactoryConfig],
    grid: Union[GridConfig, GridFactoryConfig],
    tracing: Union[TracingConfig, TracingFactoryConfig],
    template: Union[FieldLine, TemplateFactoryConfig],
    output_dir: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExperimentSeriesConfig:
    """Create a simple experiment series configuration.

    Convenience wrapper for common single-config experiment series creation.
    This function simplifies the creation of ExperimentSeriesConfig when you
    don't need complex parameter variations.

    Args:
        name: Series name
        field: Field configuration (single or factory)
        grid: Grid configuration (single or factory)
        tracing: Tracing configuration (single or factory)
        template: Field line template (single or factory)
        output_dir: Optional output directory (defaults to current directory)
        description: Optional series description
        metadata: Optional metadata dictionary

    Returns:
        ExperimentSeriesConfig instance

    Example:
        >>> field = FieldConfig(components=[...])
        >>> grid = GridConfig(healpix=HealpixGridConfig(nside=128))
        >>> tracing = TracingConfig()
        >>> template = FieldLine.template_all()
        >>> series = create_simple_series("my_series", field, grid, tracing, template)
    """
    return ExperimentSeriesConfig(
        name=name,
        field=field,
        grid=grid,
        tracing=tracing,
        template=template,
        output_dir=output_dir,
        description=description,
        metadata=metadata,
    )
