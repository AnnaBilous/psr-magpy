"""Experiment series configuration and generation.

This module provides functionality to create multiple experiments from flexible
parameter inputs, including lists for parameter variation.

Architecture boundary note
--------------------------
This module is in transition toward a domain-focused role. New orchestration,
persistence, and workflow-heavy logic should prefer dedicated modules under
``experiment.services`` and ``experiment.infrastructure``.
"""

from typing import List, Optional, Union, Dict, Any, ClassVar
from pydantic import BaseModel
import yaml
from pathlib import Path
import shutil
import warnings

from .experiment_config import ExperimentConfig
from .field_config import FieldConfig
from .grid_config import GridConfig, SphericalGridConfig
from .tracing_config import TracingConfig
from .factory_configs import (
    FieldFactoryConfig,
    GridFactoryConfig,
    TemplateFactoryConfig,
    TracingFactoryConfig,
)
from .services.series_builder import SeriesBuilder
from .infrastructure.path_policy import (
    series_dir as policy_series_dir,
    configs_dir as policy_configs_dir,
    grids_dir as policy_grids_dir,
    lines_dir as policy_lines_dir,
    logs_dir as policy_logs_dir,
    experiment_config_filename,
    subset_experiment_config_filename,
    results_summary_filename,
)
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError

logger = get_logger(__name__, console_format_brief=True)

_SIDE_EFFECT_DEPRECATION = (
    "ExperimentSeriesConfig model I/O/workflow helpers are deprecated and will "
    "move to services/repositories. Prefer dedicated service APIs."
)


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
    template: Union[FieldLineTemplate, FieldLine, TemplateFactoryConfig]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = "."

    model_config = {"arbitrary_types_allowed": True}

    def model_dump(self, **kwargs):
        """Dump model to dictionary, preserving grid type information.
        
        Parameters
        ----------
        **kwargs
            Additional arguments passed to parent model_dump.
        
        Returns
        -------
        dict
            Model data as dictionary.
        """
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

        Parameters
        ----------
        save_configs : bool, optional
            Whether to save individual experiment configs. Default is True.
        delete_existing_configs : bool, optional
            Whether to delete existing configs. Default is False.
        create_directories : bool, optional
            Whether to create directory structure. Default is True.

        Returns
        -------
        List[ExperimentConfig]
            List of experiment configurations.
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
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
        base_dir = policy_series_dir(output_dir, self.name)
        directories = [self.CONFIGS_DIR, self.GRIDS_DIR, self.LINES_DIR, self.LOGS_DIR]

        for dir_name in directories:
            dir_path = base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

        logger.info(f"Created output directory structure under: {base_dir}")

    def get_configs_dir(self) -> Path:
        """Get path to configs directory.

        Returns
        -------
        Path
            Path to configs directory.
        """
        return policy_configs_dir(self.output_dir, self.name)

    def get_grids_dir(self) -> Path:
        """Get the full path to the grids directory."""
        return policy_grids_dir(self.output_dir, self.name)

    def get_lines_dir(self) -> Path:
        """Get the full path to the lines directory."""
        return policy_lines_dir(self.output_dir, self.name)

    def get_logs_dir(self) -> Path:
        """Get the full path to the logs directory."""
        return policy_logs_dir(self.output_dir, self.name)

    def verify_directories(
        self, create_if_missing: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Verify existence of standard directories and optionally create them.

        Parameters
        ----------
        create_if_missing : bool, optional
            If True, create directories that don't exist. Default is False.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with status for each directory type. Keys are 'configs',
            'grids', 'lines', 'logs'. Each value is a dict with 'exists' (bool),
            'path' (Path), and 'created' (bool) keys.
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

        Parameters
        ----------
        experiment_number : int
            The experiment number (0-based).

        Returns
        -------
        Path
            Path to the experiment config file.
        """
        # Try regular config first, then subset config
        regular_path = self.get_configs_dir() / experiment_config_filename(
            self.name, experiment_number
        )
        subset_path = self.get_configs_dir() / subset_experiment_config_filename(
            self.name, experiment_number
        )
        
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

        Parameters
        ----------
        index : int
            Experiment index (0-based).
        experiments : Optional[List[ExperimentConfig]], optional
            Optional list of experiments. If None, creates series. Default is None.

        Returns
        -------
        Optional[ExperimentConfig]
            ExperimentConfig if index is valid, None otherwise.
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

        Parameters
        ----------
        experiment_number : int
            The experiment number (0-based).

        Returns
        -------
        ExperimentConfig
            Loaded experiment configuration.
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        from psr_magpy.experiment.infrastructure.config_repository import ConfigRepository

        repository = ConfigRepository()
        return repository.load_experiment_config(
            self, experiment_number, prefer_subset=True
        )

    def get_total_experiments_from_summary(self) -> int:
        """Get the total number of experiments from the summary YAML file.

        Returns
        -------
        int
            Total number of experiments from summary.

        Raises
        ------
        FileNotFoundError
            If summary.yaml does not exist.
            KeyError: If 'total_experiments' key is missing from summary
        """
        summary_path = policy_series_dir(self.output_dir, self.name) / results_summary_filename(
            self.name
        )
        
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            summary = yaml.safe_load(f)
        
        if 'total_experiments' not in summary:
            raise KeyError(f"Key 'total_experiments' not found in summary file: {summary_path}")
        
        return summary['total_experiments']

    def save(self, filename: Optional[str] = None) -> None:
        """Save configuration to YAML file in the configs directory.

        Parameters
        ----------
        filename : Optional[str], optional
            Optional filename. If not provided, uses "{series_name}_config.yaml". Default is None.
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        from psr_magpy.io import save_config
        from psr_magpy.experiment.infrastructure.config_repository import ConfigRepository

        if filename is None:
            ConfigRepository().save_series_config(self, overwrite=True)
            return

        save_config(self, filename, overwrite=True)

    @classmethod
    def load(cls, config_file: Union[str, Path]) -> "ExperimentSeriesConfig":
        """Load configuration from YAML file.

        Parameters
        ----------
        config_file : Union[str, Path]
            Path to configuration file.

        Returns
        -------
        ExperimentSeriesConfig
            Loaded configuration.
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        from psr_magpy.io import load_config

        return load_config(cls, config_file)

    def print_yaml(self):
        """Print configuration as pretty YAML to screen."""
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        from psr_magpy.io.config_io import ExperimentSeriesConfigIO
        from psr_magpy.io.validator import ValidationLevel, RecoveryStrategy
        from psr_magpy.io.validator import ConfigIOValidator
        from psr_magpy.io.memory_monitor import MemoryMonitor, MemoryThreshold

        validator = ConfigIOValidator(
            validation_level=ValidationLevel.STANDARD,
            recovery_strategy=RecoveryStrategy.RETRY_ONCE,
        )
        memory_monitor = MemoryMonitor(
            threshold=MemoryThreshold.MODERATE,
            safety_margin_mb=100.0,
            enable_checks=True,
            enable_warnings=True,
        )
        io_instance = ExperimentSeriesConfigIO(
            validator=validator,
            memory_monitor=memory_monitor,
        )
        io_instance.print_config(self)

    def print_experiment_summary(
        self, show_changing_only: bool = True, max_experiments: int = 20
    ):
        """Log a summary of experiments with changing parameters highlighted.

        Parameters
        ----------
        show_changing_only : bool, optional
            If True, only show parameters that vary between experiments. Default is True.
        max_experiments : int, optional
            Maximum number of experiments to display (to avoid overwhelming output). Default is 20.
        """
        experiments = self.create_experiment_series()

        if len(experiments) == 0:
            logger.info("No experiments created.")
            return

        logger.info(f"\n=== Experiment Series Summary: {self.name} ===")
        logger.info(f"Total experiments: {len(experiments)}")
        logger.info(f"Description: {self.description or 'No description'}")

        if len(experiments) > max_experiments:
            logger.info(
                f"\nShowing first {max_experiments} experiments (use max_experiments parameter to see more):"
            )
            experiments = experiments[:max_experiments]

        # Find changing parameters
        changing_params = self._identify_changing_parameters(experiments)

        logger.info(
            f"\nChanging parameters: {', '.join(changing_params) if changing_params else 'None'}"
        )
        logger.info("\nExperiments:")

        for i, exp in enumerate(experiments, 1):
            logger.info(f"\n  Experiment {i}: {exp.name}")

            if show_changing_only and changing_params:
                # Show only changing parameters
                for param in changing_params:
                    value = self._get_parameter_value(exp, param)
                    if value is not None:
                        logger.info(f"    {param}: {value}")
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
        """Log all key parameters for an experiment."""
        # Field components
        logger.info(f"    Field components: {len(exp.field.components)}")
        for i, comp in enumerate(exp.field.components):
            logger.info(f"      Component {i+1}: {comp.moment_type}")
            logger.info(f"        B_magnitude: {comp.B_magnitude}")
            logger.info(f"        lat: {comp.moment_orientation_latitude_degree}°")
            logger.info(f"        lon: {comp.moment_orientation_longitude_degree}°")

        # Grid
        if exp.grid.spherical:
            logger.info(
                f"    Grid: Spherical (nlon={exp.grid.spherical.nlon}, nlat={exp.grid.spherical.nlat})"
            )
        elif exp.grid.random:
            logger.info(f"    Grid: Random (nline_ori={exp.grid.random.nline_ori})")
        elif exp.grid.healpix:
            logger.info(f"    Grid: HEALPix (nside={exp.grid.healpix.nside})")
        elif exp.grid.custom:
            logger.info(f"    Grid: Custom (nline_ori={exp.grid.custom.nline_ori})")

        # Tracing
        logger.info(
            f"    Tracing: max_radius={exp.tracing.max_integration_radius}, memory={exp.tracing.max_memory_mb}MB"
        )

    def load_experiments(self) -> List[ExperimentConfig]:
        """Load experiment configurations from saved config files.

        If series metadata indicates subset grids (has_subset_grids=True),
        prioritizes loading subset config files (``*_subset.yaml``).

        Returns
        -------
        List[ExperimentConfig]
            List of experiment configurations.

        Raises
        ------
        ValueError
            If configs directory doesn't exist or no configs found.
        PSRMagPyIOError
            If subset configs are expected but not found.
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        from psr_magpy.experiment.infrastructure.config_repository import ConfigRepository

        configs_dir = self.get_configs_dir()
        if not configs_dir.exists():
            raise PSRMagPyIOError(f"Configs directory does not exist: {configs_dir}")

        repository = ConfigRepository()

        # Check if subset grids are expected
        has_subset_grids = (
            self.metadata is not None
            and self.metadata.get("has_subset_grids", False)
        )

        # If subset grids are expected, use only subset configs
        if has_subset_grids:
            subset_config_files = repository.list_experiment_config_files(
                self, prefer_subset=True
            )
            subset_config_files = [
                f for f in subset_config_files if f.name.endswith("_subset.yaml")
            ]
            if not subset_config_files:
                error_msg = (
                    f"Series metadata indicates subset grids should exist, "
                    f"but no subset config files (*_subset.yaml) found in: {configs_dir}\n"
                    f"Please run the workflow to create subset grids first."
                )
                logger.error(error_msg)
                raise PSRMagPyIOError(error_msg)
            experiment_files = sorted(subset_config_files)
            logger.info(f"Loading {len(subset_config_files)} subset config files (has_subset_grids=True)")
        else:
            # Use regular configs if available, otherwise use subset configs as fallback
            regular_config_files = repository.list_experiment_config_files(
                self, prefer_subset=False
            )
            regular_config_files = [
                f for f in regular_config_files if not f.name.endswith("_subset.yaml")
            ]
            if regular_config_files:
                experiment_files = sorted(regular_config_files)
                logger.info(f"Loading {len(regular_config_files)} regular config files")
            else:
                raise PSRMagPyIOError(f"No experiment configs found in: {configs_dir}")

        experiments = []
        for config_file in experiment_files:
            experiment = repository.load_experiment_config(
                self, self._parse_experiment_number_from_filename(config_file.name),
                prefer_subset=config_file.name.endswith("_subset.yaml")
            )
            experiments.append(experiment)

        logger.info(f"Loaded {len(experiments)} experiments from {configs_dir}")
        return experiments

    @staticmethod
    def _parse_experiment_number_from_filename(filename: str) -> int:
        """Parse experiment number from config filename."""
        base = filename[:-5] if filename.endswith(".yaml") else filename
        if base.endswith("_subset"):
            base = base[: -len("_subset")]
        return int(base.rsplit("_", 1)[1])

    def create_directory_structure(self, delete_existing_configs: bool = False) -> None:
        """Create output directory structure.

        Parameters
        ----------
        delete_existing_configs : bool, optional
            Whether to delete existing configs directory. Default is False.
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        series_dir = (
            Path(self.output_dir) / self.name
            if self.output_dir is not None
            else Path(self.name)
        )
        series_dir.mkdir(parents=True, exist_ok=True)

        if delete_existing_configs and series_dir.exists():
            for subdir in self.get_cleanup_directories():
                subdir_path = series_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
                    logger.info(
                        "Removed existing %s directory: %s", subdir, subdir_path
                    )
            logger.info("Removed cleanup directories for: %s", series_dir)

        for subdir in self.get_subdirectories():
            (series_dir / subdir).mkdir(parents=True, exist_ok=True)

        if delete_existing_configs:
            logger.info("Recreated output directories for: %s", series_dir)
        else:
            logger.info("Using existing directory structure: %s", series_dir)

    def _create_experiments(self) -> List[ExperimentConfig]:
        """Create all experiment configurations from factory configs.

        Returns
        -------
        List[ExperimentConfig]
            List of experiment configurations.
        """
        return SeriesBuilder().build_experiments(
            series_name=self.name,
            field=self.field,
            grid=self.grid,
            tracing=self.tracing,
            template=self.template,
            description=self.description,
            metadata=self.metadata,
        )

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

        Parameters
        ----------
        coarse_series_config : ExperimentSeriesConfig
            Coarse resolution series config.
        R_LC : float
            Light cylinder radius for OFL calculation.
        h_LC_coeff : float, optional
            Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
        experiment_numbers : Optional[List[int]], optional
            Optional list of experiment numbers to process. If None, processes all experiments. Default is None.
        save_configs : bool, optional
            If True, save updated experiment configs. Default is True.

        Returns
        -------
        ExperimentSeriesConfig
            Updated fine series config with subset grids.

        Raises
        ------
        ValueError
            If directories don't exist or experiments can't be processed.
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        from psr_magpy.experiment.services.subset_workflow_service import (
            SubsetWorkflowService,
        )

        return SubsetWorkflowService().create_subset_series_from_ofl(
            fine_series_config=self,
            coarse_series_config=coarse_series_config,
            R_LC=R_LC,
            h_LC_coeff=h_LC_coeff,
            experiment_numbers=experiment_numbers,
            save_configs=save_configs,
        )

    def infer_experiment_count(self) -> int:
        """Infer ``N`` such that experiment indices are ``0 .. N-1`` (summary or configs on disk)."""
        from psr_magpy.experiment.services.experiment_lookup_service import (
            infer_experiment_count,
        )

        return infer_experiment_count(self)

    def find_experiment_numbers_by_thD_fQ(
        self,
        thD: float,
        fQ: float,
        *,
        n_experiments: Optional[int] = None,
    ) -> List[int]:
        """Return experiment indices matching dipole colatitude ``thD`` and quadrupole ``fQ``."""
        from psr_magpy.experiment.services.experiment_lookup_service import (
            find_experiment_numbers_by_thD_fQ,
        )

        return find_experiment_numbers_by_thD_fQ(
            self,
            thD,
            fQ,
            n_experiments=n_experiments,
        )


def create_simple_series(
    name: str,
    field: Union[FieldConfig, FieldFactoryConfig],
    grid: Union[GridConfig, GridFactoryConfig],
    tracing: Union[TracingConfig, TracingFactoryConfig],
    template: Union[FieldLineTemplate, FieldLine, TemplateFactoryConfig],
    output_dir: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExperimentSeriesConfig:
    """Create a simple experiment series configuration.

    Convenience wrapper for common single-config experiment series creation.
    This function simplifies the creation of ExperimentSeriesConfig when there is 
    no need for complex parameter variations.

    Parameters
    ----------
    name : str
        Series name.
    field : Union[FieldConfig, FieldFactoryConfig]
        Field configuration (single or factory).
    grid : Union[GridConfig, GridFactoryConfig]
        Grid configuration (single or factory).
    tracing : Union[TracingConfig, TracingFactoryConfig]
        Tracing configuration (single or factory).
    template : Union[FieldLine, TemplateFactoryConfig]
        Field line template (single or factory).
    output_dir : Optional[str], optional
        Optional output directory. Defaults to current directory.
    description : Optional[str], optional
        Optional series description. Default is None.
    metadata : Optional[Dict[str, Any]], optional
        Optional metadata dictionary. Default is None.

    Returns
    -------
    ExperimentSeriesConfig
        ExperimentSeriesConfig instance.

    Examples
    --------
    >>> field = FieldConfig(components=[FieldComponentConfig(moment_type="dipole")])
    >>> grid = GridConfig(healpix=HealpixGridConfig(nside=1))
    >>> tracing = TracingConfig()
    >>> template = FieldLineTemplate.template_OFL_calc()
    >>> series = create_simple_series("dipole_series", field, grid, tracing, template)
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
