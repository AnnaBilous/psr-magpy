import yaml
from dataclasses import dataclass, asdict
from typing import (
    List,
    Optional,
    Union,
    Dict,
    Any,
    Literal,
    TypedDict,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from astropy.coordinates import CartesianRepresentation


from pathlib import Path
import numpy as np

from psr_magpy.field.components import (
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
    MultipoleType,
)
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.grid.grid import (
    SphericalGrid,
    RandomGrid,
    HealpixGrid,
    CustomGrid,
    FootpointGrid,
    GridType,
)
from psr_magpy.line.tracing_parameters import (
    TracingParameters,
    IntegrationParameters,
    OutputParameters,
    PerformanceParameters,
)
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.line.tracing_parameters import TracingDefaults

logger = get_logger(__name__, console_format_brief=True)


# Type definitions for better type safety


from pydantic import BaseModel, Field


class FieldComponentConfig(BaseModel):
    """Configuration for a single field component."""

    moment_type: str
    B_magnitude: float = 1.0
    moment_orientation_latitude_degree: float = 0.0
    moment_orientation_longitude_degree: float = 0.0
    offset_r_ns: List[float] = [0.0, 0.0, 0.0]


class FieldConfig(BaseModel):
    """Configuration for magnetic field components."""

    components: List[FieldComponentConfig] = Field(
        description="List of field components"
    )

    def create_field(self) -> MultipoleField:
        """Create MultipoleField from configuration."""
        logger.debug(f"Creating field with {len(self.components)} components")

        field_components = []
        for i, comp in enumerate(self.components):
            comp_type = comp.moment_type
            logger.debug(f"Creating {comp_type} component {i+1}")

            if comp_type == MultipoleType.DIPOLE.value:
                component = DipoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns,
                )
            elif comp_type == MultipoleType.QUADRUPOLE.value:
                component = QuadrupoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns,
                )
            else:
                error_msg = f"Unknown field component type: {comp_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            field_components.append(component)

        return MultipoleField(field_components)

    def get_complete_components(self) -> List[Dict[str, Any]]:
        """Get complete component configurations including all default parameters."""
        logger.debug("Getting complete component configurations")

        complete_components = []
        for comp in self.components:
            comp_type = comp.moment_type

            # Create component to get all parameters (including defaults)
            if comp_type == MultipoleType.DIPOLE.value:
                component = DipoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns,
                )
            elif comp_type == MultipoleType.QUADRUPOLE.value:
                component = QuadrupoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns,
                )
            else:
                error_msg = f"Unknown field component type: {comp_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Get complete parameters including defaults
            complete_params = component.params.model_dump()
            # Convert enum to string for JSON serialization
            complete_params["moment_type"] = complete_params["moment_type"].value

            complete_components.append(complete_params)
            logger.debug(f"Complete params for {comp_type}: {complete_params}")

        return complete_components


class SphericalGridConfig(BaseModel):
    """Configuration for spherical grid."""

    nlon: int
    nlat: int
    pole_lat: float = 0.0
    subset: bool = Field(default=False, description="Whether this is a subset grid")
    grid_file_path: Optional[str] = Field(
        default=None, description="Path to saved grid file (required for subset grids)"
    )


class RandomGridConfig(BaseModel):
    """Configuration for random grid."""

    nline_ori: int
    grid_file_path: Optional[str] = Field(
        default=None,
        description="Path to saved grid file (required when saving results to disk)",
    )


class HealpixGridConfig(BaseModel):
    """Configuration for HEALPix grid."""

    nside: int
    subset: bool = Field(default=False, description="Whether this is a subset grid")
    grid_file_path: Optional[str] = Field(
        default=None, description="Path to saved grid file (required for subset grids)"
    )


class CustomGridConfig(BaseModel):
    """Configuration for custom grid."""

    nline_ori: int
    coordinates: Optional[Any] = None  # Accept any type, validate in create_grid
    grid_file_path: Optional[str] = Field(
        default=None,
        description="Path to saved grid file (required when saving results to disk)",
    )

    model_config = {"arbitrary_types_allowed": True}


class GridConfig(BaseModel):
    """Configuration for field line grid."""

    spherical: Optional[SphericalGridConfig] = Field(
        default=None, description="Spherical grid parameters"
    )
    random: Optional[RandomGridConfig] = Field(
        default=None, description="Random grid parameters"
    )
    healpix: Optional[HealpixGridConfig] = Field(
        default=None, description="HEALPix grid parameters"
    )
    custom: Optional[CustomGridConfig] = Field(
        default=None, description="Custom grid parameters"
    )

    def model_post_init(self, __context) -> None:
        """Validate that exactly one grid type is specified."""
        configs = [self.spherical, self.random, self.healpix, self.custom]
        non_none_configs = [c for c in configs if c is not None]

        # Skip validation for temporary objects (all fields are None or default)
        if len(non_none_configs) == 0:
            # This might be a temporary object during I/O operations
            # Only log a warning instead of raising an error
            logger.warning(
                "GridConfig created with no grid type specified (likely temporary object)"
            )
            return

        if len(non_none_configs) != 1:
            error_msg = (
                f"Exactly one grid type must be specified, got {len(non_none_configs)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_grid_type(self) -> str:
        """Get the grid type from the configuration."""
        if self.spherical is not None:
            return GridType.SPHERICAL.value
        elif self.random is not None:
            return GridType.RANDOM.value
        elif self.healpix is not None:
            return GridType.HEALPIX.value
        elif self.custom is not None:
            return GridType.CUSTOM.value
        else:
            error_msg = "No grid type specified"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def create_grid(self) -> Optional[FootpointGrid]:
        """Create grid from config."""
        if self.random is not None:
            config = self.random
            if config.grid_file_path:
                logger.info(f"Loading RandomGrid from {config.grid_file_path}")
                return RandomGrid.load_from_fits(config.grid_file_path)
            else:
                logger.info("Creating RandomGrid from scratch")
                params = config.model_dump()
                params.pop(
                    "grid_file_path", None
                )  # Remove file path when creating from scratch
                return RandomGrid(**params)
        elif self.custom is not None:
            config = self.custom
            if config.grid_file_path:
                logger.info(f"Loading CustomGrid from {config.grid_file_path}")
                return CustomGrid.load_from_fits(config.grid_file_path)
            else:
                logger.info("Creating CustomGrid from scratch")
                params = config.model_dump()
                params.pop(
                    "grid_file_path", None
                )  # Remove file path when creating from scratch

                # Handle different coordinate formats
                if params["coordinates"] is not None:
                    from astropy.coordinates import CartesianRepresentation
                    import numpy as np

                    if isinstance(params["coordinates"], list):
                        # Convert list coordinates to CartesianRepresentation
                        params["coordinates"] = CartesianRepresentation(
                            x=np.array(params["coordinates"][0]),
                            y=np.array(params["coordinates"][1]),
                            z=np.array(params["coordinates"][2]),
                        )
                    elif not isinstance(params["coordinates"], CartesianRepresentation):
                        error_msg = f"Coordinates must be list of lists or CartesianRepresentation, got {type(params['coordinates'])}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                return CustomGrid(**params)
        elif self.spherical is not None:
            config = self.spherical
            if config.subset and config.grid_file_path:
                logger.info(
                    f"Loading subset SphericalGrid from {config.grid_file_path}"
                )
                return SphericalGrid.load_from_fits(config.grid_file_path)
            else:
                logger.info("Creating SphericalGrid from scratch")
                params = config.model_dump()
                params.pop(
                    "grid_file_path", None
                )  # Remove file path when creating from scratch
                params.pop(
                    "subset", None
                )  # Remove subset flag when creating from scratch
                return SphericalGrid(**params)
        elif self.healpix is not None:
            config = self.healpix
            if config.subset and config.grid_file_path:
                logger.info(f"Loading subset HealpixGrid from {config.grid_file_path}")
                return HealpixGrid.load_from_fits(config.grid_file_path)
            else:
                logger.info("Creating HealpixGrid from scratch")
                params = config.model_dump()
                params.pop(
                    "grid_file_path", None
                )  # Remove file path when creating from scratch
                params.pop(
                    "subset", None
                )  # Remove subset flag when creating from scratch
                return HealpixGrid(**params)
        else:
            error_msg = "No grid type specified"
            logger.error(error_msg)
            raise ValueError(error_msg)


class TracingConfig(BaseModel):
    """Configuration for field line tracing parameters."""

    # Integration parameters
    max_integration_radius: float = TracingDefaults.MAX_INTEGRATION_RADIUS
    threshold_line_distance: float = TracingDefaults.THRESHOLD_LINE_DISTANCE
    t_span: Tuple[float, float] = TracingDefaults.T_SPAN
    rtol: float = TracingDefaults.RTOL
    atol: float = TracingDefaults.ATOL
    surface_radius: float = TracingDefaults.SURFACE_RADIUS
    dt: float = TracingDefaults.DT
    max_step: float = TracingDefaults.MAX_STEP
    method: str = TracingDefaults.METHOD
    trace_open_lines_backwards: bool = TracingDefaults.TRACE_OPEN_LINES_BACKWARDS
    verify_by_reversing: bool = TracingDefaults.VERIFY_BY_REVERSING
    reversed_error_threshold: float = TracingDefaults.REVERSED_ERROR_THRESHOLD

    # Performance parameters
    max_memory_mb: float = TracingDefaults.MAX_MEMORY_MB
    safety_factor: float = TracingDefaults.SAFETY_FACTOR
    max_batch_size: int = TracingDefaults.MAX_BATCH_SIZE
    batch_size_safety_factor: int = TracingDefaults.BATCH_SIZE_SAFETY_FACTOR
    check_interval_sec: float = TracingDefaults.CHECK_INTERVAL_SEC
    n_cores: Optional[int] = TracingDefaults.N_CORES
    n_trial: int = TracingDefaults.N_TRIAL
    progress_bar: bool = TracingDefaults.PROGRESS_BAR
    verbose: bool = TracingDefaults.VERBOSE
    skip_memory_estimation: bool = TracingDefaults.SKIP_MEMORY_ESTIMATION

    # Output parameters
    save_results_to_disk: bool = TracingDefaults.SAVE_RESULTS_TO_DISK
    keep_results_in_memory: bool = TracingDefaults.KEEP_RESULTS_IN_MEMORY
    filename: Optional[str] = TracingDefaults.FILENAME
    output_format: str = TracingDefaults.OUTPUT_FORMAT
    overwrite: bool = TracingDefaults.OVERWRITE

    def create_tracing_parameters(self) -> TracingParameters:
        """Create TracingParameters from configuration."""
        logger.debug("Creating tracing parameters from configuration")

        integration = IntegrationParameters(
            max_integration_radius=self.max_integration_radius,
            threshold_line_distance=self.threshold_line_distance,
            t_span=self.t_span,
            rtol=self.rtol,
            atol=self.atol,
            surface_radius=self.surface_radius,
            dt=self.dt,
            max_step=self.max_step,
            method=self.method,
            trace_open_lines_backwards=self.trace_open_lines_backwards,
            verify_by_reversing=self.verify_by_reversing,
            reversed_error_threshold=self.reversed_error_threshold,
        )

        performance = PerformanceParameters(
            max_memory_mb=self.max_memory_mb,
            safety_factor=self.safety_factor,
            max_batch_size=self.max_batch_size,
            batch_size_safety_factor=self.batch_size_safety_factor,
            check_interval_sec=self.check_interval_sec,
            n_cores=self.n_cores,
            n_trial=self.n_trial,
            progress_bar=self.progress_bar,
            verbose=self.verbose,
            skip_memory_estimation=self.skip_memory_estimation,
        )

        output = OutputParameters(
            save_results_to_disk=self.save_results_to_disk,
            keep_results_in_memory=self.keep_results_in_memory,
            filename=self.filename,
            output_format=self.output_format,
            overwrite=self.overwrite,
        )

        return TracingParameters(
            integration=integration, performance=performance, output=output
        )

    def get_complete_config(self) -> Dict[str, Any]:
        """Get complete tracing configuration including all default parameters."""
        logger.debug("Getting complete tracing configuration")

        # Return flat configuration for backward compatibility
        return self.model_dump()


class FieldFactoryConfig(BaseModel):
    """Factory configuration for field components with flexible parameter inputs."""

    components: List[Union[FieldComponentConfig, List[FieldComponentConfig]]]


class GridFactoryConfig(BaseModel):
    """Factory configuration for grids with flexible parameter inputs."""

    grids: List[
        Union[
            SphericalGridConfig, RandomGridConfig, HealpixGridConfig, CustomGridConfig
        ]
    ]


class TemplateFactoryConfig(BaseModel):
    """Factory configuration for template field lines."""

    templates: List[FieldLine]

    model_config = {"arbitrary_types_allowed": True}


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str
    field: FieldConfig
    grid: GridConfig
    tracing: TracingConfig
    template: FieldLine
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration."""
        logger.debug(f"Initializing experiment: {self.name}")
        if self.description:
            logger.debug(f"Description: {self.description}")

        # Automatically enable save_reversed_error if verify_by_reversing is enabled
        if (
            self.tracing is not None
            and self.tracing.verify_by_reversing
            and not self.template.save_flags.save_reversed_error
        ):
            logger.debug(
                "Enabling save_reversed_error because verify_by_reversing=True"
            )
            self.template.save_flags.save_reversed_error = True

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        # Ensure I/O module is imported for registration
        import psr_magpy.io
        from psr_magpy.io import get_io_factory
        from psr_magpy.io.config_io import ExperimentConfigIO

        # Ensure registration
        if type(self) not in get_io_factory()._config_registry:
            get_io_factory().register_config_io(type(self), ExperimentConfigIO)

        get_io_factory().save_config(self, filepath)

    def print_yaml(self):
        """Print configuration as pretty YAML to screen."""
        # Ensure I/O module is imported for registration
        import psr_magpy.io
        from psr_magpy.io import get_io_factory
        from psr_magpy.io.config_io import ExperimentConfigIO

        # Ensure registration
        if type(self) not in get_io_factory()._config_registry:
            get_io_factory().register_config_io(type(self), ExperimentConfigIO)

        io_instance = get_io_factory().create_config_io(type(self))
        io_instance.print_config(self)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        # Ensure I/O module is imported for registration
        import psr_magpy.io
        from psr_magpy.io import get_io_factory
        from psr_magpy.io.config_io import ExperimentConfigIO

        # Ensure registration
        if cls not in get_io_factory()._config_registry:
            get_io_factory().register_config_io(cls, ExperimentConfigIO)

        return get_io_factory().load_config(cls, filepath)

    def load_combined_batch_data(
        self, template: FieldLine, lines_dir: Optional[str] = None
    ) -> Optional["FieldLineSet"]:
        """
        Load and combine batch files for this experiment.

        Args:
            template: FieldLine template specifying which properties to load
            lines_dir: Optional path to lines directory (defaults to experiment's output dir)

        Returns:
            FieldLineSet: Combined field line set with specified properties
        """
        import glob
        from psr_magpy.line.field_line_set import FieldLineSet
        from psr_magpy.io import get_io_factory

        # Determine lines directory
        if lines_dir is None:
            error_msg = "lines_dir must be specified"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Find all batch files for this experiment
        batch_pattern = f"{lines_dir}/{self.name}_batch_*.fits"
        batch_files = glob.glob(batch_pattern)

        if not batch_files:
            logger.warning(f"No batch files found for {self.name}")
            return None

        logger.info(f"Found {len(batch_files)} batch files for {self.name}")

        # Create combined field line set
        combined_field_line_set = FieldLineSet(template, copy_data=False)

        # Load data from each batch file
        for batch_file in sorted(batch_files):
            logger.debug(f"Loading {batch_file}")

            # Load the field line set with template
            field_line_set = get_io_factory().load_data(
                FieldLineSet, batch_file, template=template
            )

            # Append to combined set
            combined_field_line_set.append(field_line_set)

        logger.info(
            f"Combined field line set: {combined_field_line_set.n_field_lines} field lines"
        )

        # Check for duplicate footpoint indices
        if (
            hasattr(combined_field_line_set, "footpoint_id")
            and combined_field_line_set.footpoint_id is not None
        ):
            footpoint_ids = combined_field_line_set.footpoint_id
            unique_ids = set(footpoint_ids)
            if len(unique_ids) != len(footpoint_ids):
                duplicates = [id for id in unique_ids if footpoint_ids.count(id) > 1]
                error_msg = f"Duplicate footpoint indices found: {duplicates[:10]}{'...' if len(duplicates) > 10 else ''}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                logger.info(f"Verified {len(footpoint_ids)} unique footpoint indices")

        return combined_field_line_set

    @classmethod
    def create_dipole_example(cls) -> "ExperimentConfig":
        """Create a simple dipole field example."""
        field = FieldConfig(
            components=[
                FieldComponentConfig(
                    moment_type="dipole",
                    B_magnitude=1.0,
                    moment_orientation_latitude_degree=0.0,
                    moment_orientation_longitude_degree=0.0,
                )
            ]
        )

        grid = GridConfig(spherical=SphericalGridConfig(nlon=2, nlat=5, pole_lat=0.0))

        tracing = TracingConfig(
            max_integration_radius=TracingDefaults.MAX_INTEGRATION_RADIUS,
            threshold_line_distance=TracingDefaults.THRESHOLD_LINE_DISTANCE,
            t_span=TracingDefaults.T_SPAN,
            rtol=TracingDefaults.RTOL,
            atol=TracingDefaults.ATOL,
            surface_radius=TracingDefaults.SURFACE_RADIUS,
            dt=5,
            max_step=TracingDefaults.MAX_STEP,
            trace_open_lines_backwards=True,
            verify_by_reversing=False,
            max_memory_mb=TracingDefaults.MAX_MEMORY_MB,
            safety_factor=TracingDefaults.SAFETY_FACTOR,
            max_batch_size=TracingDefaults.MAX_BATCH_SIZE,
            check_interval_sec=TracingDefaults.CHECK_INTERVAL_SEC,
            n_cores=2,
            n_trial=100,
            progress_bar=TracingDefaults.PROGRESS_BAR,
            verbose=TracingDefaults.VERBOSE,
            skip_memory_estimation=True,
            save_results_to_disk=False,
            filename=None,
            output_format=TracingDefaults.OUTPUT_FORMAT,
            overwrite=TracingDefaults.OVERWRITE,
        )

        template = FieldLine.template_all()

        return cls(
            name="dipole_example",
            description="Simple dipole field experiment",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template,
        )
