"""Experiment configuration for field line calculations.

Architecture boundary note
--------------------------
This module should remain primarily a configuration/data model. Batch data
access and filesystem-heavy helpers are candidates for migration to
``experiment.services`` and ``experiment.infrastructure``.
"""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Iterator, Callable
from pathlib import Path
import warnings

from .field_config import FieldConfig, FieldComponentConfig
from .grid_config import GridConfig, SphericalGridConfig
from .tracing_config import TracingConfig
from .services.batch_data_reader import BatchDataReader
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.line.tracing_parameters import TracingDefaults

logger = get_logger(__name__, console_format_brief=True)

_SIDE_EFFECT_DEPRECATION = (
    "ExperimentConfig model I/O/workflow helpers are deprecated and will move "
    "to services/repositories. Prefer dedicated service APIs."
)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str
    field: FieldConfig
    grid: GridConfig
    tracing: TracingConfig
    template: Union[FieldLineTemplate, FieldLine]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration."""
        logger.debug(f"Initializing experiment: {self.name}")
        if self.description:
            logger.debug(f"Description: {self.description}")

        # Convert FieldLineTemplate to FieldLine if needed (for mutability)
        if isinstance(self.template, FieldLineTemplate):
            self.template = FieldLine.from_template(self.template)

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

    def save(self, filepath: Union[str, Path], overwrite: bool = False) -> None:
        """Save configuration to YAML file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path where to save the configuration file.
        overwrite : bool, optional
            If True, delete existing file before saving. If False and file exists,
            raises FileExistsError (handled by underlying I/O system). Default is False.
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        filepath = Path(filepath)
        
        # Handle overwrite flag
        if overwrite and filepath.exists():
            filepath.unlink()
            logger.debug(f"Deleted existing config file: {filepath}")
        
        from psr_magpy.io import save_config

        save_config(self, filepath)

    def print_yaml(self):
        """Print configuration as pretty YAML to screen."""
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        from psr_magpy.io.config_io import ExperimentConfigIO
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
        io_instance = ExperimentConfigIO(
            validator=validator,
            memory_monitor=memory_monitor,
        )
        io_instance.print_config(self)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        from psr_magpy.io import load_config

        return load_config(cls, filepath)

    def load_combined_batch_data(
        self, template: FieldLine, lines_dir: Optional[str] = None
    ) -> Optional["FieldLineSet"]:
        """Load and combine batch files for this experiment."""
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        return BatchDataReader(self.name).load_combined_batch_data(
            template=template,
            lines_dir=lines_dir,
        )

    def iter_batch_files(self, lines_dir: str) -> Iterator[Path]:
        """Iterate over batch file paths for this experiment.
        
        Parameters
        ----------
        lines_dir : str
            Directory containing batch files
            
        Yields
        ------
        Path
            Path to each batch file in sorted order
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        yield from BatchDataReader(self.name).iter_batch_files(lines_dir)

    def iter_batch_data(
        self, 
        template: FieldLine, 
        lines_dir: str
    ) -> Iterator[FieldLineSet]:
        """Iterate over loaded batch data for this experiment.
        
        Parameters
        ----------
        template : FieldLine
            FieldLine template specifying which properties to load
        lines_dir : str
            Directory containing batch files
            
        Yields
        ------
        FieldLineSet
            Loaded field line set for each batch (skips empty batches)
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        yield from BatchDataReader(self.name).iter_batch_data(
            template=template,
            lines_dir=lines_dir,
        )

    def process_batches_incremental(
        self,
        template: FieldLine,
        lines_dir: str,
        processor_fn: Callable[[FieldLineSet], Any],
        accumulator_fn: Callable[[Any, Any], Any],
        initial_value: Any = None,
    ) -> Any:
        """Process batches incrementally, accumulating results.
        
        This method loads batches one at a time, processes each batch using
        processor_fn, and accumulates results using accumulator_fn. This is
        memory-efficient for operations that transform FieldLineSet data into
        other formats (e.g., dictionaries, arrays, custom objects).
        
        Parameters
        ----------
        template : FieldLine
            Template for loading batch data
        lines_dir : str
            Directory containing batch files
        processor_fn : Callable[[FieldLineSet], Any]
            Function that processes a batch and returns result.
            Signature: (batch_data: FieldLineSet) -> Any
        accumulator_fn : Callable[[Any, Any], Any]
            Function that accumulates results.
            Signature: (accumulated_result: Any, new_result: Any) -> Any
            First call: accumulator_fn(initial_value, first_batch_result)
        initial_value : Any, optional
            Initial accumulator value. If None, first batch result becomes initial value.
            
        Returns
        -------
        Any
            Final accumulated result
            
        Examples
        --------
        >>> # Build a dictionary mapping footpoint_id to max radius
        >>> def process_batch(batch_data):
        ...     footpoint_ids = batch_data.footpoint_id
        ...     r_cyl_max = batch_data.get_property('r_cyl_max')
        ...     return {fp_id: r_max for fp_id, r_max in zip(footpoint_ids, r_cyl_max)}
        >>> 
        >>> def accumulate_dict(accumulated, new):
        ...     if accumulated is None:
        ...         return new.copy()
        ...     for fp_id, r_max in new.items():
        ...         accumulated[fp_id] = max(accumulated.get(fp_id, 0), r_max)
        ...     return accumulated
        >>> 
        >>> result = exp_config.process_batches_incremental(
        ...     template=template,
        ...     lines_dir=str(lines_dir),
        ...     processor_fn=process_batch,
        ...     accumulator_fn=accumulate_dict,
        ...     initial_value=None
        ... )
        """
        warnings.warn(_SIDE_EFFECT_DEPRECATION, DeprecationWarning, stacklevel=2)
        return BatchDataReader(self.name).process_batches_incremental(
            template=template,
            lines_dir=lines_dir,
            processor_fn=processor_fn,
            accumulator_fn=accumulator_fn,
            initial_value=initial_value,
        )

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

        from psr_magpy.line.field_line_template import FieldLineTemplate
        template = FieldLineTemplate.template_all()

        return cls(
            name="dipole_example",
            description="Simple dipole field experiment",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template,
        )
