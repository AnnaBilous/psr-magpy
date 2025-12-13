"""Experiment configuration for field line calculations."""

from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Iterator, Callable
from pathlib import Path

from .field_config import FieldConfig, FieldComponentConfig
from .grid_config import GridConfig, SphericalGridConfig
from .tracing_config import TracingConfig
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyValidationError
from psr_magpy.line.tracing_parameters import TracingDefaults

logger = get_logger(__name__, console_format_brief=True)


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
        filepath = Path(filepath)
        
        # Handle overwrite flag
        if overwrite and filepath.exists():
            filepath.unlink()
            logger.debug(f"Deleted existing config file: {filepath}")
        
        from psr_magpy.io import save_config

        save_config(self, filepath)

    def print_yaml(self):
        """Print configuration as pretty YAML to screen."""
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
        from psr_magpy.io import load_config

        return load_config(cls, filepath)

    def load_combined_batch_data(
        self, template: FieldLine, lines_dir: Optional[str] = None
    ) -> Optional["FieldLineSet"]:
        """
        Load and combine batch files for this experiment.

        Parameters
        ----------
        template : FieldLine
            FieldLine template specifying which properties to load.
        lines_dir : Optional[str], optional
            Optional path to lines directory. Defaults to experiment's output dir.

        Returns
        -------
        Optional[FieldLineSet]
            Combined field line set with specified properties.
        """
        import glob
        from psr_magpy.line.field_line_set import FieldLineSet
        from psr_magpy.io import load_data

        # Determine lines directory
        if lines_dir is None:
            raise PSRMagPyValidationError("lines_dir must be specified")

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
            from psr_magpy.io import load_data
            field_line_set = load_data(
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
                raise PSRMagPyValidationError(
                    f"Duplicate footpoint indices found: {duplicates[:10]}{'...' if len(duplicates) > 10 else ''}"
                )
            else:
                logger.info(f"Verified {len(footpoint_ids)} unique footpoint indices")

        return combined_field_line_set

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
        import glob
        
        if lines_dir is None:
            raise PSRMagPyValidationError("lines_dir must be specified")
        
        batch_pattern = f"{lines_dir}/{self.name}_batch_*.fits"
        batch_files = sorted(glob.glob(batch_pattern))
        
        for batch_file in batch_files:
            yield Path(batch_file)

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
        from psr_magpy.io import load_data
        
        for batch_file in self.iter_batch_files(lines_dir):
            logger.debug(f"Loading batch: {batch_file.name}")
            batch_data = load_data(FieldLineSet, batch_file, template=template)
            if batch_data is not None and batch_data.n_field_lines > 0:
                yield batch_data
            else:
                logger.debug(f"Skipping empty batch: {batch_file.name}")

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
        result = initial_value
        batch_count = 0
        
        for batch_data in self.iter_batch_data(template, lines_dir):
            batch_count += 1
            logger.debug(f"Processing batch {batch_count}: {batch_data.n_field_lines} field lines")
            
            batch_result = processor_fn(batch_data)
            
            if result is None and initial_value is None:
                # First batch: use its result as initial value
                result = batch_result
            else:
                result = accumulator_fn(result, batch_result)
        
        logger.info(f"Processed {batch_count} batches")
        return result

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
