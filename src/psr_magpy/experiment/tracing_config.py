"""Tracing configuration classes for experiment setup."""

from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel

from psr_magpy.line.tracing_parameters import (
    TracingParameters,
    IntegrationParameters,
    OutputParameters,
    PerformanceParameters,
    TracingDefaults,
)
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


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
    field_backend: str = TracingDefaults.FIELD_BACKEND
    solver_backend: str = TracingDefaults.SOLVER_BACKEND
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
            field_backend=self.field_backend,
            solver_backend=self.solver_backend,
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

