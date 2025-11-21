"""
Field line tracing parameters and configuration.

This module defines the parameters used for field line tracing,
including integration parameters, performance settings, and output configuration.
"""

import os
from typing import Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, model_validator

import psutil

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyValidationError

logger = get_logger(__name__, console_format_brief=True)


# =============================================================================
# SINGLE SOURCE OF TRUTH FOR ALL DEFAULTS
# =============================================================================
# All default values are defined here to ensure consistency across the codebase


class TracingDefaults:
    """Default values for all tracing parameters."""

    # Integration parameters
    MAX_INTEGRATION_RADIUS = 10.0
    THRESHOLD_LINE_DISTANCE = 1.0
    T_SPAN = (0.0, 1e8)
    T_COEFF = 10.0  # Coefficient for dynamic t_span calculation: max_integration_radius * t_coeff / dt
    RTOL = 5.0e-9
    ATOL = 5.0e-9
    SURFACE_RADIUS = 1.0 - 1.0e-8
    DT = 0.01
    MAX_STEP = 0.01
    METHOD = "RK45"  # Integration method for ODE solver
    INTEGRATION_SYSTEM = "A"  # Integration system: "A" (current) or "B" (t_eval based)
    FIELD_BACKEND = "python"  # Field calculation backend: "python" or "numba"
    SOLVER_BACKEND = "scipy"  # ODE solver backend: "scipy", "numba", or "cyrk"
    TRACE_OPEN_LINES_BACKWARDS = False
    VERIFY_BY_REVERSING = False
    REVERSED_ERROR_THRESHOLD = 1.0e-3  # Maximum allowed reversed error
    DOTPROD_ERROR_THRESHOLD = 1.0e-3  # Maximum allowed dotprod error

    # Performance parameters
    MAX_MEMORY_MB = 100.0
    SAFETY_FACTOR = 0.9
    MAX_BATCH_SIZE = 10000  # Single source of truth for batch size
    BATCH_SIZE_SAFETY_FACTOR = 5  # Safety factor for batch size calculation
    CHECK_INTERVAL_SEC = 1.0
    N_CORES = None  # Auto-detect
    N_TRIAL = 10
    PROGRESS_BAR = True
    VERBOSE = False
    SKIP_MEMORY_ESTIMATION = False

    # Output parameters
    SAVE_RESULTS_TO_DISK = False
    KEEP_RESULTS_IN_MEMORY = False
    FILENAME = None
    OUTPUT_FORMAT = "fits"
    OVERWRITE = False


class IntegrationParameters(BaseModel):
    """Physics and integration parameters for field line tracing."""

    max_integration_radius: float = Field(
        default=TracingDefaults.MAX_INTEGRATION_RADIUS,
        gt=0,
        description="Maximum radius to trace field lines to (in NS radii)",
    )
    threshold_line_distance: float = Field(
        default=TracingDefaults.THRESHOLD_LINE_DISTANCE,
        gt=0,
        description="Threshold maximum line distance from star surface to save the line",
    )
    t_span: Tuple[float, float] = Field(
        default=TracingDefaults.T_SPAN, description="Integration time span (start, end)"
    )
    t_coeff: float = Field(
        default=TracingDefaults.T_COEFF,
        gt=0,
        description="Coefficient for dynamic t_span calculation: max_integration_radius * t_coeff / dt",
    )
    rtol: float = Field(
        default=TracingDefaults.RTOL,
        gt=0,
        description="Relative tolerance for integration",
    )
    atol: float = Field(
        default=TracingDefaults.ATOL,
        gt=0,
        description="Absolute tolerance for integration",
    )
    surface_radius: float = Field(
        default=TracingDefaults.SURFACE_RADIUS,
        gt=0,
        description="Radius of neutron star surface",
    )
    dt: float = Field(
        default=TracingDefaults.DT,
        gt=0,
        description="Integration time step for output sampling",
    )
    max_step: float = Field(
        default=TracingDefaults.MAX_STEP,
        gt=0,
        description="Maximum step size for integration",
    )
    method: str = Field(
        default=TracingDefaults.METHOD,
        description="Integration method for ODE solver (e.g., 'RK45', 'DOP853', 'BDF', 'Radau')",
    )
    integration_system: str = Field(
        default=TracingDefaults.INTEGRATION_SYSTEM,
        description="Integration system: 'A' (current dense_output) or 'B' (t_eval based)",
    )
    field_backend: str = Field(
        default=TracingDefaults.FIELD_BACKEND,
        description="Field calculation backend: 'python' (Python implementation) or 'numba' (Numba-compiled)",
    )
    solver_backend: str = Field(
        default=TracingDefaults.SOLVER_BACKEND,
        description="ODE solver backend: 'scipy' (SciPy solve_ivp), 'numba' (Numba-compiled RK45), or 'cyrk' (CyRK)",
    )
    trace_open_lines_backwards: bool = Field(
        default=TracingDefaults.TRACE_OPEN_LINES_BACKWARDS,
        description="Whether to trace open field lines backwards from surface",
    )
    verify_by_reversing: bool = Field(
        default=TracingDefaults.VERIFY_BY_REVERSING,
        description="Whether to verify field line tracing by reversing the integration",
    )
    reversed_error_threshold: float = Field(
        default=TracingDefaults.REVERSED_ERROR_THRESHOLD,
        gt=0,
        description="Maximum allowed reversed error before marking as error",
    )
    dotprod_error_threshold: float = Field(
        default=TracingDefaults.DOTPROD_ERROR_THRESHOLD,
        gt=0,
        description="Maximum allowed dotprod error before marking as error",
    )

    @model_validator(mode="after")
    def validate_radii(self):
        """Validate that threshold_line_distance is less than max_integration_radius."""
        if self.threshold_line_distance >= self.max_integration_radius:
            raise ValueError(
                f"threshold_line_distance ({self.threshold_line_distance}) must be < max_integration_radius ({self.max_integration_radius})"
            )
        return self

    @model_validator(mode="after")
    def validate_tolerances(self):
        """Validate integration tolerances."""
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError(
                f"Tolerances must be positive: rtol={self.rtol}, atol={self.atol}"
            )
        return self

    @model_validator(mode="after")
    def validate_dt(self):
        """Validate that dt is reasonable."""
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.dt > 1.0:
            logger.warning(
                f"Large dt value: {self.dt}. This may affect integration accuracy."
            )
        return self

    @model_validator(mode="after")
    def validate_integration_system(self):
        """Validate integration system parameter."""
        if self.integration_system not in ["A", "B"]:
            raise ValueError(
                f"integration_system must be 'A' or 'B', got {self.integration_system}"
            )
        return self

    @model_validator(mode="after")
    def validate_field_backend(self):
        """Validate field backend parameter."""
        if self.field_backend not in ["python", "numba"]:
            raise ValueError(
                f"field_backend must be 'python' or 'numba', got {self.field_backend}"
            )
        return self

    @model_validator(mode="after")
    def validate_solver_backend(self):
        """Validate solver backend parameter."""
        if self.solver_backend not in ["scipy", "numba", "cyrk"]:
            raise ValueError(
                f"solver_backend must be 'scipy', 'numba', or 'cyrk', got {self.solver_backend}"
            )
        return self


class PerformanceParameters(BaseModel):
    """Memory and processing parameters."""

    max_memory_mb: float = Field(
        default=TracingDefaults.MAX_MEMORY_MB,
        gt=0,
        description="Maximum memory allowed for processing in megabytes",
    )
    safety_factor: float = Field(
        default=TracingDefaults.SAFETY_FACTOR,
        gt=0,
        le=1.0,
        description="Fraction of available memory to leave as buffer",
    )
    max_batch_size: int = Field(
        default=TracingDefaults.MAX_BATCH_SIZE,
        gt=0,
        description="Maximum batch size to process",
    )
    batch_size_safety_factor: int = Field(
        default=TracingDefaults.BATCH_SIZE_SAFETY_FACTOR,
        gt=0,
        description="Safety factor for batch size calculation (max_memory_mb / batch_size_safety_factor)",
    )
    check_interval_sec: float = Field(
        default=TracingDefaults.CHECK_INTERVAL_SEC,
        gt=0,
        description="How often to check memory usage in seconds",
    )
    n_cores: Optional[int] = Field(
        default=TracingDefaults.N_CORES,
        ge=1,
        description="Number of CPU cores to use (None for auto-detection)",
    )
    n_trial: int = Field(
        default=TracingDefaults.N_TRIAL,
        gt=0,
        description="Number of trial field lines for memory estimation",
    )
    progress_bar: bool = Field(
        default=TracingDefaults.PROGRESS_BAR,
        description="Whether to show progress bars",
    )
    verbose: bool = Field(
        default=TracingDefaults.VERBOSE, description="Verbosity level for logging"
    )
    skip_memory_estimation: bool = Field(
        default=TracingDefaults.SKIP_MEMORY_ESTIMATION,
        description="Whether to skip memory estimation and process all field lines at once",
    )

    @model_validator(mode="after")
    def setup_cores(self):
        """Set up and validate the number of CPU cores to use."""
        available_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
        logger.debug(f"Available physical cores: {available_cores}")

        if self.n_cores is None:
            self.n_cores = available_cores
            logger.debug(
                f"No core count specified, using all available cores: {self.n_cores}"
            )

        self.n_cores = max(1, self.n_cores)
        max_cores = psutil.cpu_count()
        if self.n_cores > max_cores:
            warning_msg = f"Requested {self.n_cores} cores, but only {max_cores} available. Using {max_cores} cores."
            logger.warning(warning_msg)
            self.n_cores = max_cores

        logger.info(f"Using {self.n_cores} CPU cores for processing")
        return self

    def validate_against_grid(self, grid_size: int) -> None:
        """Validate n_trial against grid size.

        Raises
        ------
        ValueError
            If n_trial exceeds grid size
        """
        if self.n_trial > grid_size:
            raise PSRMagPyValidationError(
                f"n_trial ({self.n_trial}) cannot exceed grid size ({grid_size})"
            )


class OutputParameters(BaseModel):
    """Output and file I/O parameters."""

    save_results_to_disk: bool = Field(
        default=TracingDefaults.SAVE_RESULTS_TO_DISK,
        description="Whether to save results to disk",
    )
    keep_results_in_memory: bool = Field(
        default=TracingDefaults.KEEP_RESULTS_IN_MEMORY,
        description="Whether to keep full results in memory (use at your own risk)",
    )
    filename: Optional[str] = Field(
        default=TracingDefaults.FILENAME,
        description="Base filename for output (without extension)",
    )
    output_format: str = Field(
        default=TracingDefaults.OUTPUT_FORMAT, description="Output file format"
    )
    overwrite: bool = Field(
        default=TracingDefaults.OVERWRITE,
        description="Whether to overwrite existing files",
    )

    @model_validator(mode="after")
    def validate_filename(self):
        """Validate filename configuration."""
        if self.save_results_to_disk and not self.filename:
            raise ValueError("Filename must be specified when save_results_to_disk is True")
        if not self.save_results_to_disk and self.filename:
            logger.warning("Filename specified but save_results_to_disk is False")
            raise ValueError(
                "filename should not be specified when save_results_to_disk is False"
            )

        if self.keep_results_in_memory:
            logger.warning(
                "keep_results_in_memory=True: Use at your own risk - this may consume significant memory"
            )
            if not self.save_results_to_disk:
                logger.warning(
                    "keep_results_in_memory=True with save_results_to_disk=False: Results will be kept in memory but not saved to disk"
                )

        if self.filename:
            directory = os.path.dirname(self.filename)
            if directory and not os.path.exists(directory):
                # For experiment series, directories may be created later
                # Just log a warning instead of raising an error
                logger.warning(
                    f"Output directory does not exist yet (will be created): {directory}"
                )

        logger.debug(f"OutputParameters initialized: {self}")
        return self


class TracingParameters(BaseModel):
    """Main configuration container for field line tracing."""

    integration: IntegrationParameters = Field(
        default_factory=IntegrationParameters,
        description="Physics and integration parameters",
    )
    performance: PerformanceParameters = Field(
        default_factory=PerformanceParameters,
        description="Memory and processing parameters",
    )
    output: OutputParameters = Field(
        default_factory=OutputParameters, description="Output and file I/O parameters"
    )

    # Convenience property for common access
    @property
    def threshold_line_distance(self) -> float:
        """Minimum value for max_integration_radius to save the line."""
        return self.integration.threshold_line_distance
