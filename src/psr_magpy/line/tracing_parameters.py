"""Unified parameter management for field line tracing.

This module consolidates all configuration parameters used throughout
the field line tracing system into a single, well-organized parameter
hierarchy with validation and documentation.

Classes:
    TracingParameters: Main configuration container
    PhysicsParameters: Physics and integration settings
    PerformanceParameters: Memory and processing settings
    OutputParameters: File I/O and output settings
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple
from pydantic import BaseModel, Field, model_validator

import psutil

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class PhysicsParameters(BaseModel):
    """Physics and integration parameters for field line tracing.
    
    Configures the physics of field line tracing including integration
    tolerances, time span, surface radius, and tracing limits.
    
    Attributes:
        R_max: Maximum radius to trace field lines to
        min_R_max_line: Minimum value for R_max to save the line
        t_span: Integration time span (start, end)
        rtol: Relative tolerance for integration
        atol: Absolute tolerance for integration
        surface_radius: Radius of neutron star surface
        dt: Integration time step for output sampling
    """
    
    R_max: float = Field(
        default=10.0,
        gt=0,
        description="Maximum radius to trace field lines to (in NS radii)"
    )
    min_R_max_line: float = Field(
        default=1.0,
        gt=0,
        description="Minimum value for R_max to save the line"
    )
    t_span: Tuple[float, float] = Field(
        default=(0.0, 1e8),
        description="Integration time span (start, end)"
    )
    rtol: float = Field(
        default=5.0e-9,
        gt=0,
        description="Relative tolerance for integration"
    )
    atol: float = Field(
        default=5.0e-9,
        gt=0,
        description="Absolute tolerance for integration"
    )
    surface_radius: float = Field(
        default=0.999999,
        gt=0,
        description="Radius of neutron star surface"
    )
    dt: float = Field(
        default=0.01,
        gt=0,
        description="Integration time step for output sampling"
    )
    
    @model_validator(mode='after')
    def validate_radii(self):
        """Validate that min_R_max_line is less than R_max."""
        if self.min_R_max_line >= self.R_max:
            error_msg = f"min_R_max_line ({self.min_R_max_line}) must be < R_max ({self.R_max})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return self
    
    @model_validator(mode='after')
    def validate_tolerances(self):
        """Validate integration tolerances."""
        if self.rtol <= 0 or self.atol <= 0:
            error_msg = f"Tolerances must be positive: rtol={self.rtol}, atol={self.atol}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        return self
    
    @model_validator(mode='after')
    def validate_dt(self):
        """Validate that dt is reasonable."""
        if self.dt <= 0:
            error_msg = f"dt must be positive, got {self.dt}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.dt > 1.0:
            logger.warning(f"Large dt value: {self.dt}. This may affect integration accuracy.")
        return self


class PerformanceParameters(BaseModel):
    """Memory and processing parameters.
    
    Configures memory limits, batch processing, CPU cores, and
    performance tuning for field line calculations.
    
    Attributes:
        max_memory_mb: Maximum memory allowed for processing in megabytes
        safety_factor: Fraction of available memory to leave as buffer
        max_batch_size: Maximum batch size to process
        check_interval_sec: How often to check memory usage
        n_cores: Number of CPU cores to use
        n_trial: Number of trial field lines for estimation
        progress_bar: Whether to show progress bars
        verbose: Verbosity level for logging
    """
    
    max_memory_mb: float = Field(
        default=100.0,
        gt=0,
        description="Maximum memory allowed for processing in megabytes"
    )
    safety_factor: float = Field(
        default=0.9,
        gt=0,
        le=1.0,
        description="Fraction of available memory to leave as buffer"
    )
    max_batch_size: int = Field(
        default=100,
        gt=0,
        description="Maximum batch size to process"
    )
    check_interval_sec: float = Field(
        default=1.0,
        gt=0,
        description="How often to check memory usage in seconds"
    )
    n_cores: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of CPU cores to use (None for auto-detection)"
    )
    n_trial: int = Field(
        default=100,
        gt=0,
        description="Number of trial field lines for memory estimation"
    )
    progress_bar: bool = Field(
        default=True,
        description="Whether to show progress bars"
    )
    verbose: bool = Field(
        default=False,
        description="Verbosity level for logging"
    )
    
    @model_validator(mode='after')
    def setup_cores(self):
        """Setup and validate the number of CPU cores to use."""
        available_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
        logger.debug(f"Available physical cores: {available_cores}")
        
        if self.n_cores is None:
            self.n_cores = available_cores
            logger.debug(f"No core count specified, using all available cores: {self.n_cores}")

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
        
        Parameters
        ----------
        grid_size : int
            Number of field lines in the grid
            
        Raises
        ------
        ValueError
            If n_trial exceeds grid size
        """
        if self.n_trial > grid_size:
            error_msg = f"n_trial ({self.n_trial}) cannot exceed grid size ({grid_size})"
            logger.error(error_msg)
            raise ValueError(error_msg)


class OutputParameters(BaseModel):
    """Output and file I/O parameters.
    
    Configures file output settings, including save options,
    filename patterns, and output formats.
    
    Attributes:
        save_results: Whether to save results to disk
        filename: Base filename for output (without extension)
        output_format: Output file format
        overwrite: Whether to overwrite existing files
    """
    
    save_results: bool = Field(
        default=True,
        description="Whether to save results to disk"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Base filename for output (without extension)"
    )
    output_format: str = Field(
        default="fits",
        description="Output file format"
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing files"
    )
    
    @model_validator(mode='after')
    def validate_filename(self):
        """Validate filename configuration."""
        if self.save_results and not self.filename:
            error_msg = "Filename must be specified when save_results is True"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not self.save_results and self.filename:
            logger.warning("Filename specified but save_results is False")
            raise ValueError(
                "filename should not be specified when save_results is False"
            )

        if self.filename:
            directory = os.path.dirname(self.filename)
            if directory and not os.path.exists(directory):
                error_msg = f"Output directory does not exist: {directory}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        logger.debug(f"OutputParameters initialized: {self}")
        return self


class TracingParameters(BaseModel):
    """Main configuration container for field line tracing.
    
    Consolidates all parameters needed for field line tracing into
    a single, validated configuration object with 3 core parameter groups.
    
    Attributes:
        physics: Physics and integration parameters
        performance: Memory and processing parameters
        output: Output and file I/O parameters
    """
    
    physics: PhysicsParameters = Field(
        default_factory=PhysicsParameters,
        description="Physics and integration parameters"
    )
    performance: PerformanceParameters = Field(
        default_factory=PerformanceParameters,
        description="Memory and processing parameters"
    )
    output: OutputParameters = Field(
        default_factory=OutputParameters,
        description="Output and file I/O parameters"
    )
    
    # Convenience properties for backward compatibility
    @property
    def R_max(self) -> float:
        """Maximum radius to trace field lines to."""
        return self.physics.R_max
    
    @property
    def min_R_max_line(self) -> float:
        """Minimum value for R_max to save the line."""
        return self.physics.min_R_max_line
    
    @property
    def integration(self) -> PhysicsParameters:
        """Integration parameters (alias for physics)."""
        return self.physics 