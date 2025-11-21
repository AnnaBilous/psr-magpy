"""Custom exception classes with auto-logging for psr-magpy.

This module provides a unified exception hierarchy with automatic logging
to reduce boilerplate and ensure consistent error handling across the codebase.
"""

import logging
from typing import Optional

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class PSRMagPyError(Exception):
    """Base exception class with auto-logging.
    
    All custom exceptions in psr-magpy should inherit from this class.
    It automatically logs the error message when the exception is raised.
    
    Parameters
    ----------
    message : str
        Error message to display and log
    log_level : int, optional
        Logging level (default: logging.ERROR)
    """
    
    def __init__(self, message: str, log_level: int = logging.ERROR):
        self.message = message
        logger.log(log_level, message)
        super().__init__(message)


class PSRMagPyValidationError(PSRMagPyError, ValueError):
    """Exception for validation errors.
    
    Raised when input validation fails (e.g., invalid parameters,
    missing required fields, type mismatches).
    """
    
    pass


class PSRMagPyIOError(PSRMagPyError, OSError):
    """Exception for I/O operation errors.
    
    Raised when file operations fail (e.g., file not found,
    permission denied, read/write errors).
    """
    
    pass


class PSRMagPyGridError(PSRMagPyError, ValueError):
    """Exception for grid-related errors.
    
    Raised when grid operations fail (e.g., invalid grid parameters,
    coordinate errors, subset creation failures).
    """
    
    pass


class PSRMagPyFieldError(PSRMagPyError, ValueError):
    """Exception for field calculation errors.
    
    Raised when field computation fails (e.g., invalid field components,
    calculation errors, initialization failures).
    """
    
    pass


class PSRMagPyTracingError(PSRMagPyError, RuntimeError):
    """Exception for field line tracing errors.
    
    Raised when field line integration fails (e.g., solver errors,
    integration failures, convergence issues).
    """
    
    pass


class PSRMagPyPropertyError(PSRMagPyError, ValueError):
    """Exception for property-related errors.
    
    Raised when property operations fail (e.g., invalid property names,
    property calculation errors, validation failures).
    """
    
    pass


class PSRMagPyExperimentError(PSRMagPyError, RuntimeError):
    """Exception for experiment-related errors.
    
    Raised when experiment operations fail (e.g., config errors,
    series creation failures, runner errors).
    """
    
    pass

