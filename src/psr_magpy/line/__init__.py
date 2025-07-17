"""Field line tracing system.

This package provides functionality for tracing magnetic field lines
through 3D space, including integration, property calculation,
and result management.
"""

from .tracer import FieldLineTracer
from .field_line import FieldLine
from .field_line_set import FieldLineSet
from .property_system import PropertySystem, PropertyName, SaveFlags
from .tracing_parameters import TracingParameters, IntegrationParameters, PerformanceParameters, OutputParameters
from .property_calculator import PropertyCalculator
from .progress_tracker import ProgressTracker
from .memory_manager import MemoryManager
from .calculation_context import CalculationContext
from .calculator import FieldLineCalculator
from .io import FieldLineSetFitsIO

__all__ = [
    'FieldLine',
    'FieldLineSet',
    'FieldLineTracer',
    'TracingParameters',
    'IntegrationParameters',
    'PerformanceParameters',
    'OutputParameters',
    'PropertySystem',
    'PropertyName',
    'SaveFlags',
    'MemoryManager',
    'ProgressTracker',
    'CalculationContext',
    'FieldLineCalculator',
    'FieldLineSetFitsIO',
]
