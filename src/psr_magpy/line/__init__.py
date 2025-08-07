"""Field line tracing system.

This package provides functionality for tracing magnetic field lines
through 3D space, including integration, property calculation,
and result management.

**FieldLine Dual Role:**

FieldLine objects serve two distinct purposes in the system:

1. **Configuration Templates** (Primary):
   - Define which properties to save during tracing
   - Used as templates: FieldLine.template_all(), FieldLine.template_OFL_calc()
   - Stored as YAML configuration files
   - Human-readable configuration format

2. **Data Containers** (Secondary):
   - Hold actual field line data (coordinates, properties)
   - Combined into FieldLineSet for bulk storage
   - Stored as FITS files (scientific data format)

**I/O Distinction:**
- Configuration FieldLines → YAML files
- Data FieldLines → FITS files via FieldLineSet
"""

from .tracer import FieldLineTracer
from .field_line import FieldLine
from .field_line_set import FieldLineSet
from .property_system import PropertySystem, PropertyName, SaveFlags
from .tracing_parameters import (
    TracingParameters,
    IntegrationParameters,
    PerformanceParameters,
    OutputParameters,
)
from .property_calculator import PropertyCalculator
from .progress_tracker import ProgressTracker
from .memory_manager import MemoryManager
from .calculation_context import CalculationContext
from .calculator import FieldLineCalculator


__all__ = [
    "FieldLine",
    "FieldLineSet",
    "FieldLineTracer",
    "TracingParameters",
    "IntegrationParameters",
    "PerformanceParameters",
    "OutputParameters",
    "PropertySystem",
    "PropertyName",
    "SaveFlags",
    "MemoryManager",
    "ProgressTracker",
    "CalculationContext",
    "FieldLineCalculator",
]
