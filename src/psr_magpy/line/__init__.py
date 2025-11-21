"""Field line tracing system.

This package provides functionality for tracing magnetic field lines
through 3D space, including integration, property calculation,
and result management.

**Configuration and Data Separation:**

1. **FieldLineTemplate** (Configuration):
   - Defines which properties to save during tracing
   - Used as templates: FieldLineTemplate.template_all(), FieldLineTemplate.template_OFL_calc()
   - Stored as YAML configuration files
   - Human-readable configuration format

2. **FieldLine** (Data Container):
   - Holds actual field line data (coordinates, properties)
   - Combined into FieldLineSet for bulk storage
   - Stored as FITS files (scientific data format)

**I/O Distinction:**
- FieldLineTemplate → YAML files (configuration)
- FieldLine/FieldLineSet → FITS files (scientific data)
"""

from .tracer import FieldLineTracer
from .field_line import FieldLine
from .field_line_template import FieldLineTemplate
from .field_line_set import FieldLineSet
from .property_system import PropertySystem, PropertyName, SaveFlags
from .tracing_parameters import (
    TracingParameters,
    IntegrationParameters,
    PerformanceParameters,
    OutputParameters,
)
from .property_calculator import PropertyCalculator
from .progress_tracker import ProgressTracker, ProgressReporter
from .memory_manager import MemoryManager
from .calculation_context import CalculationContext
from .calculator import FieldLineCalculator
from .open_field_line_edges import (
    OFLEdgeCalculator,
    OFLEdgeResult,
    calculate_and_save_OFL_edges,
)
from .solver import SolverError


__all__ = [
    "FieldLine",
    "FieldLineTemplate",
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
    "ProgressReporter",
    "CalculationContext",
    "FieldLineCalculator",
    "OFLEdgeCalculator",
    "OFLEdgeResult",
    "calculate_and_save_OFL_edges",
    "SolverError",
]
