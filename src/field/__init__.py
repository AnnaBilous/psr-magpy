"""Field package for magnetic field calculations.

This package provides classes and utilities for working with magnetic multipole fields.
"""

from .components import (
    MultipoleType,
    FieldComponent,
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
    FieldComponentProtocol,
)
from .multipole import MultipoleField

__all__ = [
    'MultipoleType',
    'FieldComponent',
    'DipoleFieldComponent',
    'QuadrupoleFieldComponent',
    'FieldComponentProtocol',
    'MultipoleField',
] 