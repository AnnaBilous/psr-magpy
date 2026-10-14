"""Field module for magnetic field calculations."""

from .components import (
    FieldComponent,
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
    MultipoleType,
    FieldComponentParams,
)
from .multipole import MultipoleField, MultipoleFieldParams

__all__ = [
    "FieldComponent",
    "DipoleFieldComponent",
    "QuadrupoleFieldComponent",
    "MultipoleType",
    "FieldComponentParams",
    "MultipoleField",
    "MultipoleFieldParams",
]

"""Magnetic field components and calculations."""
