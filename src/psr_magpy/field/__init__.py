"""Field module for magnetic field calculations and visualization."""

from .components import (
    FieldComponent,
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
    MultipoleType,
    FieldComponentParams,
)
from .multipole import MultipoleField, MultipoleFieldParams
from .visualization import FieldVisualizer

__all__ = [
    "FieldComponent",
    "DipoleFieldComponent", 
    "QuadrupoleFieldComponent",
    "MultipoleType",
    "FieldComponentParams",
    "MultipoleField",
    "MultipoleFieldParams",
    "FieldVisualizer",
]

"""Magnetic field components and calculations."""
