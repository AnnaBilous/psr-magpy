from pathlib import Path
# Create source code directories
src_dir = Path("src")
field_dir = src_dir / "field"
footpoint_grid_dir = src_dir / "footpoint_grid"
line_dir = src_dir / "line"

for directory in [field_dir, footpoint_grid_dir, line_dir]:
    directory.mkdir(parents=True, exist_ok=True)

from .definition import MultipoleField, DipoleFieldComponent, QuadrupoleFieldComponent, FieldComponentProtocol
from .types import MultipoleType
from .io import FitsIO

__all__ = [
    'MultipoleField',
    'DipoleFieldComponent',
    'QuadrupoleFieldComponent',
    'FieldComponentProtocol',
    'MultipoleType',
    'FitsIO'
] 