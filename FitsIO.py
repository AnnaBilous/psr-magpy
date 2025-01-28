from typing import List, Type
from astropy.io import fits as pf
from FieldLineDefinition import FieldLine
from FieldComponent import FieldComponent
from numpy.typing import NDArray
from numpy import ndarray, integer, floating

# Define FITS format mapping at module level
FITS_TYPE_MAPPING = {
    float: "D",
    int: "I",
    str: "10A",
    ndarray: "PE",  # Variable-length array of doubles
    integer: "I",   # numpy integers
    floating: "D"   # numpy floats
}

def get_fits_format(value):
    """Helper function to determine FITS format for a value."""
    for type_class, format_str in FITS_TYPE_MAPPING.items():
        if isinstance(value, type_class):
            return format_str
    return FITS_TYPE_MAPPING[type(value)]


class FitsIO:
    def __init__(self, data):
        self.data = data

    def save_to_fits(self, filename: str):
        hdu = self.create_hdu()
        hdu.writeto(filename, overwrite=True)

    def create_hdu(self):
        column_info = self.generate_column_info()
        columns = [
            self.create_column(key, column_info[key]) 
            for key in column_info.keys()
        ]
        hdu = pf.BinTableHDU.from_columns(columns, name="TEST")
        return hdu

    def generate_column_info(self):
        raise NotImplementedError("Subclasses must implement generate_column_info")

    def create_column(self, key: str, format: str):
        raise NotImplementedError("Subclasses must implement create_column")

    @classmethod
    def load_from_fits(cls, hdu, component_class: Type):
        data = hdu.data
        n_components = data.shape[0]
        components = [
            component_class(**{key: hdu.data[key][i] for key in hdu.columns.names})
            for i in range(n_components)
        ]
        return cls(components)

class FieldComponentFitsIO(FitsIO):
    def generate_column_info(self):
        first_component = self.data[0]
        if not isinstance(first_component, FieldComponent):
            raise TypeError(f"Expected FieldComponent, got {type(first_component).__name__}")
            
        params_dict = vars(first_component.params)
        
        return {
            param_name: FITS_TYPE_MAPPING[str] if param_name == "moment_type" 
                       else FITS_TYPE_MAPPING[ndarray] if param_name == "field_lines"
                       else get_fits_format(param_value)
            for param_name, param_value in params_dict.items()
            if param_value is not None
        }

    def create_column(self, key: str, format: str):
        if key == "moment_type":
            value = [v.params.moment_type.value for v in self.data]
        else:
            value = [getattr(v.params, key) for v in self.data]
        return pf.Column(name=key, format=format, array=value)

class FieldLineFitsIO(FitsIO):
    def generate_column_info(self):
        first_line = self.data[0]
        if not isinstance(first_line, FieldLine):
            raise TypeError(f"Expected FieldLine, got {type(first_line).__name__}")
            
        params_dict = vars(first_line)
        
        return {
            param_name: get_fits_format(param_value)
            for param_name, param_value in params_dict.items()
            if param_value is not None
        }

    def create_column(self, key: str, format: str):
        value = [getattr(v, key) for v in self.data]
        return pf.Column(name=key, format=format, array=value)  