from astropy.io import fits as pf

from src.fitsio.base import FitsIO
from src.field.components import FieldComponentProtocol
from src.field.components import MultipoleType, DipoleFieldComponent, QuadrupoleFieldComponent
from numpy import ndarray


class FieldComponentFitsIO(FitsIO):
    """FITS I/O operations for field components."""

    def generate_column_info(self):
        """Generate column information for field component data."""
        first_component = self.data[0]
        if not isinstance(first_component, FieldComponentProtocol):
            raise TypeError(
                f"Expected field component, got {type(first_component).__name__}"
            )
       
        params_dict = first_component.params._asdict()
        

        return {
            param_name: (
                FitsIO.FITS_TYPE_MAPPING[str]
                if param_name == "moment_type"
                else (
                    FitsIO.FITS_TYPE_MAPPING[list]
                    if param_name == "offset_r_ns"
                    else FitsIO.get_fits_format(param_value)
                )
            )
            for param_name, param_value in params_dict.items()
            if param_value is not None
        }

    def create_column(self, key: str, format: str):
        """Create a FITS column for field component data.

        Args:
            key: Column name.
            format: FITS format string.

        Returns:
            FITS column object.
        """
        if key == "moment_type":
            value = [v.params.moment_type.value for v in self.data]
        else:
            value = [getattr(v.params, key) for v in self.data]
        column = pf.Column(name=key, format=format, array=value)
        print(column)
        return column

    @classmethod
    def load_from_fits(cls, hdu):
        """Load field components from a FITS HDU.

        Args:
            hdu: FITS HDU containing the field component data.

        Returns:
            List of field component instances.
        """
        data = hdu.data
        n_components = data.shape[0]
        components = []
        
        for i in range(n_components):
            # Determine the component type from the moment_type column
            moment_type = MultipoleType(data['moment_type'][i])
            component_class = (
                DipoleFieldComponent if moment_type == MultipoleType.DIPOLE
                else QuadrupoleFieldComponent
            )
            
            # Create component instance with parameters from FITS file
            component_params = {
                key: data[key][i] for key in hdu.columns.names
                if key != 'moment_type'  # Handle moment_type separately
            }
            component_params['moment_type'] = moment_type
            
            components.append(component_class(**component_params))
            
        return cls(components)

