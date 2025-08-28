"""Field line set module."""

from dataclasses import dataclass
from typing import Optional
from typing import List, Union
from sys import getsizeof
import numpy as np

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.property_system import PropertyName


logger = get_logger(__name__, console_format_brief=True)


@dataclass
class LightFieldLineSet:
    """Lightweight version of FieldLineSet containing only necessary attributes."""
    n_field_lines: int
    reversed_error_warning: Optional[str] = None
    dotprod_error: Optional[str] = None


class FieldLineSet(FieldLine):
    """Collection of field lines with array values.

    FieldLineSet is the primary data container for field line tracing results.
    It inherits from FieldLine but specializes in storing bulk field line data.

    **Role in I/O System:**
    - **Data Container**: Holds actual field line data (coordinates, properties, etc.)
    - **FITS Storage**: All field line data is stored in FITS format via FieldLineSet
    - **Bulk Operations**: Efficiently handles large collections of field lines

    **Relationship to FieldLine:**
    - FieldLineSet uses FieldLine objects as templates to define data structure
    - Individual FieldLine objects can be data containers or configuration templates
    - FieldLineSet is always a data container (never a configuration template)

    **I/O Distinction:**
    - FieldLineSet → FITS files (scientific data)
    - FieldLine templates → YAML files (configuration)
    """

    # List of properties that are stored as lists of arrays
    ARRAY_LIST_PROPERTIES = [PropertyName.R_LINE, PropertyName.PA_LINE, PropertyName.B_LINE, PropertyName.CURVATURE_LINE, PropertyName.TANGENT_THETA, PropertyName.TANGENT_PHI]

    def __init__(
        self, field_lines: Union[List[FieldLine], FieldLine], copy_data: bool = True
    ) -> None:
        """Initialize a FieldLineSet from a list of field lines or a single field line."""
        logger.debug(f"Initializing FieldLineSet with {type(field_lines).__name__}")

        # Get template and validate input
        template = self._get_template(field_lines)

        # Use PropertySystem from parent class
        super().__init__(**template.property_system.get_flags_dict())

        self._n_field_lines = 0

        logger.debug(f"Initialized FieldLineSet with template: {self.save_flags}")
        if copy_data:
            logger.debug("Copying data from field lines")
            self.append(field_lines)
        else:
            logger.debug("Skipping data copy (copy_data=False)")

    @property
    def n_field_lines(self) -> int:
        """Get the number of field lines in the set."""
        return self._n_field_lines

    @n_field_lines.setter
    def n_field_lines(self, value: int) -> None:
        """Set the number of field lines in the set."""
        self._n_field_lines = value

    def copy(self) -> "FieldLineSet":
        """Create a deep copy of the FieldLineSet.
        
        Returns:
            FieldLineSet: A new FieldLineSet with copied data
        """
        logger.debug(f"Creating copy of FieldLineSet with {self.n_field_lines} lines")
        
        # Create new instance with same structure
        copy_instance = FieldLineSet(self, copy_data=False)
        
        # Copy all properties
        for prop in self.property_system.get_enabled_properties():
            source_data = self.property_system.get_property(prop)
            if source_data is not None:
                if prop in self.ARRAY_LIST_PROPERTIES:
                    # Deep copy list of arrays
                    copied_data = []
                    for item in source_data:
                        if item is not None:
                            copied_data.append(item.copy())
                        else:
                            copied_data.append(None)
                    copy_instance.property_system._properties[prop] = copied_data
                else:
                    # Deep copy numpy arrays
                    copy_instance.property_system._properties[prop] = source_data.copy()
        
        # Set the field line count
        copy_instance._n_field_lines = self._n_field_lines
        
        logger.debug(f"Created copy with {copy_instance.n_field_lines} lines")
        return copy_instance

    def _get_template(
        self, field_lines: Union[List[FieldLine], FieldLine]
    ) -> FieldLine:
        """Get template field line and validate input."""
        if isinstance(field_lines, list):
            if not field_lines:
                error_msg = "Empty field_lines list provided"
                logger.error(error_msg)
                raise ValueError(error_msg)
            return field_lines[0]
        elif isinstance(field_lines, FieldLine):
            return field_lines
        else:
            error_msg = f"Expected FieldLine or list of FieldLine, got {type(field_lines).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def append(
        self, field_lines: Union[List[FieldLine], FieldLine, "FieldLineSet"]
    ) -> None:
        """Append field lines to the set.

        Raises
        ------
        ValueError
            If field lines have incompatible save flags
        """
        logger.debug(f"Appending {type(field_lines).__name__} to FieldLineSet")

        # Handle FieldLineSet
        if isinstance(field_lines, FieldLineSet):
            self._append_field_line_set(field_lines)
            return

        # Convert single FieldLine to list
        if isinstance(field_lines, FieldLine):
            field_lines = [field_lines]
            logger.debug("Converted single FieldLine to list")

        # Handle list of FieldLine objects
        if isinstance(field_lines, list):
            self._append_field_line_list(field_lines)
        else:
            error_msg = f"Unexpected type: {type(field_lines).__name__}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _append_field_line_set(self, field_line_set: "FieldLineSet") -> None:
        """Append another FieldLineSet."""
        # Validate save flags compatibility using PropertySystem
        if (
            self.property_system.get_flags_dict()
            != field_line_set.property_system.get_flags_dict()
        ):
            error_msg = "FieldLineSet has incompatible save flags configuration"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(
            f"Appending FieldLineSet with {field_line_set.n_field_lines} lines"
        )

        # Append each property directly
        for prop in self.property_system.get_enabled_properties():
            source_data = field_line_set.property_system.get_property(prop)
            if source_data is None:
                logger.debug(f"Source data for {prop.key_name} is None, skipping")
                continue

            # Special handling for properties that are lists of arrays
            if prop in self.ARRAY_LIST_PROPERTIES:
                current_data = self.property_system.get_property(prop)
                if current_data is None:
                    logger.debug(f"Initializing {prop.key_name} with source data")
                    # Set property directly to avoid validation issues
                    self.property_system._properties[prop] = source_data.copy()
                else:
                    logger.debug(
                        f"Extending {prop.key_name} with {len(source_data)} arrays"
                    )
                    current_data.extend(source_data)
                continue

            # For other properties, use numpy append
            current_data = self.property_system.get_property(prop)
            if current_data is None:
                logger.debug(f"Initializing {prop.key_name} with source data")
                # Set property directly to avoid validation issues with arrays
                self.property_system._properties[prop] = np.array(source_data)
            else:
                logger.debug(
                    f"Appending to {prop.key_name}, adding {len(source_data)} items"
                )
                # Filter out None values before appending
                valid_source_data = [x for x in source_data if x is not None]
                if valid_source_data:
                    # Set property directly to avoid validation issues with arrays
                    self.property_system._properties[prop] = np.append(
                        current_data, valid_source_data, axis=0
                    )

        self.n_field_lines += field_line_set.n_field_lines
        logger.debug(
            f"Appended FieldLineSet with {field_line_set.n_field_lines} lines, total now: {self.n_field_lines}"
        )

    def _append_field_line_list(self, field_lines: List[FieldLine]) -> None:
        """Append a list of FieldLine objects."""
        # Validate save flags compatibility using PropertySystem
        for i, line in enumerate(field_lines):
            if (
                self.property_system.get_flags_dict()
                != line.property_system.get_flags_dict()
            ):
                error_msg = f"Field line {i} has incompatible save flags configuration"
                logger.error(error_msg)
                raise ValueError(
                    "All field lines must have the same save flags configuration"
                )

        logger.debug(f"Appending {len(field_lines)} individual field lines")

        for prop in self.property_system.get_enabled_properties():
            values = [line.property_system.get_property(prop) for line in field_lines]
            logger.debug(f"Collected {len(values)} values for {prop.key_name}")

            # Special handling for properties that are lists of arrays
            if prop in self.ARRAY_LIST_PROPERTIES:
                current_data = self.property_system.get_property(prop)
                if current_data is None:
                    logger.debug(f"Initializing {prop.key_name} with collected values")
                    # Set property directly to avoid validation issues
                    self.property_system._properties[prop] = values
                else:
                    logger.debug(f"Extending {prop.key_name} with {len(values)} arrays")
                    current_data.extend(values)
                continue

            # For other properties, convert to numpy array
            current_data = self.property_system.get_property(prop)
            if current_data is None:
                logger.debug(f"Initializing {prop.key_name} with numpy array")
                # Filter out None values before creating array
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    # Set property directly to avoid validation issues with arrays
                    self.property_system._properties[prop] = np.array(valid_values)
            else:
                logger.debug(f"Appending to {prop.key_name} numpy array")
                # Filter out None values before appending
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    # Set property directly to avoid validation issues with arrays
                    self.property_system._properties[prop] = np.append(
                        current_data, valid_values, axis=0
                    )

        self.n_field_lines += len(field_lines)
        logger.debug(
            f"Appended {len(field_lines)} field lines, total now: {self.n_field_lines}"
        )

    def estimate_size(self) -> float:
        """Estimate the memory size of the FieldLineSet in bytes."""
        logger.debug("Estimating memory size of FieldLineSet")
        total_bytes = 0

        # Get size of each enabled property using PropertySystem
        for prop in self.property_system.get_enabled_properties():
            values = self.property_system.get_property(prop)
            if values is None:
                logger.debug(f"Property {prop.key_name} is None, skipping")
                continue

            # Special case for properties that are lists of arrays
            if prop in self.ARRAY_LIST_PROPERTIES:
                prop_size = sum(
                    r.nbytes for r in values if r is not None and hasattr(r, "nbytes")
                )
                logger.debug(
                    f"{prop.key_name} size: {prop_size} bytes from {len(values)} arrays"
                )
            else:
                # For numpy arrays, use nbytes property
                if isinstance(values, np.ndarray):
                    prop_size = values.nbytes
                    logger.debug(
                        f"{prop.key_name} size: {prop_size} bytes (numpy array)"
                    )
                # For scalar values, use getsizeof
                else:
                    prop_size = getsizeof(values)
                    logger.debug(f"{prop.key_name} size: {prop_size} bytes (scalar)")

            total_bytes += prop_size

        logger.debug(
            f"Total estimated size: {total_bytes} bytes ({total_bytes/1024/1024:.2f} MB)"
        )
        return float(total_bytes)

    def print_summary(self) -> None:
        """Print summary statistics for all enabled properties."""
        print(f"FieldLineSet Summary:")
        print(f"  Number of field lines: {self.n_field_lines}")
        print(f"  Memory usage: {self.estimate_size()/1024/1024:.2f} MB")
        print(
            f"  Enabled properties: {len(self.property_system.get_enabled_properties())}"
        )
        print()

        for prop in self.property_system.get_enabled_properties():
            values = self.property_system.get_property(prop)
            if values is None:
                print(f"  {prop.key_name}: None")
                continue

            # Handle properties that are lists of arrays
            if prop in self.ARRAY_LIST_PROPERTIES:
                if not values:
                    print(f"  {prop.key_name}: Empty list")
                    continue

                # Calculate statistics across all arrays
                all_values = []
                for arr in values:
                    if arr is not None and len(arr) > 0:
                        all_values.extend(arr.flatten())

                if all_values:
                    all_values = np.array(all_values)
                    print(
                        f"  {prop.key_name}:\tArrays: {len(values)} Total points: {len(all_values)}\t Min: {all_values.min():.3e}\t Max: {all_values.max():.3e}"
                    )
                else:
                    print(f"  {prop.key_name}: No valid data")
                continue

            # Handle numpy arrays
            if isinstance(values, np.ndarray):
                if values.size == 0:
                    print(f"  {prop.key_name}: Empty array")
                    continue

                print(
                    f"  {prop.key_name}:\t Min: {values.min():.2e}\t Max: {values.max():.2e}"
                )
            else:
                # Handle scalar values
                print(f"  {prop.key_name}: {values}")

        print()

    def get_property(
        self, property_name: Union[str, PropertyName]
    ) -> Union[np.ndarray, List, None]:
        """Get property value with enabled check."""
        # Convert string to PropertyName if needed
        if isinstance(property_name, str):
            try:
                # First try using the string as an enum name (e.g., 'R_MAX')
                property_name = PropertyName[property_name.upper()]
            except KeyError:
                # If that fails, try to find by key_name (e.g., 'r_max')
                for prop in PropertyName:
                    if prop.key_name == property_name.lower():
                        property_name = prop
                        break
                else:
                    logger.warning(f"Unknown property name: {property_name}")
                    return None

        # Check if property is enabled
        if property_name not in self.property_system.get_enabled_properties():
            logger.debug(f"Property {property_name.key_name} is not enabled")
            return None

        return self.property_system.get_property(property_name)

    def __getattr__(self, name: str):
        """Dynamic property access for enabled properties."""
        # Try to get property if it exists and is enabled
        try:
            # First try using the name as an enum name (e.g., 'R_MAX')
            prop_name = PropertyName[name.upper()]
        except KeyError:
            # If that fails, try to find by key_name (e.g., 'r_max')
            prop_name = None
            for prop in PropertyName:
                if prop.key_name == name.lower():
                    prop_name = prop
                    break

        if prop_name and prop_name in self.property_system.get_enabled_properties():
            return self.property_system.get_property(prop_name)

        # Fallback to normal attribute error
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __repr__(self) -> str:
        """Return string representation of the field line set."""
        enabled_props = self.property_system.get_enabled_property_names()
        return f"FieldLineSet(n_lines={self.n_field_lines}, enabled_properties={enabled_props})"

    def get_OFL_footpoints(self, R_LC: float, h_LC_coeff: float = 3**1.5 / 2.0) -> np.ndarray:
        """
        Get the footpoints of the Open Field Lines (OFL).
        
        Args:
            R_LC: Light cylinder radius
            h_LC_coeff: Height coefficient for LC (default: 3^1.5 / 2.0)
        
        Returns:
            np.ndarray: Array of OFL footpoint indices
        """
        R_cyl_max = self.r_cyl_max
        z_abs_max = self.z_abs_max
        footpoint_ids = self.footpoint_id
        
        # OFL criteria: either beyond LC or within LC but high enough
        open_fl_crit = (R_cyl_max > R_LC) | (
            (R_cyl_max < R_LC) & (np.abs(z_abs_max) > (h_LC_coeff * R_LC))
        )
        
        logger.info(f"R_LC: {R_LC:.2f}")
        logger.info(f"OFL criteria met: {np.sum(open_fl_crit)} out of {len(open_fl_crit)}")
        
        # Return OFL footpoint indices
        ofl_footpoint_ids = footpoint_ids[open_fl_crit].astype(int)
        
  
        return ofl_footpoint_ids
    
    def subset_by_footpoint_ids(self, footpoint_ids: List[int]) -> "FieldLineSet":
        """
        Create a subset of the FieldLineSet containing only field lines with the specified footpoint IDs.
        
        Args:
            footpoint_ids: List of footpoint IDs to include in the subset
            
        Returns:
            FieldLineSet: New FieldLineSet containing only the specified field lines
            
        Raises:
            ValueError: If footpoint_id property is not enabled or footpoint_ids is empty
        """
        # Convert numpy array to list if needed
        if isinstance(footpoint_ids, np.ndarray):
            footpoint_ids = footpoint_ids.tolist()
            
        if not footpoint_ids:
            error_msg = "footpoint_ids list cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Check if footpoint_id property is enabled
        if PropertyName.FOOTPOINT_ID not in self.property_system.get_enabled_properties():
            error_msg = "footpoint_id property must be enabled to subset by footpoint IDs"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Get current footpoint IDs
        current_footpoint_ids = self.property_system.get_property(PropertyName.FOOTPOINT_ID)
        if current_footpoint_ids is None:
            error_msg = "footpoint_id property is None"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Convert footpoint_ids to set for efficient lookup
        target_footpoint_set = set(footpoint_ids)
        
        # Find indices of field lines with matching footpoint IDs
        matching_indices = []
        for i, fp_id in enumerate(current_footpoint_ids):
            if fp_id in target_footpoint_set:
                matching_indices.append(i)
                
        if not matching_indices:
            logger.warning(f"No field lines found with the specified footpoint IDs: {footpoint_ids}")
            # Return empty FieldLineSet with same template
            empty_set = FieldLineSet(self, copy_data=False)
            empty_set.n_field_lines = 0
            return empty_set
            
        logger.info(f"Found {len(matching_indices)} field lines matching {len(footpoint_ids)} footpoint IDs")
        
        # Create new FieldLineSet with same template
        subset = FieldLineSet(self, copy_data=False)
        
        # Copy data for matching indices
        for prop in self.property_system.get_enabled_properties():
            source_data = self.property_system.get_property(prop)
            if source_data is None:
                continue
                
            # Special handling for properties that are lists of arrays
            if prop in self.ARRAY_LIST_PROPERTIES:
                subset_data = [source_data[i] for i in matching_indices if i < len(source_data)]
                subset.property_system._properties[prop] = subset_data
            else:
                # For numpy arrays, use advanced indexing
                if isinstance(source_data, np.ndarray):
                    subset_data = source_data[matching_indices]
                    subset.property_system._properties[prop] = subset_data
                else:
                    # For other types, use list comprehension
                    subset_data = [source_data[i] for i in matching_indices if i < len(source_data)]
                    subset.property_system._properties[prop] = np.array(subset_data)
                    
        # Update field line count
        subset.n_field_lines = len(matching_indices)
        
        logger.debug(f"Created subset with {subset.n_field_lines} field lines")
        return subset
    
        
    def trim_to_light_cylinder(self, R_LC, h_LC_coeff=3**1.5 / 2.0, trim_type="cylinder"):
        """
        Trim field lines to points before light cylinder penetration.
        
        Args:
            R_LC: Light cylinder radius
            h_LC_coeff: Height coefficient for cylinder trimming
            trim_type: "cylinder" or "sphere" trimming method
            
        Returns:
            None (modifies self in-place)
        """
        
        assert trim_type in ["cylinder", "sphere"], "trim_type should be cylinder or sphere"
        print("Trimming to light cylinder")
        # Check if r_line property is enabled
        if PropertyName.R_LINE not in self.property_system.get_enabled_properties():
            error_msg = "r_line property must be enabled to trim field lines"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        r_line_data = self.property_system.get_property(PropertyName.R_LINE)
        if r_line_data is None:
            error_msg = "r_line property is None"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Properties that should be trimmed along with r_line
        line_properties = [
            PropertyName.R_LINE,
            PropertyName.PA_LINE,
            PropertyName.B_LINE,
            PropertyName.CURVATURE_LINE,
            PropertyName.TANGENT_THETA,
            PropertyName.TANGENT_PHI
        ]
        
        # Calculate trim indices for each field line
        trim_indices = []
        for i, r_line in enumerate(r_line_data):
            if r_line is None or len(r_line) == 0:
                trim_indices.append(0)  # No points to keep
                continue
                
            # r_line is numpy array with shape (n_points, 3) where each row is [x, y, z]
            # Handle both 1D and 2D arrays
            if r_line.ndim == 1:
                # If 1D, assume it's a flattened array that needs reshaping
                if len(r_line) % 3 == 0:
                    r_line = r_line.reshape(-1, 3)
                else:
                    logger.warning(f"Field line {i} has invalid shape {r_line.shape}, skipping")
                    trim_indices.append(0)
                    continue
            elif r_line.ndim != 2 or r_line.shape[1] != 3:
                logger.warning(f"Field line {i} has unexpected shape {r_line.shape}, skipping")
                trim_indices.append(0)
                continue
                
            x, y, z = r_line[:, 0], r_line[:, 1], r_line[:, 2]
                
            # Calculate which points are outside light cylinder
            if trim_type == "cylinder":
                z_abs = np.abs(z)
                R_cyl = np.sqrt(x**2 + y**2)
                outside_LC = (R_cyl > R_LC) | ((R_cyl < R_LC) & (z_abs > (h_LC_coeff * R_LC)))
                
            elif trim_type == "sphere":
                R_sph = np.sqrt(x**2 + y**2 + z**2)
                outside_LC = R_sph > R_LC
                
            # Find first point outside LC
            outside_indices = np.where(outside_LC)[0]
            print("Outside indices: ", outside_indices)
            
            if len(outside_indices) > 0:
                first_LC_penetration_point = outside_indices[0]
                trim_indices.append(first_LC_penetration_point)
            else:
                # Keep all points if no LC penetration
                trim_indices.append(len(x))
                
        # Trim each line property
        for prop in line_properties:
            # Check if property is enabled and has data
            if prop in self.property_system.get_enabled_properties():
                prop_data = self.property_system.get_property(prop)
                if prop_data is None:
                    logger.debug(f"Property {prop} is enabled but has no data, skipping")
                    continue
                    
                # Trim each field line's data
                trimmed_data = []
                for i, (line_data, trim_idx) in enumerate(zip(prop_data, trim_indices)):
                    if line_data is None or len(line_data) == 0:
                        trimmed_data.append(line_data)
                    else:
                        print("Trim index: ", trim_idx)
                        # Trim to points before LC
                        if trim_idx > 0:
                            # Special handling for r_line (2D array)
                            if prop == PropertyName.R_LINE:
                                print("Line data: ", line_data.shape)
                                if line_data.ndim == 1:
                                    trimmed_data.append(line_data[:trim_idx*3])
                                else:
                                    trimmed_data.append(line_data[:trim_idx,:])
                                print("Trimmed line data: ", trimmed_data[-1].shape)
                            else:
                                # For other properties, use 1D slice
                                trimmed_data.append(line_data[:trim_idx])
                        else:
                            # No points to keep, use empty array
                            if isinstance(line_data, np.ndarray):
                                if prop == PropertyName.R_LINE:
                                    # For r_line, return empty 2D array
                                    trimmed_data.append(np.array([]).reshape(0, 3))
                                else:
                                    # For other properties, return empty 1D array
                                    trimmed_data.append(np.array([]))
                            else:
                                trimmed_data.append([])
                                
                # Update the property
                self.property_system._properties[prop] = trimmed_data
                print("Trimmed property: ", prop, "to length: ", len(trimmed_data))
            else:
                logger.debug(f"Property {prop} is not enabled, skipping")
                
        logger.info(f"Trimmed {self.n_field_lines} field lines to light cylinder (R_LC={R_LC})")
