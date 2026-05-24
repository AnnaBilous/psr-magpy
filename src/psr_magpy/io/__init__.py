"""IO utilities for psr_magpy."""

from .data_products import (
    DataProductSpec,
    format_data_product_filename,
    get_data_path,
    load_extracted_pa_fits,
    load_extracted_pa_product,
    load_footpoint_map_fits,
    load_footpoint_map_product,
    resolve_data_product_path,
    resolve_data_products_directory,
)
from .dispatch import (
    save_data,
    load_data,
    save_config,
    load_config,
    detect_format,
)

__all__ = [
    "save_data",
    "load_data",
    "save_config",
    "load_config",
    "detect_format",
    "DataProductSpec",
    "format_data_product_filename",
    "get_data_path",
    "load_extracted_pa_fits",
    "load_extracted_pa_product",
    "load_footpoint_map_fits",
    "load_footpoint_map_product",
    "resolve_data_product_path",
    "resolve_data_products_directory",
]
