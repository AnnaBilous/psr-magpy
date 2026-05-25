"""Paths and filename patterns for per-series ``data_products`` trees."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple, Type, Union

import numpy as np
from astropy.io import fits

from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DataProductSpec:
    """Location and filename pattern for one data-product kind.

    Parameters
    ----------
    subdir
        Subdirectory under ``<series>/data_products/``.
    filename_pattern
        :func:`format_data_product_filename` pattern; ignored when ``is_directory`` is True.
    requires_r_lc
        If True, ``r_lc`` must be supplied to :func:`get_data_path`.
    is_directory
        If True, return the product directory (no FITS file).
    """

    subdir: str
    filename_pattern: str = ""
    requires_r_lc: bool = False
    is_directory: bool = False


def resolve_data_products_directory(
    parent_dir: Union[str, Path],
    series_name: str,
    subdir: str,
) -> Path:
    """Return ``<parent_dir>/<series_name>/data_products/<subdir>``."""
    return Path(parent_dir).expanduser().resolve() / series_name / "data_products" / subdir


def format_data_product_filename(
    pattern: str,
    *,
    experiment_number: int,
    r_lc: Optional[float] = None,
    experiment_name: Optional[str] = None,
    series_name: Optional[str] = None,
    nside: Optional[int] = None,
    **extra_fields: object,
) -> str:
    """Build a data-product basename from a :meth:`str.format` pattern.

    Parameters
    ----------
    pattern
        Format string with at least ``experiment_number`` (often ``{experiment_number:03d}``).
        Include ``r_lc`` in the pattern when ``r_lc`` is passed.
    experiment_number
        0-based experiment index.
    r_lc, experiment_name, series_name, nside
        Optional fields for the pattern.
    **extra_fields
        Additional ``str.format`` keys.

    Returns
    -------
    str
        Filename; ``.fits`` is appended when the pattern has no FITS extension.
    """
    fields: dict = {"experiment_number": int(experiment_number)}
    if r_lc is not None:
        fields["r_lc"] = float(r_lc)
    if experiment_name is not None:
        fields["experiment_name"] = experiment_name
    if series_name is not None:
        fields["series_name"] = series_name
    if nside is not None:
        fields["nside"] = int(nside)
    fields.update(extra_fields)
    try:
        name = pattern.format(**fields)
    except KeyError as exc:
        error_msg = f"Pattern {pattern!r} missing field for format: {exc}"
        logger.error(error_msg)
        raise ValueError(error_msg) from exc
    lower = name.lower()
    if not (lower.endswith(".fits") or lower.endswith(".fit")):
        name = f"{name}.fits"
    return name


def resolve_data_product_path(
    parent_dir: Union[str, Path],
    series_name: str,
    subdir: str,
    filename_pattern: str,
    experiment_number: int,
    *,
    r_lc: Optional[float] = None,
    **format_fields: object,
) -> Path:
    """Resolve a single data-product FITS path under ``data_products/<subdir>``.

    Parameters
    ----------
    parent_dir
        Series root parent (heavy-computing data root).
    series_name
        Experiment series directory name.
    subdir
        Product subdirectory (e.g. ``skymaps``, ``OFLR_maps``, ``extracted_PA``).
    filename_pattern
        :func:`format_data_product_filename` pattern (project-specific).
    experiment_number
        Experiment index.
    r_lc
        Light-cylinder radius when required by the pattern.
    **format_fields
        Extra keys forwarded to :func:`format_data_product_filename`.

    Returns
    -------
    Path
        Full path to the product file.
    """
    product_dir = resolve_data_products_directory(parent_dir, series_name, subdir)
    filename = format_data_product_filename(
        filename_pattern,
        experiment_number=experiment_number,
        r_lc=r_lc,
        series_name=series_name,
        **format_fields,
    )
    return product_dir / filename


def get_data_path(
    parent_dir: Union[str, Path],
    series_name: str,
    data_type: str,
    experiment_number: int,
    products: Mapping[str, DataProductSpec],
    *,
    r_lc: Optional[float] = None,
    **format_fields: object,
) -> Path:
    """Resolve a data-product path from a named registry (project-specific patterns).

    Parameters
    ----------
    parent_dir
        Heavy-computing or series root parent directory.
    series_name
        Experiment series directory name.
    data_type
        Key into ``products`` (e.g. ``\"skymap\"``, ``\"OFLR_map\"``).
    experiment_number
        0-based experiment index.
    products
        Map from ``data_type`` strings to :class:`DataProductSpec`.
    r_lc
        Light-cylinder radius when required by the spec or pattern.
    **format_fields
        Extra keys for :func:`format_data_product_filename`.

    Returns
    -------
    Path
        FITS file path, or product directory when the spec is directory-only.
    """
    if data_type not in products:
        error_msg = (
            f"Unknown data_type {data_type!r}; known: {sorted(products.keys())}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    spec = products[data_type]
    if spec.is_directory:
        return resolve_data_products_directory(parent_dir, series_name, spec.subdir)
    if spec.requires_r_lc and r_lc is None:
        error_msg = f"R_LC is required for data_type {data_type!r}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return resolve_data_product_path(
        parent_dir,
        series_name,
        spec.subdir,
        spec.filename_pattern,
        experiment_number,
        r_lc=r_lc,
        **format_fields,
    )


def load_footpoint_map_fits(
    map_path: Union[str, Path],
    *,
    column_name: Optional[str] = None,
    hdu_index: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a footpoint-indexed map from a FITS binary table.

    Reads one table column (first column by default) and sets
    ``grid_indices = np.arange(len(data))`` for HEALPix pixel order.

    Parameters
    ----------
    map_path
        Path to the FITS file.
    column_name
        Table column name; default is the first column in the HDU.
    hdu_index
        HDU index for the binary table (default 1).

    Returns
    -------
    data, grid_indices
        Footpoint-indexed values and index array.
    """
    map_path = Path(map_path)
    if not map_path.is_file():
        error_msg = f"Map FITS not found: {map_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with fits.open(map_path) as hdul:
        table_data = hdul[hdu_index].data
        if column_name is None:
            column_name = table_data.columns[0].name
        data = np.asarray(table_data[column_name])
        grid_indices = np.arange(len(data), dtype=np.int64)

    return data, grid_indices


def load_footpoint_map_product(
    parent_dir: Union[str, Path],
    series_name: str,
    data_type: str,
    experiment_number: int,
    products: Mapping[str, DataProductSpec],
    *,
    r_lc: Optional[float] = None,
    column_name: Optional[str] = None,
    **format_fields: object,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve a data-product path and load a footpoint map FITS table."""
    map_path = get_data_path(
        parent_dir,
        series_name,
        data_type,
        experiment_number,
        products,
        r_lc=r_lc,
        **format_fields,
    )
    return load_footpoint_map_fits(map_path, column_name=column_name)


def load_extracted_pa_fits(
    pa_path: Union[str, Path],
    *,
    field_line_set_type: Type[FieldLineSet] = FieldLineSet,
) -> FieldLineSet:
    """Load a :class:`~psr_magpy.line.field_line_set.FieldLineSet` from extracted PA FITS.

    Loads batched files (``{stem}_batch_*.fits``) from the same directory.
    Falls back to a legacy single file if no batches are found.
    """
    from psr_magpy.io import load_data
    from psr_magpy.experiment.services.batch_data_reader import BatchDataReader
    from psr_magpy.line.field_line import FieldLine
    from psr_magpy.workflow.extract_PA import _PA_SAVE_TEMPLATE

    pa_path = Path(pa_path)
    template = FieldLine.from_template(_PA_SAVE_TEMPLATE)
    reader = BatchDataReader(experiment_name=pa_path.stem)
    combined = reader.load_combined_batch_data(
        template=template,
        lines_dir=str(pa_path.parent),
        batch_filename_prefix=pa_path.stem,
    )
    if combined is not None and combined.n_field_lines > 0:
        logger.info(
            "Loaded %d field lines from batched files (%s_batch_*.fits)",
            combined.n_field_lines,
            pa_path.stem,
        )
        return combined

    # Legacy single-file fallback
    if pa_path.is_file():
        logger.info("No batched files found, loading legacy single file: %s", pa_path)
        field_line_set = load_data(field_line_set_type, str(pa_path))
        if field_line_set is not None:
            logger.info(
                "Loaded %d field lines from %s",
                field_line_set.n_field_lines,
                pa_path,
            )
            return field_line_set

    error_msg = f"No PA data found (batched or single file): {pa_path}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def load_extracted_pa_product(
    parent_dir: Union[str, Path],
    series_name: str,
    experiment_number: int,
    products: Mapping[str, DataProductSpec],
    *,
    r_lc: float,
    data_type: str = "PA",
    **format_fields: object,
) -> FieldLineSet:
    """Resolve extracted-PA path and load :class:`FieldLineSet`."""
    pa_path = get_data_path(
        parent_dir,
        series_name,
        data_type,
        experiment_number,
        products,
        r_lc=float(r_lc),
        **format_fields,
    )
    return load_extracted_pa_fits(pa_path)
