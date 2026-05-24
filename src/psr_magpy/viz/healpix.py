"""HEALPix array → equirectangular image projection for plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import healpy as hp
import numpy as np
from astropy.io import fits

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__)

ImageSize = Tuple[int, int]


def equirectangular_coords(
    image_size: ImageSize = (512, 512),
) -> Tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude coordinate vectors (degrees) for equirectangular maps.

    Parameters
    ----------
    image_size
        ``(height, width)`` of the output image.

    Returns
    -------
    longitude_deg
        Length-``width`` array, degrees.
    latitude_deg
        Length-``height`` array, degrees (north-positive, decreasing row index).
    """
    image_height, image_width = image_size
    colatitude = np.linspace(0.0, np.pi, image_height)
    azimuth = np.linspace(0.0, 2.0 * np.pi, image_width)
    longitude_deg = np.rad2deg(azimuth)
    latitude_deg = (np.rad2deg(colatitude) - 90.0)[::-1]
    return longitude_deg, latitude_deg


def project_array_to_equirectangular(
    data: np.ndarray,
    grid_indices: np.ndarray,
    nside: int,
    image_size: ImageSize = (512, 512),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project footpoint- or pixel-indexed HEALPix data to an equirectangular image.

    ``grid_indices[healpix_pixel]`` gives the index into ``data`` for that pixel
    (typically ``np.arange(npix)`` for a dense HEALPix table).

    Parameters
    ----------
    data
        Values indexed by ``grid_indices`` (length ≥ ``max(grid_indices)+1``).
    grid_indices
        Length-``12 * nside**2`` mapping from HEALPix pixel ID to ``data`` index.
    nside
        HEALPix resolution parameter.
    image_size
        Output ``(height, width)``.

    Returns
    -------
    image, longitude_deg, latitude_deg
        ``image`` is 2D; coordinate vectors match :func:`equirectangular_coords`.
    """
    data = np.asarray(data)
    grid_indices = np.asarray(grid_indices)
    nside = int(nside)
    image_height, image_width = image_size
    npix_expected = hp.nside2npix(nside)

    if grid_indices.size != npix_expected:
        error_msg = (
            f"grid_indices length {grid_indices.size} != 12*nside^2 ({npix_expected}) "
            f"for nside={nside}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    colatitude = np.linspace(0.0, np.pi, image_height)
    azimuth = np.linspace(0.0, 2.0 * np.pi, image_width)
    azimuth_mesh, colatitude_mesh = np.meshgrid(azimuth, colatitude)
    healpix_pixels = hp.pixelfunc.ang2pix(nside, colatitude_mesh, azimuth_mesh)

    data_indices = grid_indices[healpix_pixels]
    image = data[data_indices]

    longitude_deg, latitude_deg = equirectangular_coords(image_size)
    return image, longitude_deg, latitude_deg


# Alias for migration from Multipole_RVM analysis scripts.
project_to_equirectangular = project_array_to_equirectangular


def project_skymap_fits(
    fits_path: Union[str, Path],
    image_size: ImageSize = (1024, 1024),
    *,
    skymap_column_prefix: str = "SKYMAP",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a skymap column from FITS and project to equirectangular coordinates.

    Expects primary HDU header ``NSIDE`` / ``NPIX`` and a binary table HDU with at
    least one column whose name starts with ``skymap_column_prefix``.

    Parameters
    ----------
    fits_path
        Path to skymap FITS file.
    image_size
        Output ``(height, width)``.
    skymap_column_prefix
        Column name prefix for skymap data (default ``SKYMAP``).

    Returns
    -------
    image, longitude_deg, latitude_deg
        Same as :func:`project_array_to_equirectangular`.
    """
    fits_path = Path(fits_path)
    if not fits_path.is_file():
        error_msg = f"Skymap FITS not found: {fits_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        nside = int(header["NSIDE"])
        npix = int(header["NPIX"])
        table_data = hdul[1].data
        skymap_columns = [
            col.name
            for col in hdul[1].columns
            if col.name.startswith(skymap_column_prefix)
        ]
        if not skymap_columns:
            error_msg = (
                f"No columns starting with {skymap_column_prefix!r} in {fits_path}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        skymap = np.asarray(table_data[skymap_columns[0]])
        if skymap.size != npix:
            error_msg = (
                f"Skymap column length {skymap.size} != NPIX {npix} in {fits_path}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    grid_indices = np.arange(npix, dtype=np.int64)
    return project_array_to_equirectangular(
        skymap, grid_indices, nside, image_size=image_size
    )
