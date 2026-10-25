"""Tests for data product path helpers."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from psr_magpy.io.data_products import (
    DataProductSpec,
    format_data_product_filename,
    get_data_path,
    load_footpoint_map_fits,
    resolve_data_product_path,
    resolve_data_products_directory,
)

_PRODUCTS = {
    "skymap": DataProductSpec("skymaps", "skymap_{experiment_number:06d}_RLC_{r_lc:05.2f}", requires_r_lc=True),
    "edges": DataProductSpec("OFLR_edges", is_directory=True),
}


def test_resolve_data_products_directory(tmp_path):
    p = resolve_data_products_directory(tmp_path, "my_series", "skymaps")
    assert p == tmp_path / "my_series" / "data_products" / "skymaps"


def test_format_data_product_filename_appends_fits():
    name = format_data_product_filename(
        "skymap_{experiment_number:06d}_RLC_{r_lc:05.2f}",
        experiment_number=3,
        r_lc=4.77,
    )
    assert name == "skymap_000003_RLC_04.77.fits"


def test_format_data_product_filename_keeps_existing_extension():
    name = format_data_product_filename(
        "OFLR_map_{experiment_number:06d}.fits",
        experiment_number=0,
    )
    assert name == "OFLR_map_000000.fits"


def test_resolve_data_product_path(tmp_path):
    path = resolve_data_product_path(
        tmp_path,
        "series_a",
        "extracted_PA",
        "extract_PA_{experiment_number:06d}_RLC_{r_lc:05.2f}",
        12,
        r_lc=23.8,
    )
    assert path == (
        tmp_path
        / "series_a"
        / "data_products"
        / "extracted_PA"
        / "extract_PA_000012_RLC_23.80.fits"
    )
    assert isinstance(path, Path)


def test_get_data_path_file_and_directory(tmp_path):
    skymap = get_data_path(
        tmp_path, "series_a", "skymap", 0, _PRODUCTS, r_lc=4.77
    )
    assert skymap.name == "skymap_000000_RLC_04.77.fits"
    edges = get_data_path(tmp_path, "series_a", "edges", 0, _PRODUCTS)
    assert edges == tmp_path / "series_a" / "data_products" / "OFLR_edges"


def test_get_data_path_requires_r_lc():
    import pytest

    with pytest.raises(ValueError, match="R_LC is required"):
        get_data_path("/tmp", "s", "skymap", 0, _PRODUCTS)


def test_load_footpoint_map_fits(tmp_path):
    values = np.array([1.0, 2.0, 3.0])
    col = fits.Column(name="OFLR_MAP", format="D", array=values)
    table_hdu = fits.BinTableHDU.from_columns([col])
    table_hdu.header["NPIX"] = 3
    path = tmp_path / "map.fits"
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(path, overwrite=True)

    data, grid_indices = load_footpoint_map_fits(path)
    np.testing.assert_array_equal(data, values)
    np.testing.assert_array_equal(grid_indices, np.arange(3))
