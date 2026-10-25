"""OFL regions, edges, a fine subset, and a snapshot load from one HEALPix series."""

import tempfile
from pathlib import Path

import numpy as np

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_series_runner import (
    OverwriteStrategy,
    run_experiment_series,
)
from psr_magpy.experiment.field_config import FieldComponentConfig, FieldConfig
from psr_magpy.experiment.grid_config import GridConfig, HealpixGridConfig
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.io.data_products import DataProductSpec
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.open_field_line_edges import (
    calculate_and_save_OFL_edges,
    calculate_and_save_OFL_regions,
)
from psr_magpy.viz.snapshot import load_snapshot_data
from psr_magpy.workflow.coarse_to_fine import create_fine_subset_workflow
from psr_magpy.workflow.extract_PA import make_extract_PA
from psr_magpy.workflow.extract_ofl_map import (
    extract_ofl_map,
    find_OFL_regions_from_batch_files,
)
from psr_magpy.workflow.line_batch_workflow_prep import (
    get_workflow_batch_load_template,
    merge_field_line_templates,
)
from psr_magpy.workflow.make_skymap import make_skymap

SERIES_NAME = "healpix_series"


def _series(output_dir: Path, name: str) -> ExperimentSeriesConfig:
    return ExperimentSeriesConfig(
        name=name,
        output_dir=str(output_dir),
        field=FieldConfig(
            components=[
                FieldComponentConfig(
                    moment_type="dipole",
                    B_magnitude=1.0,
                    moment_orientation_latitude_degree=30.0,
                ),
                FieldComponentConfig(moment_type="quadrupole", B_magnitude=0.5),
            ]
        ),
        grid=GridConfig(healpix=HealpixGridConfig(nside=1)),
        tracing=TracingConfig(
            max_integration_radius=20.0,
            save_results_to_disk=True,
            filename=name,
            max_memory_mb=100.0,
            skip_memory_estimation=True,
            max_step=1.0,
            verify_by_reversing=True,
        ),
        template=merge_field_line_templates(
            FieldLineTemplate.template_OFL_calc(),
            get_workflow_batch_load_template(),
        ),
    )


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="psr_magpy_products_"))
    series = _series(out, SERIES_NAME)
    run_experiment_series(series, overwrite_strategy=OverwriteStrategy.ALL, save_grids=False)
    r_lc = np.array([2.0])
    experiment = ExperimentConfig.load(
        out / SERIES_NAME / "configs" / f"{SERIES_NAME}_000000.yaml"
    )
    lines = out / SERIES_NAME / "lines"
    regions = find_OFL_regions_from_batch_files(
        experiment, lines, r_lc, save=True, output_dir=out / SERIES_NAME / "data_products" / "OFLR_maps"
    )
    template = FieldLine.from_template(get_workflow_batch_load_template())
    edges = calculate_and_save_OFL_edges(
        out / SERIES_NAME / "configs" / f"{SERIES_NAME}_000000.yaml",
        lines,
        template,
        r_lc,
        output_dir=out / SERIES_NAME / "data_products" / "OFLR_edges",
    )
    ofl_regions = calculate_and_save_OFL_regions(
        out / SERIES_NAME / "configs" / f"{SERIES_NAME}_000000.yaml",
        lines,
        template,
        r_lc,
        output_dir=out / SERIES_NAME / "data_products" / "OFLR",
    )
    extract_ofl_map(series, r_lc, save=True)
    make_skymap(
        experiment_config=experiment,
        parent_dir=out,
        series_name=SERIES_NAME,
        line_source="lines",
        healpix_grid_config=HealpixGridConfig(nside=1),
        R_LC_values=r_lc,
        save=True,
    )
    make_extract_PA(
        experiment_config=experiment,
        R_LC_values=r_lc,
        theta_tolerance=30.0,
        grid_theta=np.array([90.0]),
        parent_dir=out,
        series_name=SERIES_NAME,
        line_source="lines",
        save=True,
    )
    fine = create_fine_subset_workflow(
        coarse_series_config=series,
        fine_series_config=_series(out, "healpix_fine"),
        R_LC=2.0,
    )
    products = {
        "OFLR_map": DataProductSpec(
            subdir="OFLR_maps",
            filename_pattern="OFLR_map_{experiment_number:06d}.fits",
        ),
        "PA": DataProductSpec(
            subdir="extracted_PA",
            filename_pattern="extract_PA_{experiment_number:06d}_RLC_{r_lc:05.2f}.fits",
            requires_r_lc=True,
        ),
    }
    snapshot = load_snapshot_data(
        experiment_number=0,
        r_lc=2.0,
        series_config=series,
        data_dir=out,
        products=products,
        nside_skymap=1,
        nside_oflr=1,
        image_size=(8, 4),
        skymap_filename_pattern="skymap_{experiment_number:06d}_RLC_{r_lc:05.2f}.fits",
    )
    print(f"output={out}")
    print(f"regions_finite={int(np.sum(~np.isnan(regions)))}")
    print(f"edges={len(edges.edge_footpoints)}")
    print(f"ofl_regions={len(ofl_regions.edge_footpoints)}")
    print(f"fine={fine.name}")
    print(f"snapshot_lines={snapshot.field_line_set.n_field_lines}")


if __name__ == "__main__":
    main()
