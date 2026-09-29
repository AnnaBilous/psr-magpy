"""OFL map, skymap, and PA extract from the same series layout as series_disk.py."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series_runner import (
    DeletionStrategy,
    OverwriteStrategy,
    run_experiment_series,
)
from psr_magpy.experiment.grid_config import HealpixGridConfig
from psr_magpy.workflow.extract_PA import make_extract_PA
from psr_magpy.workflow.extract_ofl_map import extract_ofl_map
from psr_magpy.workflow.make_skymap import make_skymap

from tutorial_layout import SERIES_NAME, build_series


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="psr_magpy_post_"))
    series = build_series(out)
    run_experiment_series(
        series,
        series_deletion_strategy=DeletionStrategy.NONE,
        overwrite_strategy=OverwriteStrategy.ALL,
        save_grids=False,
    )
    r_lc = np.array([2.0])
    maps = extract_ofl_map(series, r_lc, save=True)
    experiment = ExperimentConfig.load(
        out / SERIES_NAME / "configs" / f"{SERIES_NAME}_000.yaml"
    )
    skymap = make_skymap(
        experiment_config=experiment,
        parent_dir=out,
        series_name=SERIES_NAME,
        line_source="lines",
        healpix_grid_config=HealpixGridConfig(nside=1),
        R_LC_values=r_lc,
        save=True,
    )
    pa = make_extract_PA(
        experiment_config=experiment,
        R_LC_values=r_lc,
        theta_tolerance=30.0,
        grid_theta=np.array([90.0]),
        parent_dir=out,
        series_name=SERIES_NAME,
        line_source="lines",
        save=True,
    )
    print(f"output={out}")
    print(f"ofl_experiments={sorted(maps)}")
    print(f"ofl_open={int(np.sum(~np.isnan(maps[0])))}")
    sky = skymap[2.0] if isinstance(skymap, dict) else skymap
    print(f"skymap_sum={float(np.sum(sky))}")
    pa_dir = out / SERIES_NAME / "data_products" / "extracted_PA"
    print(f"pa_files={[p.name for p in sorted(pa_dir.glob('*.fits'))]}")


if __name__ == "__main__":
    main()
