"""Disk series: expand two latitudes, save YAML, load it, run the series."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_series_runner import (
    DeletionStrategy,
    OverwriteStrategy,
    run_experiment_series,
)
from psr_magpy.experiment.parameter_expansion import get_parameter_combinations

from tutorial_layout import SERIES_NAME, build_series


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="psr_magpy_series_"))
    combos = get_parameter_combinations({"nlon": 4, "nlat": [3, 4]})
    print(f"combinations={len(combos)}")

    series = build_series(out)
    yaml_path = out / f"{SERIES_NAME}_config.yaml"
    series.save(str(yaml_path))
    loaded = ExperimentSeriesConfig.load(yaml_path)
    print(f"yaml={yaml_path} name={loaded.name}")

    result = run_experiment_series(
        loaded,
        series_deletion_strategy=DeletionStrategy.NONE,
        overwrite_strategy=OverwriteStrategy.ALL,
        save_grids=False,
    )
    print(f"output={out}")
    print(f"lines={out / SERIES_NAME / 'lines'}")
    print(
        f"success={result.successful_experiments}/{result.total_experiments}"
    )


if __name__ == "__main__":
    main()
