"""Same tiny series, kept in memory instead of written as FITS batches."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from psr_magpy.experiment.experiment_series_runner import run_experiment_series_in_memory

from tutorial_layout import build_series


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="psr_magpy_memory_"))
    series = build_series(out)
    series.tracing.save_results_to_disk = False
    result = run_experiment_series_in_memory(
        series,
        keep_results=True,
        max_memory_mb=200.0,
        save_grids=False,
    )
    print(f"output={out}")
    print(
        f"success={result.successful_experiments}/{result.total_experiments}"
    )
    print(f"kept={len(result.experiment_results)}")


if __name__ == "__main__":
    main()
