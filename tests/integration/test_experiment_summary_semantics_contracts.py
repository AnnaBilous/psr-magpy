"""Contract tests for summary merge semantics."""

from psr_magpy.experiment.experiment_result import ExperimentResult
from psr_magpy.experiment.infrastructure.summary_repository import SummaryRepository


def _touch_batch_files(tmp_path, series_name: str, experiment_name: str, count: int) -> None:
    lines_dir = tmp_path / series_name / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        filename = f"{experiment_name}_batch_{idx:03d}_out_of_{count:03d}.fits"
        (lines_dir / filename).write_text("", encoding="utf-8")


def test_first_run_summary_contains_single_result(tmp_path):
    repo = SummaryRepository()
    series_name = "summary_contract"
    output_dir = str(tmp_path)

    _touch_batch_files(tmp_path, series_name, "summary_contract_000", 2)
    merged = repo.merge_summary(
        output_dir=output_dir,
        series_name=series_name,
        experiment_results=[
            ExperimentResult(
                experiment_name="summary_contract_000",
                success=True,
                duration=5.0,
            )
        ],
    )
    repo.save_summary(output_dir, series_name, merged)
    loaded = repo.load_summary(output_dir, series_name)

    assert loaded["total_experiments"] == 1
    assert loaded["successful_experiments"] == 1
    assert loaded["failed_experiments"] == 0
    assert loaded["success_rate"] == 100.0
    assert len(loaded["experiment_results"]) == 1
    assert loaded["experiment_results"][0]["name"] == "summary_contract_000"
    assert loaded["experiment_results"][0]["batch_files"] == 2


def test_rerun_updates_existing_experiment_entry(tmp_path):
    repo = SummaryRepository()
    series_name = "summary_contract"
    output_dir = str(tmp_path)

    _touch_batch_files(tmp_path, series_name, "summary_contract_000", 1)
    first = repo.merge_summary(
        output_dir=output_dir,
        series_name=series_name,
        experiment_results=[
            ExperimentResult(
                experiment_name="summary_contract_000",
                success=False,
                duration=2.0,
                error_message="initial error",
            )
        ],
    )
    repo.save_summary(output_dir, series_name, first)

    _touch_batch_files(tmp_path, series_name, "summary_contract_000", 3)
    rerun = repo.merge_summary(
        output_dir=output_dir,
        series_name=series_name,
        experiment_results=[
            ExperimentResult(
                experiment_name="summary_contract_000",
                success=True,
                duration=4.0,
            )
        ],
    )
    repo.save_summary(output_dir, series_name, rerun)
    loaded = repo.load_summary(output_dir, series_name)

    assert loaded["total_experiments"] == 1
    assert loaded["successful_experiments"] == 1
    assert loaded["failed_experiments"] == 0
    assert loaded["experiment_results"][0]["name"] == "summary_contract_000"
    assert loaded["experiment_results"][0]["success"] is True
    # Current contract: count all matching batch files present on disk.
    # Rerun artifacts accumulate unless old files are explicitly deleted.
    assert loaded["experiment_results"][0]["batch_files"] == 4
    assert loaded["experiment_results"][0]["duration"] == "4.0s"


def test_partial_rerun_preserves_other_experiments(tmp_path):
    repo = SummaryRepository()
    series_name = "summary_contract"
    output_dir = str(tmp_path)

    _touch_batch_files(tmp_path, series_name, "summary_contract_000", 1)
    _touch_batch_files(tmp_path, series_name, "summary_contract_001", 1)
    first = repo.merge_summary(
        output_dir=output_dir,
        series_name=series_name,
        experiment_results=[
            ExperimentResult("summary_contract_000", True, 2.0),
            ExperimentResult("summary_contract_001", False, 3.0, error_message="boom"),
        ],
    )
    repo.save_summary(output_dir, series_name, first)

    _touch_batch_files(tmp_path, series_name, "summary_contract_001", 2)
    partial = repo.merge_summary(
        output_dir=output_dir,
        series_name=series_name,
        experiment_results=[ExperimentResult("summary_contract_001", True, 1.5)],
    )
    repo.save_summary(output_dir, series_name, partial)
    loaded = repo.load_summary(output_dir, series_name)

    assert loaded["total_experiments"] == 2
    assert loaded["successful_experiments"] == 2
    assert loaded["failed_experiments"] == 0
    names = [r["name"] for r in loaded["experiment_results"]]
    assert names == ["summary_contract_000", "summary_contract_001"]
    exp0 = loaded["experiment_results"][0]
    exp1 = loaded["experiment_results"][1]
    assert exp0["success"] is True
    assert exp1["success"] is True
    # Current contract: partial rerun updates one experiment and uses on-disk count
    # of all matching batch files for that experiment.
    assert exp1["batch_files"] == 3
