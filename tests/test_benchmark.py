import pytest

from cuopt_scheduler import Backend
from cuopt_scheduler.benchmark import main, run_benchmark


def test_run_benchmark_cpu() -> None:
    rows = run_benchmark(
        wards=[1, 2],
        staff_per_ward=45,
        days=[7],
        seeds=[0],
        backends=[Backend.CPU],
        time_limit=10,
    )
    assert [(r.wards, r.employees, r.backend) for r in rows] == [(1, 45, "cpu"), (2, 90, "cpu")]
    assert all(r.outcome == "optimal" and r.uncovered == 0 for r in rows)


def test_cli_prints_table_and_writes_csv(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("cuopt_scheduler.benchmark.cuopt_available", lambda: False)
    out = tmp_path / "bench.csv"
    main(["--wards", "1", "--days", "7", "--time-limit", "10", "--csv", str(out)])
    printed = capsys.readouterr().out
    assert "running CPU only" in printed and "optimal" in printed
    assert out.read_text().startswith("wards,employees,days,seed,backend,outcome")
