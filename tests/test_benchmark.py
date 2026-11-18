import pytest

from cuopt_scheduler import Backend, Outcome, SolveResult
from cuopt_scheduler.benchmark import main, run_benchmark, seconds_to_reach


def result(objective: float | None, incumbents: tuple[tuple[float, float], ...]) -> SolveResult:
    return SolveResult(
        outcome=Outcome.FEASIBLE if objective is not None else Outcome.NO_SOLUTION,
        solver_status="",
        backend=Backend.CPU,
        solve_seconds=9.0,
        objective=objective,
        assignments=(),
        incumbents=incumbents,
    )


def test_seconds_to_reach_uses_the_incumbent_trace() -> None:
    r = result(100, ((1.0, 500), (2.0, 101), (3.0, 100)))
    assert seconds_to_reach(r, float("inf")) == 1.0
    assert seconds_to_reach(r, 101) == 2.0
    assert seconds_to_reach(r, 99) is None


def test_seconds_to_reach_without_a_trace_falls_back_to_the_final_result() -> None:
    assert seconds_to_reach(result(100, ()), 100) == 9.0
    assert seconds_to_reach(result(None, ()), float("inf")) is None


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
    # Alone, each run is its own best: it reaches "near" no later than it finishes.
    assert all(r.first_seconds is not None and r.first_seconds <= r.seconds for r in rows)
    assert all(r.near_seconds is not None and r.near_seconds <= r.seconds for r in rows)


def test_cli_prints_table_and_writes_csv(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("cuopt_scheduler.benchmark.cuopt_available", lambda: False)
    out = tmp_path / "bench.csv"
    main(["--wards", "1", "--days", "7", "--time-limit", "10", "--csv", str(out)])
    printed = capsys.readouterr().out
    assert "running CPU only" in printed and "optimal" in printed
    header = out.read_text().splitlines()[0]
    assert header.startswith("wards,employees,days,seed,backend,outcome")
    assert header.endswith("first_seconds,near_seconds")
