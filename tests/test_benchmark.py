import dataclasses

import pytest

from cuopt_scheduler import Backend, Outcome, SolveResult, generate_problem
from cuopt_scheduler.benchmark import Method, _row, iter_benchmark, main, seconds_to_reach


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


CPU_METHODS = [m for m in Method if m.device is Backend.CPU]


def test_benchmark_cpu_methods() -> None:
    batches = list(
        iter_benchmark(
            wards=[1, 2],
            staff_per_ward=45,
            days=[7],
            seeds=[0],
            methods=CPU_METHODS,
            time_limit=10,
            levels=[0, 3],
        )
    )
    assert len(batches) == 4  # one per problem: 2 sizes x 2 levels
    rows = [r for batch in batches for r in batch]
    assert {(r.wards, r.level) for r in rows} == {(1, 0), (1, 3), (2, 0), (2, 3)}
    assert {r.method for r in rows} == set(CPU_METHODS)
    assert {r.device for r in rows} == {"cpu"}
    mip = [r for r in rows if r.method == Method.HIGHS_MIP]
    assert all(r.outcome == "optimal" and r.uncovered == 0 for r in mip)
    # LP + rounding is optimal when the LP bound proves it; HiGHS simplex's exact duals do.
    simplex = [r for r in rows if r.method == Method.HIGHS_SIMPLEX_ROUND]
    assert all(r.outcome == "optimal" for r in simplex)
    assert all(r.outcome in ("optimal", "feasible") for r in rows)
    # Alone, each run is its own best: it reaches "near" no later than it finishes.
    for r in rows:
        assert r.seconds is not None
        assert r.first_seconds is not None
        assert r.near_seconds is not None
        assert r.first_seconds <= r.seconds
        assert r.near_seconds <= r.seconds


def test_cli_prints_table_and_writes_csv(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("cuopt_scheduler.benchmark.cuopt_available", lambda: False)
    out = tmp_path / "bench.csv"
    main(["--wards", "1", "--days", "7", "--time-limit", "10", "--csv", str(out)])
    printed = capsys.readouterr().out
    assert "running CPU only" in printed
    assert "optimal" in printed
    lines = out.read_text().splitlines()
    assert lines[0].startswith("wards,employees,level,days,seed,method,device,outcome")
    assert lines[0].endswith("first_seconds,near_seconds")
    assert len(lines) == 1 + len(CPU_METHODS)


def test_cli_rejects_unavailable_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cuopt_scheduler.benchmark.cuopt_available", lambda: False)
    with pytest.raises(SystemExit, match="not available here: cuopt-mip"):
        main(["--methods", "highs-mip", "cuopt-mip"])


def test_a_gpu_method_that_fell_back_to_the_cpu_is_recorded_as_failed() -> None:
    fallback = dataclasses.replace(
        result(100, ((1.0, 100),)), warnings=("cuOpt failed (Memory allocation failed)",)
    )
    row = _row(generate_problem(10, 7, seed=0), 1, 0, 7, 0, Method.CUOPT_MIP, fallback, 101)
    assert row.outcome == "failed: cuOpt failed (Memory allocation failed)"
    assert (row.device, row.seconds, row.objective, row.near_seconds) == ("gpu", None, None, None)
