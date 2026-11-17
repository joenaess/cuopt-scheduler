from dataclasses import dataclass, replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cuopt_scheduler import (
    Assignment,
    Backend,
    Outcome,
    Problem,
    SolveResult,
    coverage_gaps,
    generate_problem,
    solve,
    violations,
)
from cuopt_scheduler import solver as solver_module
from cuopt_scheduler.model import UNCOVERED_PENALTY, ScheduleModel, build_model, rest_conflicts
from cuopt_scheduler.solver import _gap_closed, classify

from .conftest import SMALL_DEMAND, rn


@dataclass(frozen=True)
class SolveResultStub:
    assignments: tuple[Assignment, ...]


def expected_objective(problem: Problem, result: SolveResult | SolveResultStub) -> float:
    """Objective recomputed from the schedule via the independent validator."""
    gaps = coverage_gaps(problem, result.assignments)
    return UNCOVERED_PENALTY * sum(gaps.values()) + len(result.assignments)


def test_build_model_size(small_feasible: Problem) -> None:
    model = build_model(small_feasible)
    assert len(model.x_keys) == 3 * 2 * 3  # employees x shifts x days
    assert len(model.gap_keys) == 3 * 2  # days x (shift, skill)
    assert model.num_cols == 18 + 6
    assert model.offsets[-1] == len(model.indices) == len(model.values)


def test_unavailability_is_a_column_bound() -> None:
    p = Problem(employees=(rn("A", unavailable={(1, "Night")}),), num_days=2, demand=SMALL_DEMAND)
    model = build_model(p)
    blocked = [k for k, ub in zip(model.x_keys, model.col_upper, strict=False) if ub == 0]
    assert blocked == [("A", "Night", 1)]


def test_rest_conflicts_default_shifts() -> None:
    p = Problem(employees=(rn("A"),), num_days=2)
    # Night (ends 07:00) leaves <11h before Day or Evening; Evening (ends 23:00) before Day.
    assert set(rest_conflicts(p)) == {("Night", "Day"), ("Night", "Evening"), ("Evening", "Day")}
    assert set(rest_conflicts(replace(p, min_rest_hours=0))) == set()


def test_fully_covered_schedule(small_feasible: Problem) -> None:
    result = solve(small_feasible)
    assert result.outcome is Outcome.OPTIMAL
    assert result.backend is Backend.CPU
    assert violations(small_feasible, result.assignments) == []
    assert coverage_gaps(small_feasible, result.assignments) == {}
    assert result.objective == len(result.assignments) == 6


def test_understaffed_reports_gaps_instead_of_failing(understaffed: Problem) -> None:
    result = solve(understaffed)
    assert result.outcome is Outcome.OPTIMAL
    assert violations(understaffed, result.assignments) == []
    assert coverage_gaps(understaffed, result.assignments) == {(1, "Night", "RN"): 1}
    assert result.objective == expected_objective(understaffed, result)


def test_rest_rule_changes_the_schedule() -> None:
    # One nurse who can only work Night on day 0 and Day on day 1: back to back with no rest.
    p = Problem(
        employees=(rn("A", unavailable={(0, "Day"), (1, "Night")}),),
        num_days=2,
        demand=SMALL_DEMAND,
    )
    relaxed, strict = solve(replace(p, min_rest_hours=0)), solve(p)
    # Day 0 Day and day 1 Night are uncoverable either way; the rest rule costs one more.
    assert sum(coverage_gaps(p, relaxed.assignments).values()) == 2
    assert sum(coverage_gaps(p, strict.assignments).values()) == 3
    assert violations(p, strict.assignments) == []


def test_hour_cap_resets_each_week() -> None:
    # 8h/week cap, one Day shift needed on days 0 and 7: both coverable by the same nurse.
    off = {(d, "Day") for d in range(1, 7)}
    p = Problem(employees=(rn("A", 8, unavailable=off),), num_days=8, demand={"Day": {"RN": 1}})
    result = solve(p)
    assert result.assignments and {a.day for a in result.assignments} == {0, 7}
    assert violations(p, result.assignments) == []


def test_gpu_falls_back_to_cpu_when_unavailable(
    small_feasible: Problem, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("cuopt_scheduler.solver.cuopt_available", lambda: False)
    result = solve(small_feasible, Backend.GPU)
    assert result.feasible
    assert result.backend is Backend.CPU
    assert result.warnings == ("cuOpt is not available; solved on CPU instead.",)


def test_gap_tolerance_is_accepted(small_feasible: Problem) -> None:
    assert solve(small_feasible, gap_rel=0.05).outcome is Outcome.OPTIMAL


# --- classify(): our own check of the solver's answer --------------------------------------


@pytest.fixture
def tiny() -> ScheduleModel:
    # One binary x with x >= 1.
    return ScheduleModel(
        x_keys=(("A", "Day", 0),),
        gap_keys=(),
        cost=np.ones(1),
        col_lower=np.zeros(1),
        col_upper=np.ones(1),
        integer=np.ones(1, dtype=np.bool_),
        row_lower=np.ones(1),
        row_upper=np.full(1, np.inf),
        offsets=np.array([0, 1], np.int32),
        indices=np.zeros(1, np.int32),
        values=np.ones(1),
    )


@pytest.mark.parametrize(
    ("proven", "expected"), [(True, Outcome.OPTIMAL), (False, Outcome.FEASIBLE)]
)
def test_classify_valid_solution(tiny: ScheduleModel, proven: bool, expected: Outcome) -> None:
    assert classify(tiny, np.array([1.0]), proven) is expected


@pytest.mark.parametrize(
    "values",
    [None, [0.0], [0.5], [2.0], [np.nan], [1.0, 0.0]],
    ids=["missing", "breaks-row", "fractional", "out-of-bounds", "nan", "wrong-length"],
)
def test_classify_rejects_invalid(tiny: ScheduleModel, values: list[float] | None) -> None:
    arr = None if values is None else np.array(values)
    # Even if the solver claims to have proven it.
    assert classify(tiny, arr, proven=True) is Outcome.NO_SOLUTION


@pytest.mark.parametrize(
    ("objective", "bound", "gap_rel", "closed"),
    [
        (4200, 4200, None, True),
        (4200, 4199.2, None, True),  # costs are integers: < 1 apart proves it
        (4201, 4200, None, False),
        (4_303_699, 16_800, None, False),  # what cuOpt once called "Optimal"
        (101, 100, 0.01, True),
        (103, 100, 0.01, False),
        (1, float("-inf"), None, True),  # cuOpt when presolve solves it outright
    ],
)
def test_gap_closed(objective: float, bound: float, gap_rel: float | None, closed: bool) -> None:
    assert _gap_closed(objective, bound, gap_rel) is closed


# --- properties ---------------------------------------------------------------------------


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    num_employees=st.integers(10, 80),
    num_days=st.integers(1, 15),
    min_rest=st.integers(0, 16),
    seed=st.integers(0, 2**32 - 1),
)
def test_every_schedule_obeys_the_rules(
    num_employees: int, num_days: int, min_rest: int, seed: int
) -> None:
    problem = replace(generate_problem(num_employees, num_days, seed=seed), min_rest_hours=min_rest)
    result = solve(problem)
    assert result.outcome is Outcome.OPTIMAL
    assert violations(problem, result.assignments) == []
    # The model's view of uncovered demand matches the validator's.
    assert result.objective == pytest.approx(expected_objective(problem, result))


@settings(max_examples=50, deadline=None)
@given(
    num_employees=st.integers(3, 12),
    num_days=st.integers(1, 9),
    min_rest=st.integers(0, 16),
    seed=st.integers(0, 2**32 - 1),
    data=st.data(),
)
def test_model_agrees_with_validator(
    num_employees: int, num_days: int, min_rest: int, seed: int, data: st.DataObject
) -> None:
    """Any schedule satisfies the model's rows exactly when the validator accepts it.

    Ties the array encoding to the independent rule checker, without involving a solver.
    """
    problem = replace(generate_problem(num_employees, num_days, seed=seed), min_rest_hours=min_rest)
    model = build_model(problem)
    picked = data.draw(st.sets(st.sampled_from(range(len(model.x_keys))), max_size=40))
    schedule = [
        Assignment(day=d, shift=s, employee_id=e) for e, s, d in (model.x_keys[i] for i in picked)
    ]
    gaps = coverage_gaps(problem, schedule)
    v = np.zeros(model.num_cols)
    v[list(picked)] = 1
    v[len(model.x_keys) :] = [gaps.get(k, 0) for k in model.gap_keys]
    assert model.is_feasible(v) == (violations(problem, schedule) == [])
    assert model.cost @ v == expected_objective(problem, SolveResultStub(tuple(schedule)))


@pytest.mark.gpu
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gpu_matches_cpu(seed: int) -> None:
    problem = generate_problem(45, 14, seed=seed)
    cpu, gpu = solve(problem, Backend.CPU), solve(problem, Backend.GPU)
    assert gpu.backend is Backend.GPU, gpu.warnings
    assert cpu.outcome is gpu.outcome is Outcome.OPTIMAL
    assert gpu.objective == pytest.approx(cpu.objective)
    assert violations(problem, gpu.assignments) == []


@pytest.mark.gpu
def test_gpu_reports_its_own_backend(understaffed: Problem) -> None:
    result = solve(understaffed, Backend.GPU, gap_rel=0.02)
    assert (result.backend, result.warnings) == (Backend.GPU, ())
    assert result.outcome is Outcome.OPTIMAL
    assert coverage_gaps(understaffed, result.assignments) == {(1, "Night", "RN"): 1}


def test_gpu_failure_falls_back_to_cpu(
    small_feasible: Problem, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("cuopt_scheduler.solver.cuopt_available", lambda: True)
    monkeypatch.setitem(solver_module._SOLVERS, Backend.GPU, broken)
    result = solve(small_feasible, Backend.GPU)
    assert result.backend is Backend.CPU and result.outcome is Outcome.OPTIMAL
    assert result.warnings == ("cuOpt failed (boom); solved on CPU instead.",)
