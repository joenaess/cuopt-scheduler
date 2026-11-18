from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from cuopt_scheduler import (
    Assignment,
    Backend,
    Outcome,
    Problem,
    coverage_gaps,
    generate_problem,
    schedule_cost,
    solve,
    violations,
)
from cuopt_scheduler import solver as solver_module
from cuopt_scheduler.data import LEVELS
from cuopt_scheduler.model import ScheduleModel, build_model, rest_conflicts
from cuopt_scheduler.solver import _gap_closed, classify

from .conftest import SMALL_DEMAND, rn


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
    assert result.objective == schedule_cost(understaffed, result.assignments)


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
    assert result.assignments
    assert {a.day for a in result.assignments} == {0, 7}
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
    level=st.sampled_from(sorted(LEVELS)),
)
def test_every_schedule_obeys_the_rules(
    num_employees: int, num_days: int, min_rest: int, seed: int, level: int
) -> None:
    problem = replace(
        generate_problem(num_employees, num_days, seed=seed, level=level), min_rest_hours=min_rest
    )
    result = solve(problem)
    assert result.outcome is Outcome.OPTIMAL
    assert violations(problem, result.assignments) == []
    # The model's view of uncovered demand matches the validator's.
    assert result.objective == pytest.approx(schedule_cost(problem, result.assignments))


# Substrings of the validator's messages, one per rule, for Hypothesis' statistics.
RULES = ("h in week", "shifts on day", "in a row", "shifts in week", "rests", "unavailable")


@settings(max_examples=100, deadline=None)
@given(
    num_employees=st.integers(3, 12),
    num_days=st.integers(1, 15),
    min_rest=st.integers(0, 16),
    seed=st.integers(0, 2**32 - 1),
    level=st.sampled_from(sorted(LEVELS)),
    data=st.data(),
)
def test_model_agrees_with_validator(
    num_employees: int, num_days: int, min_rest: int, seed: int, level: int, data: st.DataObject
) -> None:
    """Any schedule satisfies the model's rows exactly when the validator accepts it.

    Ties the array encoding to the independent rule checker, without involving a solver.
    """
    problem = replace(
        generate_problem(num_employees, num_days, seed=seed, level=level), min_rest_hours=min_rest
    )
    model = build_model(problem)
    # Concentrate on a few people, so per-person rules (weekly caps, days in a row, rest)
    # actually get broken in some examples.
    per_person = len(model.x_keys) // num_employees
    who = data.draw(st.sets(st.integers(0, num_employees - 1), min_size=1, max_size=3))
    columns = [i * per_person + j for i in sorted(who) for j in range(per_person)]
    picked = data.draw(st.sets(st.sampled_from(columns), max_size=3 * num_days))
    if data.draw(st.booleans()):
        # A run of the same shift on consecutive days, which random picks rarely produce.
        i, si = min(who), data.draw(st.integers(0, len(problem.shifts) - 1))
        start, length = data.draw(st.integers(0, num_days - 1)), data.draw(st.integers(2, 8))
        days = range(start, min(num_days, start + length))
        picked |= {i * per_person + si * num_days + d for d in days}
    schedule = [
        Assignment(day=d, shift=s, employee_id=e) for e, s, d in (model.x_keys[i] for i in picked)
    ]
    gaps = coverage_gaps(problem, schedule)
    v = np.zeros(model.num_cols)
    v[list(picked)] = 1
    v[len(model.x_keys) :] = [gaps.get(k, 0) for k in model.gap_keys]
    broken = violations(problem, schedule)
    for rule in RULES:
        if any(rule in message for message in broken):
            event(f"breaks {rule!r}")
    if not broken:
        event("valid")
    assert model.is_feasible(v) == (not broken)
    assert model.cost @ v == schedule_cost(problem, schedule)


@pytest.mark.gpu
@pytest.mark.parametrize("warm_start", [True, False])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gpu_matches_cpu(seed: int, warm_start: bool) -> None:
    problem = generate_problem(45, 14, seed=seed)
    cpu = solve(problem, Backend.CPU)
    gpu = solve(problem, Backend.GPU, warm_start=warm_start)
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


@pytest.mark.parametrize("warm_start", [True, False])
def test_gpu_failure_falls_back_to_cpu(
    small_feasible: Problem, monkeypatch: pytest.MonkeyPatch, warm_start: bool
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("cuopt_scheduler.solver.cuopt_available", lambda: True)
    monkeypatch.setattr(solver_module, "_solve_cuopt", broken)
    monkeypatch.setattr(solver_module, "_solve_cuopt_warm", broken)
    result = solve(small_feasible, Backend.GPU, warm_start=warm_start)
    assert result.backend is Backend.CPU
    assert result.outcome is Outcome.OPTIMAL
    assert result.warnings == ("cuOpt failed (boom); solved on CPU instead.",)


@pytest.mark.parametrize("backend", [Backend.CPU, pytest.param(Backend.GPU, marks=pytest.mark.gpu)])
def test_incumbent_trace_ends_at_the_result(backend: Backend) -> None:
    result = solve(generate_problem(45, 14, seed=1), backend)
    assert result.incumbents, "no incumbent was reported"
    times = [t for t, _ in result.incumbents]
    assert times == sorted(times)
    assert times[-1] <= result.solve_seconds
    assert result.incumbents[-1][1] == pytest.approx(result.objective)


def model_accepts(problem: Problem, schedule: list[Assignment]) -> bool:
    model = build_model(problem)
    v = np.zeros(model.num_cols)
    v[[model.x_keys.index((a.employee_id, a.shift, a.day)) for a in schedule]] = 1
    gaps = coverage_gaps(problem, schedule)
    v[len(model.x_keys) :] = [gaps.get(k, 0) for k in model.gap_keys]
    return model.is_feasible(v)


@pytest.mark.parametrize(("days_in_a_row", "ok"), [(5, True), (6, False)])
def test_model_limits_days_in_a_row(days_in_a_row: int, ok: bool) -> None:
    p = Problem(
        employees=(rn("A", 56),), num_days=7, demand={"Day": {"RN": 1}}, max_consecutive_days=5
    )
    assert model_accepts(p, [Assignment(d, "Day", "A") for d in range(days_in_a_row)]) is ok


@pytest.mark.parametrize(("nights", "ok"), [(2, True), (3, False)])
def test_model_caps_nights_per_week(nights: int, ok: bool) -> None:
    p = Problem(
        employees=(rn("A"),),
        num_days=7,
        demand={"Night": {"RN": 1}},
        weekly_shift_caps={"Night": 2},
    )
    assert model_accepts(p, [Assignment(2 * i, "Night", "A") for i in range(nights)]) is ok


def test_dislikes_raise_the_cost_and_are_avoided() -> None:
    a = replace(rn("A"), dislikes=frozenset({(0, "Day")}))
    p = Problem(employees=(a, rn("B")), num_days=1, demand={"Day": {"RN": 1}})
    result = solve(p)
    assert result.assignments == (Assignment(0, "Day", "B"),)
    assert build_model(p).cost[build_model(p).x_keys.index(("A", "Day", 0))] == 2


def test_phases_and_live_progress_on_the_cpu() -> None:
    from cuopt_scheduler.solver import HIGHS_SEARCH, ProgressLog

    progress = ProgressLog()
    result = solve(generate_problem(30, 7, seed=4, level=3), progress=progress)
    assert [label for label, _ in result.phases] == [HIGHS_SEARCH]
    assert sum(s for _, s in result.phases) == pytest.approx(result.solve_seconds, abs=0.05)
    assert progress.phase == HIGHS_SEARCH
    assert progress.incumbents == list(result.incumbents)
