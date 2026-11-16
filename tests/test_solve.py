from dataclasses import replace

import pulp
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cuopt_scheduler import (
    Backend,
    Outcome,
    Problem,
    coverage_gaps,
    generate_problem,
    solve,
    violations,
)
from cuopt_scheduler.model import UNCOVERED_PENALTY, build_model, rest_conflicts
from cuopt_scheduler.solver import classify

from .conftest import SMALL_DEMAND, rn


def expected_objective(problem: Problem, result) -> float:
    """Objective recomputed from the schedule via the independent validator."""
    gaps = coverage_gaps(problem, result.assignments)
    return UNCOVERED_PENALTY * sum(gaps.values()) + len(result.assignments)


def test_build_model_size(small_feasible: Problem) -> None:
    model = build_model(small_feasible)
    assert len(model.x) == 3 * 2 * 3  # employees x shifts x days
    assert len(model.uncovered) == 3 * 2  # days x (shift, skill)


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


# --- classify(): solver-independent result classification -------------------------------


def tiny_lp(x_value: float | None, status: int, sol_status: int) -> pulp.LpProblem:
    lp = pulp.LpProblem("t", pulp.LpMinimize)
    x = pulp.LpVariable("x", cat=pulp.LpBinary)
    lp += x
    lp += x >= 1
    x.varValue = x_value
    lp.assignStatus(status, sol_status)
    return lp


def test_classify_optimal() -> None:
    lp = tiny_lp(1, pulp.LpStatusOptimal, pulp.LpSolutionOptimal)
    assert classify(lp) is Outcome.OPTIMAL


def test_classify_cbc_time_limit_with_solution() -> None:
    # CBC: status "Optimal" but only an integer-feasible solution.
    lp = tiny_lp(1, pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible)
    assert classify(lp) is Outcome.FEASIBLE


def test_classify_cuopt_time_limit_with_solution() -> None:
    # PuLP's cuOpt adapter: "Not Solved", but valid values loaded.
    lp = tiny_lp(1, pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound)
    assert classify(lp) is Outcome.FEASIBLE


@pytest.mark.parametrize("value", [None, 0, 0.5])
def test_classify_rejects_missing_invalid_or_fractional(value: float | None) -> None:
    lp = tiny_lp(value, pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound)
    assert classify(lp) is Outcome.NO_SOLUTION


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
def test_gpu_gap_parameter_reaches_cuopt(small_feasible: Problem) -> None:
    from cuopt_scheduler.solver import _solver

    solver = _solver(Backend.GPU, time_limit=5, gap_rel=0.02)
    assert solver.cuopt_params == {"mip_relative_gap": 0.02}  # pyright: ignore[reportAttributeAccessIssue]
    assert solve(small_feasible, Backend.GPU, gap_rel=0.02).outcome is Outcome.OPTIMAL
