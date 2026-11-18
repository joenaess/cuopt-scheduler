import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from cuopt_scheduler import Backend, Outcome, Problem, generate_problem, schedule_cost, solve
from cuopt_scheduler.data import LEVELS
from cuopt_scheduler.lp_round import LPEngine, LPSolution, round_lp, solve_lp, solve_lp_round
from cuopt_scheduler.model import build_model
from cuopt_scheduler.solver import classify

CPU_ENGINES = [e for e in LPEngine if e.backend is Backend.CPU]


@settings(max_examples=50, deadline=None)
@given(
    num_employees=st.integers(3, 15),
    num_days=st.integers(1, 15),
    seed=st.integers(0, 2**32 - 1),
    level=st.sampled_from(sorted(LEVELS)),
    data=st.data(),
)
def test_rounding_any_vector_gives_a_valid_schedule(
    num_employees: int, num_days: int, seed: int, level: int, data: st.DataObject
) -> None:
    """Rounding is safe whatever the LP solver returns: even garbage stays within the rules."""
    problem = generate_problem(num_employees, num_days, seed=seed, level=level)
    model = build_model(problem)
    lp = data.draw(arrays(np.float64, model.num_cols, elements=st.floats(-0.5, 1.5)))
    v = round_lp(model, lp)
    assert classify(model, v, proven=False) is Outcome.FEASIBLE


def test_rounding_an_integral_solution_keeps_it(small_feasible: Problem) -> None:
    model = build_model(small_feasible)
    exact = solve(small_feasible)
    lp = np.zeros(model.num_cols)
    for a in exact.assignments:
        lp[model.x_keys.index((a.employee_id, a.shift, a.day))] = 1
    v = round_lp(model, lp)
    assert float(model.cost @ v) == exact.objective


@settings(max_examples=50, deadline=None)
@given(
    num_employees=st.integers(3, 15),
    num_days=st.integers(1, 15),
    seed=st.integers(0, 2**32 - 1),
    level=st.sampled_from(sorted(LEVELS)),
    data=st.data(),
)
def test_the_lp_bound_never_exceeds_a_valid_schedule_whatever_the_duals(
    num_employees: int, num_days: int, seed: int, level: int, data: st.DataObject
) -> None:
    """The bound is a proof only if it holds for any multipliers, not just good ones."""
    problem = generate_problem(num_employees, num_days, seed=seed, level=level)
    model = build_model(problem)
    duals = data.draw(arrays(np.float64, model.num_rows, elements=st.floats(-2e3, 2e3)))
    schedule = round_lp(model, np.zeros(model.num_cols))  # any valid schedule
    assert model.lower_bound(duals) <= float(model.cost @ schedule)


@settings(max_examples=25, deadline=None)
@given(
    num_employees=st.integers(3, 12),
    num_days=st.integers(1, 10),
    seed=st.integers(0, 2**32 - 1),
    level=st.sampled_from(sorted(LEVELS)),
    scale=st.floats(-3, 3),
)
def test_the_lp_bound_never_exceeds_the_optimum(
    num_employees: int, num_days: int, seed: int, level: int, scale: float
) -> None:
    problem = generate_problem(num_employees, num_days, seed=seed, level=level)
    model = build_model(problem)
    exact = solve(problem)
    assert exact.objective is not None
    duals = solve_lp(model, LPEngine.HIGHS, 10).duals
    for y in (duals, scale * duals):  # the LP's duals, and distorted ones
        assert model.lower_bound(y) <= exact.objective + 1e-6


def test_exact_lp_duals_give_the_lp_optimum() -> None:
    problem = generate_problem(45, 14, seed=1, level=3)
    model = build_model(problem)
    lp = solve_lp(model, LPEngine.HIGHS, 10)
    assert model.lower_bound(lp.duals) == pytest.approx(float(model.cost @ lp.primal), abs=1e-3)


def test_unusable_duals_give_no_bound() -> None:
    model = build_model(generate_problem(5, 3, seed=0))
    assert model.lower_bound(np.zeros(model.num_rows + 1)) == -np.inf
    assert model.lower_bound(np.full(model.num_rows, np.nan)) == -np.inf


def test_lp_round_is_proven_optimal_by_the_lp_bound() -> None:
    # The LP bound is tight for this model, so rounding a good LP solution is optimal, and
    # the LP's duals prove it.
    problem = generate_problem(45, 14, seed=1, level=3)
    result = solve_lp_round(problem, LPEngine.HIGHS)
    assert result.outcome is Outcome.OPTIMAL
    assert result.objective == schedule_cost(problem, result.assignments)
    assert result.objective == solve(problem).objective


@pytest.mark.parametrize("engine", CPU_ENGINES)
def test_lp_round_matches_the_exact_optimum_here(engine: LPEngine) -> None:
    problem = generate_problem(45, 14, seed=1, level=3)
    result = solve_lp_round(problem, engine)
    assert result.objective == schedule_cost(problem, result.assignments)
    assert result.objective == solve(problem).objective


def test_lp_round_without_a_proof_is_only_feasible(monkeypatch: pytest.MonkeyPatch) -> None:
    def useless_duals(model: object, engine: LPEngine, time_limit: float) -> LPSolution:
        lp = real_solve_lp(model, engine, time_limit)  # type: ignore[arg-type]
        return LPSolution(lp.primal, np.zeros_like(lp.duals))

    real_solve_lp = solve_lp
    monkeypatch.setattr("cuopt_scheduler.lp_round.solve_lp", useless_duals)
    result = solve_lp_round(generate_problem(45, 14, seed=1, level=3))
    assert result.outcome is Outcome.FEASIBLE


@pytest.mark.gpu
def test_gpu_lp_round() -> None:
    problem = generate_problem(45, 14, seed=1, level=3)
    result = solve_lp_round(problem, LPEngine.CUOPT_PDLP)
    assert result.backend is Backend.GPU
    assert result.objective == schedule_cost(problem, result.assignments)
    # PDLP solves the LP only approximately; allow a few extra shifts but no lost coverage.
    assert result.objective == pytest.approx(solve(problem).objective, abs=5)


@pytest.mark.gpu
def test_the_gpu_proves_the_rounded_schedule_without_the_milp_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_milp(*args: object, **kwargs: object) -> None:
        raise AssertionError("the LP bound should have proved it")

    monkeypatch.setattr("cuopt_scheduler.lp_round._solve_cuopt", no_milp)
    problem = generate_problem(68, 14, seed=2, wards=2, level=3)
    result = solve(problem, Backend.GPU, time_limit=60)
    assert result.backend is Backend.GPU
    assert result.outcome is Outcome.OPTIMAL
    assert "LP bound" in result.solver_status
    assert result.objective == solve(problem).objective
    assert result.objective == schedule_cost(problem, result.assignments)


def _no_lp_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cuopt_scheduler.lp_round.lp_bound_proves", lambda *args: (False, -np.inf))


@pytest.mark.gpu
def test_warm_started_mip_proves_the_rounded_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_lp_proof(monkeypatch)  # make the MILP solver do the proof
    problem = generate_problem(68, 14, seed=2, wards=2, level=3)
    result = solve(problem, Backend.GPU, time_limit=60)
    assert result.backend is Backend.GPU
    assert result.outcome is Outcome.OPTIMAL
    assert result.objective == solve(problem).objective
    assert result.objective == schedule_cost(problem, result.assignments)
    # The trace starts with the rounded LP schedule, before the MILP solver runs.
    assert result.objective is not None
    first_t, first_obj = result.incumbents[0]
    assert first_t <= result.solve_seconds
    assert first_obj >= result.objective


@pytest.mark.gpu
def test_warm_start_keeps_the_rounded_schedule_if_the_milp_solver_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def out_of_memory(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Memory allocation failed")

    monkeypatch.setattr("cuopt_scheduler.lp_round._solve_cuopt", out_of_memory)
    _no_lp_proof(monkeypatch)
    problem = generate_problem(45, 14, seed=1, level=3)
    result = solve(problem, Backend.GPU)
    assert result.backend is Backend.GPU
    assert result.warnings == ()
    assert result.outcome is Outcome.FEASIBLE
    assert "Memory allocation failed" in result.solver_status
    assert result.objective == schedule_cost(problem, result.assignments)
