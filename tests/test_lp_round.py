import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from cuopt_scheduler import Backend, Outcome, Problem, generate_problem, schedule_cost, solve
from cuopt_scheduler.data import LEVELS
from cuopt_scheduler.lp_round import LPEngine, cupdlpx_available, round_lp, solve_lp_round
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


@pytest.mark.parametrize("engine", CPU_ENGINES)
def test_lp_round_matches_the_exact_optimum_here(engine: LPEngine) -> None:
    # The LP bound is tight for this model, so rounding a good LP solution is optimal.
    problem = generate_problem(45, 14, seed=1, level=3)
    result = solve_lp_round(problem, engine)
    assert result.outcome is Outcome.FEASIBLE  # never claims a proof
    assert result.objective == schedule_cost(problem, result.assignments)
    assert result.objective == solve(problem).objective


@pytest.mark.gpu
@pytest.mark.parametrize("engine", [LPEngine.CUOPT_PDLP, LPEngine.CUPDLPX])
def test_gpu_lp_round(engine: LPEngine) -> None:
    if engine is LPEngine.CUPDLPX and not cupdlpx_available():
        pytest.skip("cuPDLPx not installed (uv sync --extra pdlp)")
    problem = generate_problem(45, 14, seed=1, level=3)
    result = solve_lp_round(problem, engine)
    assert result.backend is Backend.GPU
    assert result.objective == schedule_cost(problem, result.assignments)
    # PDLP solves the LP only approximately; allow a few extra shifts but no lost coverage.
    assert result.objective == pytest.approx(solve(problem).objective, abs=5)
