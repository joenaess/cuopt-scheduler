import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cuopt_scheduler import Backend, generate_problem, schedule_cost, solve
from cuopt_scheduler.data import LEVELS
from cuopt_scheduler.explain import (
    cost_breakdown,
    decision_grid,
    explain,
    is_proven,
    schedule_vector,
)
from cuopt_scheduler.model import build_model
from cuopt_scheduler.viz import decision_grid_figure


@settings(max_examples=30, deadline=None)
@given(
    num_employees=st.integers(1, 20),
    num_days=st.integers(1, 15),
    seed=st.integers(0, 2**32 - 1),
    level=st.sampled_from(sorted(LEVELS)),
)
def test_the_rules_account_for_every_constraint(
    num_employees: int, num_days: int, seed: int, level: int
) -> None:
    model = build_model(generate_problem(num_employees, num_days, seed=seed, level=level))
    assert sum(n for _, n in model.rules) == model.num_rows
    assert all(n > 0 for _, n in model.rules)


@pytest.mark.parametrize("num_employees", [12, 45])  # understaffed, fully staffed
def test_the_cost_breakdown_adds_up_to_the_solvers_objective(num_employees: int) -> None:
    problem = generate_problem(num_employees, 7, seed=42, level=3)
    result = solve(problem)
    model = build_model(problem)
    values = schedule_vector(model, result.assignments)
    assert model.is_feasible(values)
    breakdown = cost_breakdown(model, values)
    assert breakdown["Cost"].sum() == pytest.approx(result.objective)
    assert breakdown["Cost"].sum() == pytest.approx(schedule_cost(problem, result.assignments))
    shifts = breakdown.set_index("Part").loc["Shifts worked", "Count"]
    assert shifts == len(result.assignments)


def test_the_explanation_of_a_well_staffed_problem_proves_the_optimum() -> None:
    problem = generate_problem(45, 7, seed=42, level=3)
    ex = explain(problem)
    objective = float(ex.model.cost @ ex.rounded)
    assert ex.model.is_feasible(ex.rounded)
    assert objective == solve(problem).objective
    assert is_proven(objective, ex.bound)
    assert 0.9 <= ex.share_already_whole <= 1
    assert ex.num_fixed == int(np.sum(ex.model.col_upper[: ex.num_decisions] == 0))


def test_an_understaffed_problem_can_leave_a_gap_the_search_must_close() -> None:
    problem = generate_problem(12, 7, seed=42, level=3)
    ex = explain(problem)
    optimum = solve(problem).objective
    assert optimum is not None
    assert ex.bound <= optimum
    assert not is_proven(float(ex.model.cost @ ex.rounded), ex.bound)


def test_is_proven_needs_a_finite_bound_within_one() -> None:
    assert is_proven(100.0, 99.2)
    assert not is_proven(100.0, 99.0 - 1e-3)
    assert not is_proven(100.0, -np.inf)


@pytest.mark.gpu
def test_explain_uses_the_gpu_when_asked() -> None:
    from cuopt_scheduler.lp_round import LPEngine

    ex = explain(generate_problem(45, 7, seed=42), Backend.GPU)
    assert ex.engine is LPEngine.CUOPT_PDLP
    assert is_proven(float(ex.model.cost @ ex.rounded), ex.bound)


def test_decision_grid_is_a_slice_with_unavailable_cells_blank() -> None:
    problem = generate_problem(20, 14, seed=3)
    model = build_model(problem)
    values = np.linspace(0, 1, model.num_cols, dtype=np.float64)
    grid = decision_grid(problem, model, values, max_employees=5, max_days=4)
    assert grid.shape == (5, 4 * len(problem.shifts))
    for i, e in enumerate(problem.employees[:5]):
        for d in range(4):
            for s in problem.shifts:
                j = model.x_keys.index((e.id, s, d))
                cell = grid.iloc[i][(d, s)]
                if model.col_upper[j] == 0:
                    assert np.isnan(cell)
                else:
                    assert cell == values[j]


def test_decision_grid_figure_marks_fractions_and_unavailable_cells() -> None:
    problem = generate_problem(10, 7, seed=0)
    model = build_model(problem)
    values = np.full(model.num_cols, 0.5)
    grid = decision_grid(problem, model, values)
    fig = decision_grid_figure(grid, "test")
    text = [cell for row in fig.data[0].text for cell in row]  # pyright: ignore[reportAttributeAccessIssue]
    assert text.count("✕") == int(grid.isna().to_numpy().sum()) > 0
    assert "0.50" in text
