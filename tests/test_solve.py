import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cuopt_scheduler import Backend, Problem, generate_problem, solve, violations
from cuopt_scheduler.model import build_model


def test_build_model_size(small_feasible: Problem) -> None:
    model = build_model(small_feasible)
    assert len(model.x) == 3 * 2 * 3  # employees x shifts x days


def test_feasible_schedule_is_valid_and_minimal(small_feasible: Problem) -> None:
    result = solve(small_feasible)
    assert result.feasible
    assert result.backend is Backend.CPU
    assert violations(small_feasible, result.assignments) == []
    # Minimising shifts means exactly meeting demand: 3 days x 2 shifts x 1 RN.
    assert result.objective == len(result.assignments) == 6


def test_infeasible_reports_status_and_no_assignments(small_infeasible: Problem) -> None:
    result = solve(small_infeasible)
    assert not result.feasible
    assert result.status == "Infeasible"
    assert result.assignments == ()
    assert result.objective is None


def test_gpu_falls_back_to_cpu_when_unavailable(
    small_feasible: Problem, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("cuopt_scheduler.solver.cuopt_available", lambda: False)
    result = solve(small_feasible, Backend.GPU)
    assert result.feasible
    assert result.backend is Backend.CPU
    assert result.warnings == ("cuOpt is not available; solved on CPU instead.",)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    num_employees=st.integers(10, 80),
    num_days=st.integers(1, 7),
    seed=st.integers(0, 2**32 - 1),
)
def test_any_solution_satisfies_all_rules(num_employees: int, num_days: int, seed: int) -> None:
    problem = generate_problem(num_employees, num_days, seed=seed)
    result = solve(problem)
    if result.feasible:
        assert violations(problem, result.assignments) == []
        assert result.objective == len(result.assignments)


@pytest.mark.gpu
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gpu_matches_cpu_objective(seed: int) -> None:
    problem = generate_problem(80, 7, seed=seed)
    cpu, gpu = solve(problem, Backend.CPU), solve(problem, Backend.GPU)
    assert gpu.backend is Backend.GPU, gpu.warnings
    assert gpu.status == cpu.status
    if cpu.feasible:
        assert gpu.objective == pytest.approx(cpu.objective)
        assert violations(problem, gpu.assignments) == []
