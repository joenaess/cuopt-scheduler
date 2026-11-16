import pytest

from cuopt_scheduler import Problem, generate_problem
from cuopt_scheduler.data import SHIFTS, SKILLS

from .conftest import rn


def test_same_seed_same_problem() -> None:
    assert generate_problem(30, 7, seed=1) == generate_problem(30, 7, seed=1)


def test_different_seed_different_problem() -> None:
    assert generate_problem(30, 7, seed=1) != generate_problem(30, 7, seed=2)


def test_shape_and_value_ranges() -> None:
    p = generate_problem(50, 14, seed=3)
    assert len(p.employees) == 50
    assert p.num_days == 14
    assert p.shifts == SHIFTS
    for e in p.employees:
        assert e.skill in SKILLS
        assert e.max_hours in (24, 32, 40)
        assert all(d in p.days and s in SHIFTS for d, s in e.unavailable)


def test_ids_unique_even_when_names_collide() -> None:
    # Seeds that produced duplicate names in the original app broke `.loc` lookups.
    for seed in range(50):
        p = generate_problem(100, 7, seed=seed)
        assert len({e.id for e in p.employees}) == 100


def test_problem_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        Problem(employees=(rn("A"), rn("A")), num_days=1)


def test_problem_rejects_empty_horizon() -> None:
    with pytest.raises(ValueError, match="num_days"):
        Problem(employees=(rn("A"),), num_days=0)
