from dataclasses import replace

import pytest

from cuopt_scheduler import Problem, generate_problem
from cuopt_scheduler.data import DEFAULT_DEMAND, SHIFTS, SKILLS

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
        assert e.max_weekly_hours in (24, 32, 40)
        assert all(d in p.days and s in SHIFTS for d, s in e.unavailable)


def test_wards_scale_demand() -> None:
    p = generate_problem(10, 7, seed=0, wards=3)
    assert p.demand["Day"]["CNA"] == 3 * DEFAULT_DEMAND["Day"]["CNA"]
    with pytest.raises(ValueError, match="wards"):
        generate_problem(10, 7, wards=0)


def test_ids_unique_even_when_names_collide() -> None:
    # Seeds that produced duplicate names in the original app broke `.loc` lookups.
    for seed in range(50):
        p = generate_problem(100, 7, seed=seed)
        assert len({e.id for e in p.employees}) == 100


@pytest.mark.parametrize(
    ("days", "weeks"),
    [(1, [range(0, 1)]), (7, [range(0, 7)]), (10, [range(0, 7), range(7, 10)])],
)
def test_weeks(days: int, weeks: list[range]) -> None:
    assert Problem(employees=(rn("A"),), num_days=days).weeks == weeks


def test_start_hour_is_absolute() -> None:
    p = Problem(employees=(rn("A"),), num_days=3)
    assert p.start_hour(0, "Day") == 7
    assert p.start_hour(2, "Night") == 48 + 23


def test_problem_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        Problem(employees=(rn("A"), rn("A")), num_days=1)


def test_problem_rejects_empty_horizon() -> None:
    with pytest.raises(ValueError, match="num_days"):
        Problem(employees=(rn("A"),), num_days=0)


def test_problem_rejects_shift_without_start_time() -> None:
    with pytest.raises(ValueError, match="Brunch"):
        Problem(employees=(rn("A"),), num_days=1, demand={"Brunch": {"RN": 1}})


def test_levels_add_rules_to_the_same_staff() -> None:
    base, top = generate_problem(30, 14, seed=4), generate_problem(30, 14, seed=4, level=3)
    assert [replace(e, dislikes=frozenset()) for e in top.employees] == list(base.employees)
    assert base.max_consecutive_days is None and top.max_consecutive_days == 5
    assert dict(top.weekly_shift_caps) == {"Night": 2}
    assert all(not e.dislikes for e in base.employees)
    assert all(e.dislikes and not e.dislikes & e.unavailable for e in top.employees)
    with pytest.raises(ValueError, match="level"):
        generate_problem(5, 7, level=9)


def test_problem_rejects_bad_rule_settings() -> None:
    with pytest.raises(ValueError, match="max_consecutive_days"):
        Problem(employees=(rn("A"),), num_days=3, max_consecutive_days=0)
    with pytest.raises(ValueError, match="Brunch"):
        Problem(employees=(rn("A"),), num_days=3, weekly_shift_caps={"Brunch": 1})
