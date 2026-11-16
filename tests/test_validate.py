"""The validator is the oracle for the solver tests, so it gets its own tests."""

from dataclasses import replace

from cuopt_scheduler import Assignment, Employee, Problem, coverage_gaps, violations

from .conftest import SMALL_DEMAND, rn

A = Assignment


def problem(
    *employees: Employee,
    days: int = 1,
    demand: dict[str, dict[str, int]] | None = None,
) -> Problem:
    return Problem(employees=employees, num_days=days, demand=demand or SMALL_DEMAND)


def test_valid_schedule() -> None:
    p = problem(rn("A"), rn("B"))
    schedule = [A(0, "Day", "A"), A(0, "Night", "B")]
    assert violations(p, schedule) == []
    assert coverage_gaps(p, schedule) == {}


def test_unmet_demand_is_a_gap_not_a_violation() -> None:
    p = problem(rn("A"), rn("B"))
    assert violations(p, [A(0, "Day", "A")]) == []
    assert coverage_gaps(p, [A(0, "Day", "A")]) == {(0, "Night", "RN"): 1}


def test_wrong_skill_does_not_cover() -> None:
    p = problem(rn("A"), replace(rn("B"), skill="CNA"))
    assert coverage_gaps(p, [A(0, "Day", "A"), A(0, "Night", "B")]) == {(0, "Night", "RN"): 1}


def test_detects_two_shifts_same_day() -> None:
    p = problem(rn("A"))
    assert "A works 2 shifts on day 0" in violations(p, [A(0, "Day", "A"), A(0, "Night", "A")])


def test_hour_cap_is_per_week() -> None:
    p = problem(rn("A", max_weekly_hours=8), days=8, demand={"Day": {"RN": 1}})
    # Day 0 and day 7 are in different weeks: fine.
    assert violations(p, [A(0, "Day", "A"), A(7, "Day", "A")]) == []
    # Day 0 and day 2 are in the same week: over the cap.
    assert violations(p, [A(0, "Day", "A"), A(2, "Day", "A")]) == [
        "A works 16h in week 1, cap is 8h"
    ]


def test_detects_unavailability() -> None:
    p = problem(rn("A", unavailable={(0, "Day")}), demand={"Day": {"RN": 1}})
    assert violations(p, [A(0, "Day", "A")]) == ["A is unavailable for Day on day 0"]


def test_detects_short_rest() -> None:
    p = problem(rn("A"), days=2)
    # Night 23:00-07:00, then Day at 07:00: no rest at all.
    assert violations(p, [A(0, "Night", "A"), A(1, "Day", "A")]) == [
        "A rests 0h between Night day 0 and Day day 1, minimum is 11h"
    ]
    # Day 07:00-15:00, then Night at 23:00 the next day: 32h.
    assert violations(p, [A(0, "Day", "A"), A(1, "Night", "A")]) == []


def test_rest_rule_is_configurable() -> None:
    p = problem(rn("A"), days=2, demand={"Evening": {"RN": 1}, "Day": {"RN": 1}})
    evening_then_day = [A(0, "Evening", "A"), A(1, "Day", "A")]  # 8h rest
    assert len(violations(p, evening_then_day)) == 1
    assert violations(replace(p, min_rest_hours=8), evening_then_day) == []


def test_detects_duplicates() -> None:
    p = problem(rn("A"), demand={"Day": {"RN": 1}})
    assert violations(p, [A(0, "Day", "A"), A(0, "Day", "A")]) == [
        "A assigned 2 times to Day on day 0",
        "A works 2 shifts on day 0",
    ]


def test_detects_unknown_references() -> None:
    p = problem(rn("A"))
    assert violations(p, [A(5, "Brunch", "Z")]) == [
        "unknown employee Z",
        "unknown shift 'Brunch'",
        "day 5 outside horizon",
    ]
