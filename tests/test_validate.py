"""The validator is the oracle for the solver tests, so it gets its own tests."""

from cuopt_scheduler import Assignment, Problem, violations

from .conftest import SMALL_DEMAND, rn

A = Assignment


def one_day(*employees) -> Problem:
    return Problem(employees=employees, num_days=1, demand=SMALL_DEMAND)


def test_valid_schedule_has_no_violations() -> None:
    p = one_day(rn("A"), rn("B"))
    assert violations(p, [A(0, "Day", "A"), A(0, "Night", "B")]) == []


def test_detects_unmet_demand() -> None:
    p = one_day(rn("A"), rn("B"))
    assert violations(p, [A(0, "Day", "A")]) == ["day 0 Night: RN needs 1, has 0"]


def test_wrong_skill_does_not_count() -> None:
    from dataclasses import replace

    p = one_day(rn("A"), replace(rn("B"), skill="CNA"))
    assert "day 0 Night: RN needs 1, has 0" in violations(p, [A(0, "Day", "A"), A(0, "Night", "B")])


def test_detects_two_shifts_same_day() -> None:
    p = one_day(rn("A"))
    assert "A works 2 shifts on day 0" in violations(p, [A(0, "Day", "A"), A(0, "Night", "A")])


def test_detects_max_hours() -> None:
    p = Problem(employees=(rn("A", max_hours=8),), num_days=2, demand={"Day": {"RN": 1}})
    assert violations(p, [A(0, "Day", "A"), A(1, "Day", "A")]) == ["A works 16h, cap is 8h"]


def test_detects_unavailability() -> None:
    p = Problem(
        employees=(rn("A", unavailable={(0, "Day")}),), num_days=1, demand={"Day": {"RN": 1}}
    )
    assert violations(p, [A(0, "Day", "A")]) == ["A is unavailable for Day on day 0"]


def test_detects_duplicates() -> None:
    p = Problem(employees=(rn("A"),), num_days=1, demand={"Day": {"RN": 1}})
    assert violations(p, [A(0, "Day", "A"), A(0, "Day", "A")]) == [
        "A assigned 2 times to Day on day 0",
        "A works 2 shifts on day 0",
    ]


def test_detects_unknown_references() -> None:
    p = one_day(rn("A"))
    assert violations(p, [A(5, "Brunch", "Z")]) == [
        "unknown employee Z",
        "unknown shift 'Brunch'",
        "day 5 outside horizon",
    ]
