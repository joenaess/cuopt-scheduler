from __future__ import annotations

import pytest

from cuopt_scheduler import Employee, Problem, cuopt_available

# Two shifts, one RN each, per day: small enough to reason about by hand.
SMALL_DEMAND = {"Day": {"RN": 1}, "Night": {"RN": 1}}


def rn(
    id_: str, max_weekly_hours: int = 40, unavailable: set[tuple[int, str]] | None = None
) -> Employee:
    return Employee(
        id=id_,
        name=f"Nurse {id_}",
        skill="RN",
        max_weekly_hours=max_weekly_hours,
        unavailable=frozenset(unavailable or ()),
    )


@pytest.fixture
def small_feasible() -> Problem:
    # 3 days x 2 shifts = 6 slots; 3 nurses x 2 shifts (16h cap) = 6. Exactly tight.
    return Problem(
        employees=(rn("A", 16), rn("B", 16), rn("C", 16)), num_days=3, demand=SMALL_DEMAND
    )


@pytest.fixture
def understaffed() -> Problem:
    # Each day needs 2 different nurses, but on day 1 nobody can work Night: 1 slot uncovered.
    return Problem(
        employees=(rn("A", unavailable={(1, "Night")}), rn("B", unavailable={(1, "Night")})),
        num_days=2,
        demand=SMALL_DEMAND,
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if cuopt_available():
        return
    skip = pytest.mark.skip(reason="cuOpt / NVIDIA GPU not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
