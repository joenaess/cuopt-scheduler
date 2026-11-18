"""Problem definition and synthetic data generation."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from types import MappingProxyType

from faker import Faker

SKILLS: tuple[str, ...] = ("RN", "LPN", "CNA")
SKILL_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "RN": "RN (Registered Nurse)",
        "LPN": "LPN (Lic. Practical Nurse)",
        "CNA": "CNA (Cert. Nursing Asst)",
    }
)
SHIFTS: tuple[str, ...] = ("Day", "Evening", "Night")
SHIFT_START_HOUR: Mapping[str, int] = MappingProxyType({"Day": 7, "Evening": 15, "Night": 23})
SHIFT_LENGTH_HOURS = 8
MIN_REST_HOURS = 11  # common working-time rule (e.g. EU Working Time Directive)
DAYS_PER_WEEK = 7

# Objective weights. Each uncovered staff slot must cost more than the most that covering one
# more slot could ever add elsewhere, so coverage always comes first.
UNCOVERED_PENALTY = 1000
SHIFT_COST = 1
DISLIKE_COST = 1
"""Extra cost of giving someone a shift they marked as disliked."""

LEVELS: Mapping[int, str] = MappingProxyType(
    {
        0: "Base rules: skills, weekly hours, availability, one shift a day, 11h rest",
        1: "+ at most 5 consecutive working days",
        2: "+ at most 2 night shifts per week",
        3: "+ each person dislikes ~20% of shifts, which cost extra",
    }
)
"""Difficulty levels for `generate_problem`. Each adds to the one before."""

# Staff needed per shift and skill. Same every day.
DEFAULT_DEMAND: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        "Day": MappingProxyType({"RN": 2, "LPN": 2, "CNA": 3}),
        "Evening": MappingProxyType({"RN": 1, "LPN": 2, "CNA": 2}),
        "Night": MappingProxyType({"RN": 1, "LPN": 1, "CNA": 1}),
    }
)


@dataclass(frozen=True)
class Employee:
    id: str
    name: str
    skill: str
    max_weekly_hours: int
    unavailable: frozenset[tuple[int, str]] = field(default_factory=frozenset)
    """(day, shift) pairs the employee cannot work."""
    dislikes: frozenset[tuple[int, str]] = field(default_factory=frozenset)
    """(day, shift) pairs the employee can work but would rather not (cost DISLIKE_COST)."""

    def is_available(self, day: int, shift: str) -> bool:
        return (day, shift) not in self.unavailable


@dataclass(frozen=True)
class Problem:
    employees: tuple[Employee, ...]
    num_days: int
    demand: Mapping[str, Mapping[str, int]] = field(default_factory=lambda: DEFAULT_DEMAND)
    shift_length: int = SHIFT_LENGTH_HOURS
    shift_start: Mapping[str, int] = field(default_factory=lambda: SHIFT_START_HOUR)
    """Start hour (0-23) of each shift. A shift may run past midnight."""
    min_rest_hours: int = MIN_REST_HOURS
    """Minimum hours off between the end of one shift and the start of the next."""
    max_consecutive_days: int | None = None
    """Most days in a row anyone may work (None = no limit)."""
    weekly_shift_caps: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    """Most shifts of a given type per person per calendar week, e.g. {"Night": 2}."""

    def __post_init__(self) -> None:
        ids = [e.id for e in self.employees]
        if len(ids) != len(set(ids)):
            raise ValueError("employee ids must be unique")
        if self.num_days < 1:
            raise ValueError("num_days must be >= 1")
        if missing := set(self.demand) - set(self.shift_start):
            raise ValueError(f"no start hour for shifts {sorted(missing)}")
        if self.max_consecutive_days is not None and self.max_consecutive_days < 1:
            raise ValueError("max_consecutive_days must be >= 1")
        if unknown := set(self.weekly_shift_caps) - set(self.demand):
            raise ValueError(f"weekly_shift_caps for unknown shifts {sorted(unknown)}")

    @property
    def shifts(self) -> tuple[str, ...]:
        return tuple(self.demand)

    @property
    def days(self) -> range:
        return range(self.num_days)

    @property
    def weeks(self) -> list[range]:
        """Days split into weeks of 7; the last week may be shorter."""
        return [
            range(start, min(start + DAYS_PER_WEEK, self.num_days))
            for start in range(0, self.num_days, DAYS_PER_WEEK)
        ]

    def start_hour(self, day: int, shift: str) -> int:
        """Absolute start time in hours from the beginning of day 0."""
        return 24 * day + self.shift_start[shift]

    def employee(self, employee_id: str) -> Employee:
        return self._by_id[employee_id]

    @cached_property
    def _by_id(self) -> dict[str, Employee]:
        return {e.id: e for e in self.employees}


def generate_problem(
    num_employees: int,
    num_days: int,
    *,
    seed: int | None = None,
    wards: int = 1,
    level: int = 0,
) -> Problem:
    """Generate a random workforce. The same seed always gives the same problem.

    `wards` multiplies the staffing demand (one ward needs DEFAULT_DEMAND). Roughly 45
    employees per ward cover all demand; fewer leave gaps. `level` adds rules (see LEVELS);
    the same seed gives the same staff at every level.
    """
    if wards < 1:
        raise ValueError("wards must be >= 1")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)}")
    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(rng.getrandbits(32))

    employees = []
    for i in range(num_employees):
        is_full_time = rng.random() > 0.3
        max_hours = 40 if is_full_time else rng.choice([24, 32])
        skill = rng.choice(SKILLS)

        unavailable: set[tuple[int, str]] = set()
        for day in range(num_days):
            if rng.random() < 0.15:  # requested the whole day off
                unavailable.update((day, s) for s in SHIFTS)
            elif rng.random() < 0.2:  # can't work one particular shift
                unavailable.add((day, rng.choice(SHIFTS)))

        employees.append(
            Employee(
                id=f"E{i + 1:03d}",
                name=f"{fake.first_name()} {fake.last_name()}",
                skill=skill,
                max_weekly_hours=max_hours,
                unavailable=frozenset(unavailable),
            )
        )
    if level >= 3:
        # A separate stream, so lower levels' staff are unchanged.
        dislike_rng = random.Random(rng.getrandbits(64))
        employees = [
            replace(
                e,
                dislikes=frozenset(
                    (d, s)
                    for d in range(num_days)
                    for s in SHIFTS
                    if (d, s) not in e.unavailable and dislike_rng.random() < 0.2
                ),
            )
            for e in employees
        ]
    demand = {s: {k: v * wards for k, v in need.items()} for s, need in DEFAULT_DEMAND.items()}
    return Problem(
        employees=tuple(employees),
        num_days=num_days,
        demand=demand,
        max_consecutive_days=5 if level >= 1 else None,
        weekly_shift_caps=MappingProxyType({"Night": 2} if level >= 2 else {}),
    )


@dataclass(frozen=True, order=True)
class Assignment:
    day: int
    shift: str
    employee_id: str
