"""Problem definition and synthetic data generation."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
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
    max_hours: int
    unavailable: frozenset[tuple[int, str]] = field(default_factory=frozenset)
    """(day, shift) pairs the employee cannot work."""

    def is_available(self, day: int, shift: str) -> bool:
        return (day, shift) not in self.unavailable


@dataclass(frozen=True)
class Problem:
    employees: tuple[Employee, ...]
    num_days: int
    demand: Mapping[str, Mapping[str, int]] = field(default_factory=lambda: DEFAULT_DEMAND)
    shift_length: int = SHIFT_LENGTH_HOURS

    def __post_init__(self) -> None:
        ids = [e.id for e in self.employees]
        if len(ids) != len(set(ids)):
            raise ValueError("employee ids must be unique")
        if self.num_days < 1:
            raise ValueError("num_days must be >= 1")

    @property
    def shifts(self) -> tuple[str, ...]:
        return tuple(self.demand)

    @property
    def days(self) -> range:
        return range(self.num_days)

    def employee(self, employee_id: str) -> Employee:
        return self._by_id[employee_id]

    @cached_property
    def _by_id(self) -> dict[str, Employee]:
        return {e.id: e for e in self.employees}


def generate_problem(num_employees: int, num_days: int, *, seed: int | None = None) -> Problem:
    """Generate a random workforce. The same seed always gives the same problem."""
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
                max_hours=max_hours,
                unavailable=frozenset(unavailable),
            )
        )
    return Problem(employees=tuple(employees), num_days=num_days)
