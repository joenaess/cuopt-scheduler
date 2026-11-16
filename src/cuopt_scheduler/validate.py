"""Independent checker for a schedule. Deliberately does not reuse the MILP model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from cuopt_scheduler.data import Problem
from cuopt_scheduler.solver import Assignment


def violations(problem: Problem, assignments: Iterable[Assignment]) -> list[str]:
    """Return a human-readable list of every rule the schedule breaks (empty if valid)."""
    assignments = list(assignments)
    known = {e.id for e in problem.employees}
    problems: list[str] = []

    for a in assignments:
        if a.employee_id not in known:
            problems.append(f"unknown employee {a.employee_id}")
        if a.shift not in problem.shifts:
            problems.append(f"unknown shift {a.shift!r}")
        if a.day not in problem.days:
            problems.append(f"day {a.day} outside horizon")
    if problems:
        return problems

    for (emp, s, d), n in Counter((a.employee_id, a.shift, a.day) for a in assignments).items():
        if n > 1:
            problems.append(f"{emp} assigned {n} times to {s} on day {d}")

    covered = Counter((a.day, a.shift, problem.employee(a.employee_id).skill) for a in assignments)
    for d in problem.days:
        for s in problem.shifts:
            for skill, need in problem.demand[s].items():
                if (have := covered[d, s, skill]) < need:
                    problems.append(f"day {d} {s}: {skill} needs {need}, has {have}")

    per_day = Counter((a.employee_id, a.day) for a in assignments)
    for (emp, d), n in per_day.items():
        if n > 1:
            problems.append(f"{emp} works {n} shifts on day {d}")

    shifts_worked = Counter(a.employee_id for a in assignments)
    for emp, n in shifts_worked.items():
        hours, cap = n * problem.shift_length, problem.employee(emp).max_hours
        if hours > cap:
            problems.append(f"{emp} works {hours}h, cap is {cap}h")

    for a in assignments:
        if not problem.employee(a.employee_id).is_available(a.day, a.shift):
            problems.append(f"{a.employee_id} is unavailable for {a.shift} on day {a.day}")

    return problems
