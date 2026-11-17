"""Independent checker for a schedule. Deliberately does not reuse the MILP model."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from cuopt_scheduler.data import (
    DISLIKE_COST,
    SHIFT_COST,
    UNCOVERED_PENALTY,
    Assignment,
    Problem,
)

Gaps = dict[tuple[int, str, str], int]  # (day, shift, skill) -> staff missing


def violations(problem: Problem, assignments: Iterable[Assignment]) -> list[str]:
    """Every hard rule the schedule breaks (empty if valid).

    Unmet demand is not a violation: see `coverage_gaps`.
    """
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

    for (emp, d), n in Counter((a.employee_id, a.day) for a in assignments).items():
        if n > 1:
            problems.append(f"{emp} works {n} shifts on day {d}")

    for w, week in enumerate(problem.weeks):
        worked = Counter(a.employee_id for a in assignments if a.day in week)
        for emp, n in worked.items():
            hours, cap = n * problem.shift_length, problem.employee(emp).max_weekly_hours
            if hours > cap:
                problems.append(f"{emp} works {hours}h in week {w + 1}, cap is {cap}h")

    for s, cap in problem.weekly_shift_caps.items():
        for w, week in enumerate(problem.weeks):
            worked = Counter(a.employee_id for a in assignments if a.shift == s and a.day in week)
            for emp, n in worked.items():
                if n > cap:
                    problems.append(f"{emp} works {n} {s} shifts in week {w + 1}, cap is {cap}")

    if (k := problem.max_consecutive_days) is not None:
        days_worked: dict[str, set[int]] = defaultdict(set)
        for a in assignments:
            days_worked[a.employee_id].add(a.day)
        for emp, worked_days in days_worked.items():
            run = 0
            for d in problem.days:
                run = run + 1 if d in worked_days else 0
                if run == k + 1:
                    problems.append(f"{emp} works more than {k} days in a row (to day {d})")

    for a in assignments:
        if not problem.employee(a.employee_id).is_available(a.day, a.shift):
            problems.append(f"{a.employee_id} is unavailable for {a.shift} on day {a.day}")

    # Rest: compare actual clock times, independently of how the model encodes it.
    starts: dict[str, list[tuple[int, Assignment]]] = defaultdict(list)
    for a in assignments:
        starts[a.employee_id].append((problem.start_hour(a.day, a.shift), a))
    for emp, worked in starts.items():
        worked.sort()
        for (start1, a1), (start2, a2) in zip(worked, worked[1:], strict=False):
            rest = start2 - (start1 + problem.shift_length)
            if a1.day != a2.day and rest < problem.min_rest_hours:
                problems.append(
                    f"{emp} rests {rest}h between {a1.shift} day {a1.day} and "
                    f"{a2.shift} day {a2.day}, minimum is {problem.min_rest_hours}h"
                )

    return problems


def coverage_gaps(problem: Problem, assignments: Iterable[Assignment]) -> Gaps:
    """Required staff that the schedule leaves uncovered (empty if demand is fully met)."""
    have = Counter((a.day, a.shift, problem.employee(a.employee_id).skill) for a in assignments)
    return {
        (d, s, skill): need - have[d, s, skill]
        for d in problem.days
        for s in problem.shifts
        for skill, need in problem.demand[s].items()
        if have[d, s, skill] < need
    }


def schedule_cost(problem: Problem, assignments: Iterable[Assignment]) -> int:
    """The objective, recomputed from the schedule: what the solvers minimise."""
    assignments = list(assignments)
    disliked = sum(
        (a.day, a.shift) in problem.employee(a.employee_id).dislikes for a in assignments
    )
    return (
        UNCOVERED_PENALTY * sum(coverage_gaps(problem, assignments).values())
        + SHIFT_COST * len(assignments)
        + DISLIKE_COST * disliked
    )
