"""MILP formulation of the rostering problem."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pulp

from cuopt_scheduler.data import Problem

VarKey = tuple[str, str, int]  # (employee_id, shift, day)
GapKey = tuple[int, str, str]  # (day, shift, skill)

# Cost of leaving one required staff slot empty, in "shifts". It must exceed the most shifts
# that covering one more slot could ever add, so coverage always comes first.
UNCOVERED_PENALTY = 1000


@dataclass
class ScheduleModel:
    lp: pulp.LpProblem
    x: dict[VarKey, pulp.LpVariable]
    """1 if the employee works that shift on that day."""
    uncovered: dict[GapKey, pulp.LpVariable]
    """Required staff slots left empty (penalised in the objective)."""


def rest_conflicts(problem: Problem) -> Iterator[tuple[str, str]]:
    """Shift pairs (today, tomorrow) that leave less than the minimum rest in between."""
    for today in problem.shifts:
        end = problem.start_hour(0, today) + problem.shift_length
        for tomorrow in problem.shifts:
            if problem.start_hour(1, tomorrow) - end < problem.min_rest_hours:
                yield today, tomorrow


def build_model(problem: Problem) -> ScheduleModel:
    shifts, days = problem.shifts, problem.days
    x = {
        (e.id, s, d): pulp.LpVariable(f"assign_{e.id}_{s}_d{d}", cat=pulp.LpBinary)
        for e in problem.employees
        for s in shifts
        for d in days
    }
    uncovered = {
        (d, s, skill): pulp.LpVariable(f"uncovered_{s}_{skill}_d{d}", lowBound=0)
        for d in days
        for s in shifts
        for skill in problem.demand[s]
    }

    lp = pulp.LpProblem("Nurse_Scheduling", pulp.LpMinimize)
    # Objective: cover as much demand as possible, then use as few shifts as possible.
    lp += UNCOVERED_PENALTY * pulp.lpSum(uncovered.values()) + pulp.lpSum(x.values()), "cost"

    # 1. Staffing demand per day, shift and skill (shortfall allowed but penalised).
    for (d, s, skill), gap in uncovered.items():
        staff = pulp.lpSum(x[e.id, s, d] for e in problem.employees if e.skill == skill)
        lp += staff + gap >= problem.demand[s][skill], f"demand_{s}_{skill}_d{d}"

    conflicts = list(rest_conflicts(problem))
    for e in problem.employees:
        # 2. Max hours per calendar week.
        for w, week in enumerate(problem.weeks):
            hours = pulp.lpSum(x[e.id, s, d] for s in shifts for d in week) * problem.shift_length
            lp += hours <= e.max_weekly_hours, f"max_hours_{e.id}_w{w}"
        for d in days:
            # 3. At most one shift per day.
            lp += pulp.lpSum(x[e.id, s, d] for s in shifts) <= 1, f"one_shift_{e.id}_d{d}"
            # 4. Minimum rest between consecutive days' shifts.
            if d + 1 < problem.num_days:
                for today, tomorrow in conflicts:
                    lp += (
                        x[e.id, today, d] + x[e.id, tomorrow, d + 1] <= 1,
                        f"rest_{e.id}_{today}_{tomorrow}_d{d}",
                    )
            # 5. Availability.
            for s in shifts:
                if not e.is_available(d, s):
                    lp += x[e.id, s, d] == 0, f"unavailable_{e.id}_{s}_d{d}"

    return ScheduleModel(lp=lp, x=x, uncovered=uncovered)
