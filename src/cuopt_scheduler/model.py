"""MILP formulation of the rostering problem."""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from cuopt_scheduler.data import Problem

VarKey = tuple[str, str, int]  # (employee_id, shift, day)


@dataclass
class ScheduleModel:
    lp: pulp.LpProblem
    x: dict[VarKey, pulp.LpVariable]


def build_model(problem: Problem) -> ScheduleModel:
    shifts, days = problem.shifts, problem.days
    keys = [(e.id, s, d) for e in problem.employees for s in shifts for d in days]

    lp = pulp.LpProblem("Nurse_Scheduling", pulp.LpMinimize)
    x = {(e, s, d): pulp.LpVariable(f"assign_{e}_{s}_d{d}", cat=pulp.LpBinary) for e, s, d in keys}

    # Objective: use as few shifts as possible.
    lp += pulp.lpSum(x.values()), "total_shifts"

    # 1. Staffing demand per day, shift and skill.
    for d in days:
        for s in shifts:
            for skill, count in problem.demand[s].items():
                lp += (
                    pulp.lpSum(x[e.id, s, d] for e in problem.employees if e.skill == skill)
                    >= count,
                    f"demand_{s}_{skill}_d{d}",
                )

    for e in problem.employees:
        # 2. Max hours over the whole horizon.
        lp += (
            pulp.lpSum(x[e.id, s, d] for s in shifts for d in days) * problem.shift_length
            <= e.max_hours,
            f"max_hours_{e.id}",
        )
        for d in days:
            # 3. At most one shift per day.
            lp += pulp.lpSum(x[e.id, s, d] for s in shifts) <= 1, f"one_shift_{e.id}_d{d}"
            # 4. Availability.
            for s in shifts:
                if not e.is_available(d, s):
                    lp += x[e.id, s, d] == 0, f"unavailable_{e.id}_{s}_d{d}"

    return ScheduleModel(lp=lp, x=x)
