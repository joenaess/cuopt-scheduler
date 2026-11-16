"""Convert problems and results into pandas DataFrames for display and export."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from cuopt_scheduler.data import SKILL_LABELS, Assignment, Problem
from cuopt_scheduler.validate import Gaps

# A Monday, so day 0 is Mon in both the labels and the Gantt axis.
BASE_DATE = pd.Timestamp("2024-01-01")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def day_label(day: int) -> str:
    return f"Day {day + 1} ({WEEKDAYS[day % 7]})"


def employees_frame(problem: Problem) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": e.id,
            "Name": e.name,
            "Skill": SKILL_LABELS.get(e.skill, e.skill),
            "Max_Hours_Per_Week": e.max_weekly_hours,
            "Days_Off": sum(
                all(not e.is_available(d, s) for s in problem.shifts) for d in problem.days
            ),
        }
        for e in problem.employees
    ).set_index("ID")


def demand_frame(problem: Problem) -> pd.DataFrame:
    return pd.DataFrame(
        {SKILL_LABELS.get(k, k): v for k, v in by_skill.items()}
        for by_skill in problem.demand.values()
    ).set_axis(list(problem.demand), axis=0)


def schedule_frame(problem: Problem, assignments: Iterable[Assignment]) -> pd.DataFrame:
    rows = []
    for a in sorted(assignments):
        emp = problem.employee(a.employee_id)
        start = BASE_DATE + pd.Timedelta(hours=problem.start_hour(a.day, a.shift))
        rows.append(
            {
                "Day_Index": a.day,
                "Day": day_label(a.day),
                "Shift": a.shift,
                "Employee_ID": emp.id,
                "Employee": emp.name,
                "Skill": SKILL_LABELS.get(emp.skill, emp.skill),
                "Start_Time": start,
                "End_Time": start + pd.Timedelta(hours=problem.shift_length),
            }
        )
    columns = ["Day_Index", "Day", "Shift", "Employee_ID", "Employee", "Skill"]
    return pd.DataFrame(rows, columns=pd.Index([*columns, "Start_Time", "End_Time"]))


def gaps_frame(problem: Problem, gaps: Gaps) -> pd.DataFrame:
    """One row per (day, shift, skill) that is short of staff."""
    rows = [
        {
            "Day": day_label(d),
            "Shift": s,
            "Skill": SKILL_LABELS.get(skill, skill),
            "Required": problem.demand[s][skill],
            "Missing": missing,
        }
        for (d, s, skill), missing in sorted(gaps.items())
    ]
    return pd.DataFrame(rows, columns=pd.Index(["Day", "Shift", "Skill", "Required", "Missing"]))
