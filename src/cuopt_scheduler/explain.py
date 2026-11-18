"""The numbers behind the app's "How it was solved" tab: the model's size and rules, the LP
relaxation next to the rounded schedule, the cost broken into its parts, and the bound that
proves (or doesn't) that no schedule is cheaper.

Everything here is computed from the same arrays the solvers use, so what the tab shows is
what the solver saw.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cuopt_scheduler.data import (
    DISLIKE_COST,
    SHIFT_COST,
    UNCOVERED_PENALTY,
    Assignment,
    Problem,
)
from cuopt_scheduler.lp_round import LPEngine, round_lp, solve_lp, with_shortfall
from cuopt_scheduler.model import FloatArray, ScheduleModel, build_model
from cuopt_scheduler.solver import _ABS_GAP, Backend, cuopt_available


@dataclass(frozen=True)
class Explanation:
    model: ScheduleModel
    engine: LPEngine
    lp: FloatArray
    """The LP relaxation's solution (all columns): fractions allowed."""
    rounded: FloatArray
    """The LP solution rounded into a valid schedule (all columns)."""
    bound: float
    """No schedule can cost less than this (from the LP's duals)."""
    lp_seconds: float

    @property
    def num_decisions(self) -> int:
        return len(self.model.x_keys)

    @property
    def num_fixed(self) -> int:
        """Decisions fixed to "no" because the person is unavailable."""
        return int(np.sum(self.model.col_upper[: self.num_decisions] == 0))

    @property
    def num_constraints(self) -> int:
        return self.model.num_rows

    @property
    def share_already_whole(self) -> float:
        """Fraction of the LP's yes/no decisions that came out (almost) exactly 0 or 1."""
        x = self.lp[: self.num_decisions]
        if not len(x):
            return 1.0
        return float(np.mean(np.minimum(np.abs(x), np.abs(1 - x)) < 1e-3))


def explain(problem: Problem, backend: Backend = Backend.CPU) -> Explanation:
    """Solve `problem`'s LP relaxation (on the GPU if asked and possible), round it and
    bound it."""
    model = build_model(problem)
    use_gpu = backend is Backend.GPU and cuopt_available()
    engine = LPEngine.CUOPT_PDLP if use_gpu else LPEngine.HIGHS
    start = time.perf_counter()
    lp = solve_lp(model, engine, time_limit=60)
    lp_seconds = time.perf_counter() - start
    return Explanation(
        model=model,
        engine=engine,
        lp=lp.primal,
        rounded=round_lp(model, lp.primal),
        bound=model.lower_bound(lp.duals),
        lp_seconds=lp_seconds,
    )


def schedule_vector(model: ScheduleModel, assignments: Iterable[Assignment]) -> FloatArray:
    """A schedule as the solver's solution vector (assignments plus shortfall)."""
    column = {key: j for j, key in enumerate(model.x_keys)}
    x = np.zeros(model.num_cols)
    for a in assignments:
        x[column[(a.employee_id, a.shift, a.day)]] = 1.0
    return with_shortfall(model, x)


def cost_breakdown(model: ScheduleModel, values: FloatArray) -> pd.DataFrame:
    """The objective split into what it is made of; the rows add up to `cost @ values`."""
    n_x = len(model.x_keys)
    x, slack = values[:n_x], values[n_x:]
    extra = model.cost[:n_x] - SHIFT_COST  # disliked shifts cost DISLIKE_COST more
    rows = [
        ("Shifts worked", x.sum(), SHIFT_COST, SHIFT_COST * x.sum()),
        ("Disliked shifts (extra)", x @ (extra > 0), DISLIKE_COST, x @ extra),
        ("Uncovered staff slots", slack.sum(), UNCOVERED_PENALTY, model.cost[n_x:] @ slack),
    ]
    return pd.DataFrame(
        [(part, round(float(n)), each, float(cost)) for part, n, each, cost in rows],
        columns=pd.Index(["Part", "Count", "Cost each", "Cost"]),
    )


def is_proven(objective: float, bound: float) -> bool:
    """True if the bound proves `objective` optimal: costs are whole numbers, so a gap under 1
    leaves no room for a cheaper schedule (the same rule the solvers use)."""
    return math.isfinite(bound) and objective - bound <= _ABS_GAP + 1e-6


def decision_grid(
    problem: Problem,
    model: ScheduleModel,
    values: FloatArray,
    max_employees: int = 12,
    max_days: int = 7,
) -> pd.DataFrame:
    """A slice of the yes/no decisions: one row per employee ("Name (ID)"), one column per
    (day index, shift).

    Cells are the decision's value (0 to 1; fractions for an LP solution), or NaN where the
    person is unavailable, so the decision is fixed to 0.
    """
    employees = problem.employees[:max_employees]
    column = {key: j for j, key in enumerate(model.x_keys)}
    data = {
        (d, s): [
            np.nan if model.col_upper[j] == 0 else float(values[j])
            for j in (column[(e.id, s, d)] for e in employees)
        ]
        for d in list(problem.days)[:max_days]
        for s in problem.shifts
    }
    frame = pd.DataFrame(data, index=pd.Index([f"{e.name} ({e.id})" for e in employees]))
    frame.columns = pd.MultiIndex.from_tuples(list(data), names=["Day", "Shift"])
    return frame
