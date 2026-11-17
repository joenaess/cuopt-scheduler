"""MILP formulation of the rostering problem, as plain arrays both solvers accept directly.

Columns are the assignment variables x[employee, shift, day] (binary), followed by one
`uncovered` slack per (day, shift, skill) (continuous). Rows are stored in CSR form.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cuopt_scheduler.data import Problem

VarKey = tuple[str, str, int]  # (employee_id, shift, day)
GapKey = tuple[int, str, str]  # (day, shift, skill)
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]

# Cost of leaving one required staff slot empty, in "shifts". It must exceed the most shifts
# that covering one more slot could ever add, so coverage always comes first.
UNCOVERED_PENALTY = 1000


@dataclass(frozen=True, eq=False)
class ScheduleModel:
    """min cost @ v  subject to  row_lower <= A @ v <= row_upper,  col_lower <= v <= col_upper."""

    x_keys: tuple[VarKey, ...]
    """Key of each assignment column; these are columns 0 .. len(x_keys) - 1."""
    gap_keys: tuple[GapKey, ...]
    """Key of each `uncovered` column; these follow the assignment columns."""
    cost: FloatArray
    col_lower: FloatArray
    col_upper: FloatArray
    integer: npt.NDArray[np.bool_]
    row_lower: FloatArray
    row_upper: FloatArray
    offsets: IntArray
    """CSR: row i's entries are indices/values[offsets[i]:offsets[i + 1]]."""
    indices: IntArray
    values: FloatArray

    @property
    def num_cols(self) -> int:
        return len(self.cost)

    @property
    def num_rows(self) -> int:
        return len(self.row_lower)

    def row_activity(self, v: FloatArray) -> FloatArray:
        """A @ v."""
        rows = np.repeat(np.arange(self.num_rows), np.diff(self.offsets))
        weights = self.values * v[self.indices]
        # With weights, bincount returns float64; numpy's stubs only know the unweighted form.
        return np.bincount(rows, weights=weights, minlength=self.num_rows).astype(np.float64)

    def is_feasible(self, v: FloatArray, tol: float = 1e-6) -> bool:
        """True if `v` is a valid solution: right shape, integral where required, in bounds."""
        if v.shape != (self.num_cols,) or not np.all(np.isfinite(v)):
            return False
        if np.any(np.abs(v[self.integer] - np.round(v[self.integer])) > tol):
            return False
        if np.any(v < self.col_lower - tol) or np.any(v > self.col_upper + tol):
            return False
        activity = self.row_activity(v)
        return bool(
            np.all(activity >= self.row_lower - tol) and np.all(activity <= self.row_upper + tol)
        )


def rest_conflicts(problem: Problem) -> Iterator[tuple[str, str]]:
    """Shift pairs (today, tomorrow) that leave less than the minimum rest in between."""
    for today in problem.shifts:
        end = problem.start_hour(0, today) + problem.shift_length
        for tomorrow in problem.shifts:
            if problem.start_hour(1, tomorrow) - end < problem.min_rest_hours:
                yield today, tomorrow


@dataclass
class _Block:
    """Rows that all have the same number of entries (each with coefficient 1)."""

    cols: IntArray  # shape (rows, entries per row)
    lower: FloatArray | float
    upper: FloatArray | float


def build_model(problem: Problem) -> ScheduleModel:
    employees, shifts, days = problem.employees, problem.shifts, problem.days
    n_emp, n_shift, n_day = len(employees), len(shifts), problem.num_days
    x = np.arange(n_emp * n_shift * n_day, dtype=np.int32).reshape(n_emp, n_shift, n_day)
    n_x = x.size
    blocks: list[_Block] = []

    # 1. Staffing demand per day, shift and skill: staff + uncovered >= need.
    gap_keys: list[GapKey] = []
    gap_upper: list[float] = []
    for si, s in enumerate(shifts):
        for skill, need in problem.demand[s].items():
            who = np.array([i for i, e in enumerate(employees) if e.skill == skill], np.int32)
            gap = (n_x + len(gap_keys) + np.arange(n_day, dtype=np.int32))[:, None]
            blocks.append(_Block(np.hstack([x[who, si, :].T, gap]), float(need), np.inf))
            gap_keys += [(d, s, skill) for d in days]
            gap_upper += [float(need)] * n_day

    # 2. Max hours per calendar week, as a shift count (all shifts have the same length).
    max_shifts = np.array(
        [e.max_weekly_hours // problem.shift_length for e in employees], np.float64
    )
    for week in problem.weeks:
        blocks.append(
            _Block(x[:, :, week.start : week.stop].reshape(n_emp, -1), -np.inf, max_shifts)
        )

    # 3. At most one shift per day.
    blocks.append(_Block(x.transpose(0, 2, 1).reshape(-1, n_shift), -np.inf, 1.0))

    # 4. Minimum rest between consecutive days' shifts.
    for today, tomorrow in rest_conflicts(problem):
        t, m = shifts.index(today), shifts.index(tomorrow)
        pairs = np.stack([x[:, t, :-1], x[:, m, 1:]], axis=-1).reshape(-1, 2)
        blocks.append(_Block(pairs, -np.inf, 1.0))

    n_cols = n_x + len(gap_keys)
    col_upper = np.concatenate([np.ones(n_x), np.array(gap_upper)])
    # 5. Availability, as a column bound rather than a constraint.
    for i, e in enumerate(employees):
        for d, s in e.unavailable:
            if d in days and s in shifts:
                col_upper[x[i, shifts.index(s), d]] = 0.0

    lengths = np.concatenate([np.full(len(b.cols), b.cols.shape[1]) for b in blocks])
    return ScheduleModel(
        x_keys=tuple((e.id, s, d) for e, s, d in itertools.product(employees, shifts, days)),
        gap_keys=tuple(gap_keys),
        # Objective: cover as much demand as possible, then use as few shifts as possible.
        cost=np.concatenate([np.ones(n_x), np.full(len(gap_keys), float(UNCOVERED_PENALTY))]),
        col_lower=np.zeros(n_cols),
        col_upper=col_upper,
        integer=np.arange(n_cols) < n_x,
        row_lower=np.concatenate([np.broadcast_to(b.lower, len(b.cols)) for b in blocks]),
        row_upper=np.concatenate([np.broadcast_to(b.upper, len(b.cols)) for b in blocks]),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32),
        indices=np.concatenate([b.cols.ravel() for b in blocks]).astype(np.int32),
        values=np.ones(int(lengths.sum())),
    )
