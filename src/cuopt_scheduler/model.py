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

from cuopt_scheduler.data import DISLIKE_COST, SHIFT_COST, UNCOVERED_PENALTY, Problem

VarKey = tuple[str, str, int]  # (employee_id, shift, day)
GapKey = tuple[int, str, str]  # (day, shift, skill)
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]


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

    def lower_bound(self, row_duals: FloatArray) -> float:
        """A guaranteed lower bound on the optimum (of the LP relaxation, so also of the MILP)
        from any row multipliers, e.g. an approximate LP solver's duals.

        The Lagrangian bound: for multipliers y, every feasible v has
        cost @ v >= y @ (row bound) + sum_j min over [col_lower, col_upper] of (cost - A.T y)_j v_j,
        taking each row's lower bound where y > 0 and its upper bound where y < 0. Multipliers
        with the wrong sign for a row (pointing at an infinite bound) are set to 0 first, so
        the bound holds whatever `row_duals` is. It is tight only if the duals are close to
        optimal. Floating-point error is subtracted, so the result can be trusted as a proof.
        """
        y = np.asarray(row_duals, dtype=np.float64)
        if y.shape != (self.num_rows,) or not np.all(np.isfinite(y)):
            return -np.inf
        has_lower, has_upper = np.isfinite(self.row_lower), np.isfinite(self.row_upper)
        y = np.where(has_lower, y, np.minimum(y, 0.0))
        y = np.where(has_upper, y, np.maximum(y, 0.0))
        rows = np.repeat(np.arange(self.num_rows), np.diff(self.offsets))
        aty = np.bincount(self.indices, weights=self.values * y[rows], minlength=self.num_cols)
        reduced = self.cost - aty
        row_bound = np.where(y > 0, self.row_lower, self.row_upper)
        row_terms = np.where(y != 0, y * np.where(y != 0, row_bound, 0.0), 0.0)
        col_bound = np.where(reduced > 0, self.col_lower, self.col_upper)
        if np.any((reduced != 0) & ~np.isfinite(col_bound)):
            return -np.inf
        col_terms = reduced * np.where(reduced != 0, col_bound, 0.0)
        # Summing ~10^7 float64 terms loses far less than 1e-9 of their total magnitude.
        error = 1e-9 * (np.abs(row_terms).sum() + np.abs(col_terms).sum() + np.abs(aty).sum())
        return float(row_terms.sum() + col_terms.sum() - error)

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
    blocks.extend(
        _Block(x[:, :, week.start : week.stop].reshape(n_emp, -1), -np.inf, max_shifts)
        for week in problem.weeks
    )

    # 3. At most one shift per day.
    blocks.append(_Block(x.transpose(0, 2, 1).reshape(-1, n_shift), -np.inf, 1.0))

    # 4. Most days in a row: any window of k + 1 days has at most k working days.
    if (k := problem.max_consecutive_days) is not None:
        for start in range(n_day - k):
            window = x[:, :, start : start + k + 1].reshape(n_emp, -1)
            blocks.append(_Block(window, -np.inf, float(k)))

    # 5. Per-shift weekly caps (e.g. nights).
    for s, cap in problem.weekly_shift_caps.items():
        si = shifts.index(s)
        blocks.extend(
            _Block(x[:, si, week.start : week.stop], -np.inf, float(cap)) for week in problem.weeks
        )

    # 6. Minimum rest between consecutive days' shifts.
    for today, tomorrow in rest_conflicts(problem):
        t, m = shifts.index(today), shifts.index(tomorrow)
        pairs = np.stack([x[:, t, :-1], x[:, m, 1:]], axis=-1).reshape(-1, 2)
        blocks.append(_Block(pairs, -np.inf, 1.0))

    n_cols = n_x + len(gap_keys)
    col_upper = np.concatenate([np.ones(n_x), np.array(gap_upper)])
    cost = np.concatenate(
        [np.full(n_x, float(SHIFT_COST)), np.full(len(gap_keys), float(UNCOVERED_PENALTY))]
    )
    for i, e in enumerate(employees):
        # 7. Availability, as a column bound rather than a constraint.
        for d, s in e.unavailable:
            if d in days and s in shifts:
                col_upper[x[i, shifts.index(s), d]] = 0.0
        for d, s in e.dislikes:
            if d in days and s in shifts:
                cost[x[i, shifts.index(s), d]] += DISLIKE_COST

    lengths = np.concatenate([np.full(len(b.cols), b.cols.shape[1]) for b in blocks])
    return ScheduleModel(
        x_keys=tuple((e.id, s, d) for e, s, d in itertools.product(employees, shifts, days)),
        gap_keys=tuple(gap_keys),
        # Objective: cover as much demand as possible, then use as few (and as few disliked)
        # shifts as possible.
        cost=cost,
        col_lower=np.zeros(n_cols),
        col_upper=col_upper,
        integer=np.arange(n_cols) < n_x,
        row_lower=np.concatenate([np.broadcast_to(b.lower, len(b.cols)) for b in blocks]),
        row_upper=np.concatenate([np.broadcast_to(b.upper, len(b.cols)) for b in blocks]),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32),
        indices=np.concatenate([b.cols.ravel() for b in blocks]).astype(np.int32),
        values=np.ones(int(lengths.sum())),
    )
