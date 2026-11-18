"""A heuristic that can run its heavy part on a GPU LP solver: solve the LP relaxation, then
round it into a valid schedule.

For this model the LP bound equals the integer optimum (see the README), so a good LP
solution is close to a good schedule. Rounding never breaks a rule because every rule is a
packing row (a sum of assignments <= a limit): assignments are only ever added while every
packing row they touch still has room. Demand rows have a slack column that absorbs any
shortfall.

The LP solver's duals also give a guaranteed lower bound (`ScheduleModel.lower_bound`). When
the rounded schedule's cost is within the optimality gap of it (less than 1, since costs are
whole numbers), the schedule is proven optimal without any MILP search.

LP engines: HiGHS on the CPU (dual simplex or PDLP) and cuOpt's PDLP on the GPU.

`warm_started_mip` goes one step further on the GPU: it hands the rounded schedule to
cuOpt's MILP solver as a starting point, which then only has to prove it (or improve it).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from cuopt_scheduler.data import Problem
from cuopt_scheduler.model import FloatArray, ScheduleModel, build_model
from cuopt_scheduler.solver import (
    Backend,
    RawSolution,
    SolveResult,
    _gap_closed,
    _quiet_stdout,
    _solve_cuopt,
    cuopt_data_model,
    highs_model,
    make_result,
)

CUOPT_PDLP_PARAMS: dict[str, int | float] = {
    "method": 1,  # PDLP
    # Stable2 was about twice as fast as 26.10's default (Stable3) on this model, with the
    # same rounded schedules (docs/benchmarks/README.md).
    "pdlp_solver_mode": 1,
}


class LPEngine(StrEnum):
    HIGHS = "highs"
    """CPU: HiGHS's default LP method (dual simplex)."""
    HIGHS_PDLP = "highs-pdlp"
    """CPU: HiGHS's first-order PDLP."""
    CUOPT_PDLP = "cuopt-pdlp"
    """GPU: cuOpt's first-order PDLP."""

    @property
    def backend(self) -> Backend:
        return Backend.GPU if self is LPEngine.CUOPT_PDLP else Backend.CPU


@dataclass(frozen=True)
class LPSolution:
    primal: FloatArray
    duals: FloatArray
    """Row multipliers, for `ScheduleModel.lower_bound`."""


def solve_lp(model: ScheduleModel, engine: LPEngine, time_limit: float) -> LPSolution:
    """The LP relaxation's (approximate, for PDLP) optimal solution and duals."""
    if engine.backend is Backend.CPU:
        h = highs_model(model, time_limit, relax=True)
        if engine is LPEngine.HIGHS_PDLP:
            h.setOptionValue("solver", "pdlp")
        h.run()
        sol = h.getSolution()
        return LPSolution(np.asarray(sol.col_value), np.asarray(sol.row_dual, dtype=np.float64))

    from cuopt.linear_programming import solver, solver_settings  # pyright: ignore[reportMissingImports]

    dm = cuopt_data_model(model, relax=True)
    with _quiet_stdout():
        settings = solver_settings.SolverSettings()
        settings.set_parameter("log_to_console", False)
        settings.set_parameter("time_limit", float(time_limit))
        for name, value in CUOPT_PDLP_PARAMS.items():
            settings.set_parameter(name, value)
        solution = solver.Solve(dm, settings)
    return LPSolution(
        np.asarray(solution.get_primal_solution(), dtype=np.float64),
        np.asarray(solution.get_dual_solution(), dtype=np.float64),
    )


def lp_bound_proves(
    model: ScheduleModel, values: FloatArray, lp: LPSolution, gap_rel: float | None
) -> tuple[bool, float]:
    """(proven, bound): whether the LP duals' bound proves `values` optimal (within gap_rel)."""
    bound = model.lower_bound(lp.duals)
    proven = math.isfinite(bound) and _gap_closed(float(model.cost @ values), bound, gap_rel)
    return proven, bound


def round_lp(model: ScheduleModel, lp: FloatArray) -> FloatArray:
    """Turn an LP solution into a valid 0/1 schedule, preferring high LP values.

    Requires the model's structure: rows with a finite lower bound are demand rows, each with
    its own slack column; all other rows are packing rows with non-negative coefficients.
    """
    n_x = len(model.x_keys)
    rows_of = np.repeat(np.arange(model.num_rows), np.diff(model.offsets))
    on_x = model.indices < n_x
    cover = np.isfinite(model.row_lower)

    # Column-wise view of the x columns: the packing rows each one touches.
    pack = on_x & ~cover[rows_of]
    order = np.argsort(model.indices[pack], kind="stable")
    pack_rows = rows_of[pack][order].tolist()
    pack_vals = model.values[pack][order].tolist()
    pack_ptr = np.searchsorted(model.indices[pack][order], np.arange(n_x + 1)).tolist()

    # Row-wise view of demand rows: their x columns, best LP value first (then cheapest).
    room = model.row_upper.tolist()  # remaining capacity in packing rows
    x = np.zeros(model.num_cols)
    lp_x = np.clip(lp[:n_x], 0.0, 1.0)
    for r in np.flatnonzero(cover):
        start, stop = model.offsets[r], model.offsets[r + 1]
        cols = model.indices[start:stop]
        cols = cols[cols < n_x]
        cols = cols[model.col_upper[cols] > 0]
        cols = cols[np.lexsort((model.cost[cols], -lp_x[cols]))].tolist()
        need = model.row_lower[r]
        for j in cols:
            if need <= 0:
                break
            span = range(pack_ptr[j], pack_ptr[j + 1])
            if all(room[pack_rows[k]] >= pack_vals[k] for k in span):
                for k in span:
                    room[pack_rows[k]] -= pack_vals[k]
                x[j] = 1.0
                need -= 1
    # Slack: whatever demand is still unmet.
    activity = model.row_activity(x)
    is_slack = ~on_x
    slack_rows, slack_cols = rows_of[is_slack], model.indices[is_slack]
    x[slack_cols] = np.maximum(0.0, model.row_lower[slack_rows] - activity[slack_rows])
    return x


def solve_lp_round(
    problem: Problem,
    engine: LPEngine = LPEngine.HIGHS,
    time_limit: float = 30,
    gap_rel: float | None = None,
) -> SolveResult:
    """Schedule `problem` by LP relaxation plus rounding. Optimal if the LP bound proves it."""
    start = time.perf_counter()
    model = build_model(problem)
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    lp = solve_lp(model, engine, time_limit)
    values = round_lp(model, lp.primal)
    first = time.perf_counter() - start
    proven, bound = lp_bound_proves(model, values, lp, gap_rel)
    elapsed = time.perf_counter() - start
    objective = float(model.cost @ values)
    status = f"LP ({engine}) + rounding, LP bound {bound:g}"
    raw = RawSolution(values, proven, status, ((first, objective),))
    return _checked(make_result(model, raw, engine.backend, elapsed, build_seconds))


def warm_started_mip(model: ScheduleModel, time_limit: float, gap_rel: float | None) -> RawSolution:
    """cuOpt PDLP + rounding, then, unless the LP bound already proves that schedule
    optimal, cuOpt's MILP solver starting from it.

    Has a good schedule as soon as the LP is solved, and can still prove it optimal. The
    time limit covers both steps. Used by `solve(..., Backend.GPU)`.
    """
    start = time.perf_counter()
    lp = solve_lp(model, LPEngine.CUOPT_PDLP, time_limit)
    rounded = round_lp(model, lp.primal)
    first = ((time.perf_counter() - start, float(model.cost @ rounded)),)
    proven, bound = lp_bound_proves(model, rounded, lp, gap_rel)
    if proven:
        return RawSolution(rounded, True, f"LP + rounding, proven by LP bound {bound:g}", first)
    t_lp = time.perf_counter() - start
    try:
        raw = _solve_cuopt(model, max(1.0, time_limit - t_lp), gap_rel, initial=rounded)
    except Exception as exc:
        # E.g. out of GPU memory: the MILP solver needs far more than PDLP. The rounded
        # schedule is still valid, so keep it rather than start over on the CPU.
        return RawSolution(rounded, False, f"cuOpt MILP failed ({exc}); rounded LP", first)
    trace = (*first, *((t_lp + t, o) for t, o in raw.incumbents))
    if raw.values is None:  # cuOpt returned nothing: keep the rounded schedule
        return RawSolution(rounded, False, raw.status, trace)
    return RawSolution(raw.values, raw.proven, raw.status, trace)


def _checked(result: SolveResult) -> SolveResult:
    if not result.feasible:
        raise AssertionError("rounding produced an invalid schedule")  # a bug, not bad luck
    return result
