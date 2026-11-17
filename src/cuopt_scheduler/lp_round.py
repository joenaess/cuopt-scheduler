"""A heuristic that can run its heavy part on a GPU LP solver: solve the LP relaxation, then
round it into a valid schedule.

For this model the LP bound equals the integer optimum (see the README), so a good LP
solution is close to a good schedule. Rounding never breaks a rule because every rule is a
packing row (a sum of assignments <= a limit): assignments are only ever added while every
packing row they touch still has room. Demand rows have a slack column that absorbs any
shortfall.

LP engines: HiGHS on the CPU (dual simplex, interior point, or PDLP), and on the GPU cuOpt's
PDLP or cuPDLPx (optional `pdlp` extra).
"""

from __future__ import annotations

import time
from enum import StrEnum

import numpy as np

from cuopt_scheduler.data import Problem
from cuopt_scheduler.model import FloatArray, ScheduleModel, build_model
from cuopt_scheduler.solver import (
    Assignment,
    Backend,
    Outcome,
    SolveResult,
    _quiet_stdout,
    classify,
    cuopt_data_model,
    highs_model,
)


class LPEngine(StrEnum):
    HIGHS = "highs"
    """CPU: HiGHS's default LP method (dual simplex)."""
    HIGHS_IPM = "highs-ipm"
    """CPU: HiGHS's interior-point method, without crossover."""
    HIGHS_PDLP = "highs-pdlp"
    """CPU: HiGHS's first-order PDLP."""
    CUOPT_PDLP = "cuopt-pdlp"
    """GPU: cuOpt's first-order PDLP."""
    CUPDLPX = "cupdlpx"
    """GPU: cuPDLPx (MIT), needs the `pdlp` extra."""

    @property
    def backend(self) -> Backend:
        return Backend.GPU if self in (LPEngine.CUOPT_PDLP, LPEngine.CUPDLPX) else Backend.CPU


def cupdlpx_available() -> bool:
    try:
        import cupdlpx  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


def solve_lp(model: ScheduleModel, engine: LPEngine, time_limit: float) -> FloatArray:
    """The LP relaxation's (approximate, for PDLP) optimal solution."""
    if engine.backend is Backend.CPU:
        h = highs_model(model, time_limit, relax=True)
        if engine is LPEngine.HIGHS_IPM:
            h.setOptionValue("solver", "ipm")
            h.setOptionValue("run_crossover", "off")
        elif engine is LPEngine.HIGHS_PDLP:
            h.setOptionValue("solver", "pdlp")
        h.run()
        return np.asarray(h.getSolution().col_value)

    if engine is LPEngine.CUPDLPX:
        import cupdlpx  # pyright: ignore[reportMissingImports]
        import scipy.sparse as sp  # pyright: ignore[reportMissingImports]

        a = sp.csr_matrix(
            (model.values, model.indices, model.offsets), shape=(model.num_rows, model.num_cols)
        )
        m = cupdlpx.Model(
            objective_vector=model.cost,
            constraint_matrix=a,
            constraint_lower_bound=model.row_lower,
            constraint_upper_bound=model.row_upper,
            variable_lower_bound=model.col_lower,
            variable_upper_bound=model.col_upper,
        )
        m.setParams(OutputFlag=False, TimeLimit=float(time_limit))
        with _quiet_stdout():
            m.optimize()
        return np.asarray(m.X, dtype=np.float64)

    from cuopt.linear_programming import solver, solver_settings  # pyright: ignore[reportMissingImports]

    dm = cuopt_data_model(model, relax=True)
    with _quiet_stdout():
        settings = solver_settings.SolverSettings()
        settings.set_parameter("log_to_console", False)
        settings.set_parameter("time_limit", float(time_limit))
        settings.set_parameter("method", 1)  # PDLP
        solution = solver.Solve(dm, settings)
    return np.asarray(solution.get_primal_solution(), dtype=np.float64)


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
    problem: Problem, engine: LPEngine = LPEngine.HIGHS, time_limit: float = 30
) -> SolveResult:
    """Schedule `problem` by LP relaxation plus rounding. Never claims optimality."""
    start = time.perf_counter()
    model = build_model(problem)
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    lp = solve_lp(model, engine, time_limit)
    values = round_lp(model, lp)
    elapsed = time.perf_counter() - start

    outcome = classify(model, values, proven=False)
    if outcome is Outcome.NO_SOLUTION:
        raise AssertionError("rounding produced an invalid schedule")  # a bug, not bad luck
    chosen = np.flatnonzero(values[: len(model.x_keys)] > 0.5)
    objective = float(model.cost @ values)
    return SolveResult(
        outcome=outcome,
        solver_status=f"LP ({engine}) + rounding",
        backend=engine.backend,
        solve_seconds=elapsed,
        objective=objective,
        assignments=tuple(
            sorted(
                Assignment(day=d, shift=s, employee_id=e)
                for e, s, d in (model.x_keys[i] for i in chosen)
            )
        ),
        build_seconds=build_seconds,
        incumbents=((elapsed, objective),),
    )
