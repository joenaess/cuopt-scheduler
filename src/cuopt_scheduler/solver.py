"""Solve a problem on CPU (CBC) or GPU (cuOpt), falling back to CPU when needed."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

import pulp

from cuopt_scheduler.data import Problem
from cuopt_scheduler.model import build_model


class Backend(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True, order=True)
class Assignment:
    day: int
    shift: str
    employee_id: str


@dataclass(frozen=True)
class SolveResult:
    status: str
    """PuLP status string, e.g. "Optimal" or "Infeasible"."""
    backend: Backend
    """The backend that actually produced the result."""
    solve_seconds: float
    objective: float | None
    assignments: tuple[Assignment, ...]
    warnings: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.status == "Optimal"


def cuopt_available() -> bool:
    """True if PuLP can actually run cuOpt here. Importing pulp.CUOPT is not enough."""
    try:
        return bool(pulp.CUOPT(msg=False).available())
    except Exception:
        return False


def _solver(backend: Backend, time_limit: float) -> pulp.LpSolver:
    if backend is Backend.GPU:
        return pulp.CUOPT(msg=False, timeLimit=time_limit)
    # PuLP 3.x re-exports via `import *` and a Rust core, which pyright cannot follow.
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)  # pyright: ignore[reportAttributeAccessIssue]


def solve(problem: Problem, backend: Backend = Backend.CPU, time_limit: float = 30) -> SolveResult:
    model = build_model(problem)
    warnings: list[str] = []

    if backend is Backend.GPU and not cuopt_available():
        warnings.append("cuOpt is not available; solved on CPU instead.")
        backend = Backend.CPU

    start = time.perf_counter()
    try:
        model.lp.solve(_solver(backend, time_limit))
    except Exception as exc:
        if backend is Backend.CPU:
            raise
        warnings.append(f"cuOpt failed ({exc}); solved on CPU instead.")
        backend = Backend.CPU
        start = time.perf_counter()
        model.lp.solve(_solver(backend, time_limit))
    elapsed = time.perf_counter() - start

    status = pulp.LpStatus[model.lp.status]  # pyright: ignore[reportAttributeAccessIssue]
    assignments: tuple[Assignment, ...] = ()
    if status == "Optimal":
        assignments = tuple(
            sorted(
                Assignment(day=d, shift=s, employee_id=emp)
                for (emp, s, d), var in model.x.items()
                if (var.varValue or 0) > 0.5
            )
        )
    return SolveResult(
        status=status,
        backend=backend,
        solve_seconds=elapsed,
        objective=model.lp.objective.value() if assignments and model.lp.objective else None,
        assignments=assignments,
        warnings=tuple(warnings),
    )
