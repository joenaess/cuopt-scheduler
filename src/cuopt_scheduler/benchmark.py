"""Compare exact MILP solvers and LP-plus-rounding, on CPU and GPU, on the same problems.

    uv run cuopt-bench --wards 5 20 80 --levels 0 3 --csv results.csv

Methods: the two exact MILP solvers (HiGHS on CPU, cuOpt on GPU); LP relaxation plus
rounding (`lp_round`) with HiGHS (simplex or PDLP) on the CPU or cuOpt PDLP on the GPU; and
cuOpt's MILP solver started from cuOpt's rounded LP schedule. GPU methods are skipped when
cuOpt can't run.

Each problem has `--staff-per-ward` employees per ward. The default (34) is deliberately
tight, which makes the problem hard enough to be interesting; around 45 per ward covers all
demand and the problem becomes easy. `--levels` adds rules (see `data.LEVELS`).

Besides total time (which for the MILP solvers includes proving optimality), each row
reports when the method first had any schedule, and when it first had one within `--near`
(default 1%) of the best schedule any method found for that problem. That is the time a
user waits for a good roster.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from cuopt_scheduler.data import LEVELS, Problem, generate_problem
from cuopt_scheduler.lp_round import LPEngine, solve_lp_round
from cuopt_scheduler.solver import Backend, SolveResult, cuopt_available, solve
from cuopt_scheduler.validate import coverage_gaps, violations


class Method(StrEnum):
    HIGHS_MIP = "highs-mip"
    CUOPT_MIP = "cuopt-mip"
    CUOPT_MIP_WARM = "cuopt-pdlp+round+mip"
    HIGHS_SIMPLEX_ROUND = "highs-simplex+round"
    HIGHS_PDLP_ROUND = "highs-pdlp+round"
    CUOPT_PDLP_ROUND = "cuopt-pdlp+round"

    @property
    def lp_engine(self) -> LPEngine | None:
        return {
            Method.HIGHS_SIMPLEX_ROUND: LPEngine.HIGHS,
            Method.HIGHS_PDLP_ROUND: LPEngine.HIGHS_PDLP,
            Method.CUOPT_PDLP_ROUND: LPEngine.CUOPT_PDLP,
        }.get(self)

    @property
    def device(self) -> Backend:
        if self is Method.HIGHS_MIP:
            return Backend.CPU
        if self in (Method.CUOPT_MIP, Method.CUOPT_MIP_WARM):
            return Backend.GPU
        assert self.lp_engine is not None
        return self.lp_engine.backend

    def available(self) -> bool:
        return self.device is Backend.CPU or cuopt_available()

    def run(self, problem: Problem, time_limit: float, gap_rel: float | None) -> SolveResult:
        """Solve with this method. A GPU method that had to fall back to the CPU returns
        that CPU result; `iter_benchmark` records it as failed."""
        if engine := self.lp_engine:
            return solve_lp_round(problem, engine, time_limit)
        return solve(
            problem,
            self.device,
            time_limit=time_limit,
            gap_rel=gap_rel,
            warm_start=self is Method.CUOPT_MIP_WARM,
        )


@dataclass(frozen=True)
class BenchmarkRow:
    wards: int
    employees: int
    level: int
    days: int
    seed: int
    method: str
    device: str
    outcome: str
    """optimal / feasible / no_solution, or "failed: ..." if a GPU method had to fall back."""
    seconds: float | None
    """Time in the solver. Building the model is timed separately."""
    build_seconds: float
    objective: float | None
    uncovered: int | None
    """Required staff slots left empty (None if no solution was found)."""
    first_seconds: float | None
    """When the solver first had any schedule."""
    near_seconds: float | None
    """When it first had a schedule within `near` of the best any method found."""


def seconds_to_reach(result: SolveResult, target: float) -> float | None:
    """When `result`'s solver first had a schedule with objective <= target, or None."""
    for seconds, objective in result.incumbents:
        if objective <= target + 1e-6:
            return seconds
    # Some solutions arrive without an incumbent callback (e.g. found in presolve).
    if result.objective is not None and result.objective <= target + 1e-6:
        return result.solve_seconds
    return None


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


def iter_benchmark(
    wards: Iterable[int],
    staff_per_ward: int,
    days: Iterable[int],
    seeds: Iterable[int],
    methods: Sequence[Method],
    time_limit: float,
    gap_rel: float | None = None,
    near: float = 0.01,
    levels: Iterable[int] = (0,),
) -> Iterator[list[BenchmarkRow]]:
    """Yield the rows for one problem at a time, so long runs can be saved as they go."""
    for method in methods:
        if method.device is Backend.GPU:
            # The first call pays for CUDA initialisation; keep that out of the timings.
            warm_up = method.run(generate_problem(10, 7, seed=0), time_limit=10, gap_rel=None)
            if warm_up.backend is not Backend.GPU:
                raise RuntimeError(f"{method} can't run on the GPU: {warm_up.warnings}")

    for w, level, d, seed in itertools.product(wards, levels, days, seeds):
        problem = generate_problem(w * staff_per_ward, d, seed=seed, wards=w, level=level)
        results = []
        for method in methods:
            result = method.run(problem, time_limit, gap_rel)
            if broken := violations(problem, result.assignments):
                raise AssertionError(f"{method} returned an invalid schedule: {broken[:3]}")
            results.append((method, result))
        found = [
            r.objective for m, r in results if r.objective is not None and r.backend is m.device
        ]
        target = min(found) * (1 + near) if found else -float("inf")
        yield [
            _row(problem, w, level, d, seed, method, result, target) for method, result in results
        ]


def _row(
    problem: Problem,
    wards: int,
    level: int,
    days: int,
    seed: int,
    method: Method,
    result: SolveResult,
    target: float,
) -> BenchmarkRow:
    common = {
        "wards": wards,
        "employees": len(problem.employees),
        "level": level,
        "days": days,
        "seed": seed,
        "method": str(method),
        "device": str(method.device),
        "build_seconds": round(result.build_seconds, 3),
    }
    if result.backend is not method.device:
        # The GPU solver failed (e.g. out of GPU memory); the CPU fallback's result isn't its own.
        return BenchmarkRow(
            **common,
            outcome="failed: " + "; ".join(result.warnings),
            seconds=None,
            objective=None,
            uncovered=None,
            first_seconds=None,
            near_seconds=None,
        )
    return BenchmarkRow(
        **common,
        outcome=str(result.outcome),
        seconds=round(result.solve_seconds, 3),
        objective=result.objective,
        uncovered=sum(coverage_gaps(problem, result.assignments).values())
        if result.feasible
        else None,
        first_seconds=_round(seconds_to_reach(result, float("inf"))),
        near_seconds=_round(seconds_to_reach(result, target)),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--wards", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[0], choices=sorted(LEVELS), help="difficulty"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        type=Method,
        choices=list(Method),
        help="default: every method whose solver is installed",
    )
    parser.add_argument("--staff-per-ward", type=int, default=34)
    parser.add_argument("--days", type=int, nargs="+", default=[28])
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds per size")
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--gap", type=float, default=None, help="relative gap, e.g. 0.01")
    parser.add_argument(
        "--near", type=float, default=0.01, help="'good enough' fraction from the best found"
    )
    parser.add_argument("--csv", type=Path, help="also write the results to this CSV file")
    args = parser.parse_args(argv)

    methods = args.methods or [m for m in Method if m.available()]
    if skipped := [m for m in methods if not m.available()]:
        raise SystemExit(f"not available here: {', '.join(skipped)}")
    if not any(m.device is Backend.GPU for m in methods):
        print("No GPU method available: running CPU only.")

    done: list[BenchmarkRow] = []
    for rows in iter_benchmark(
        args.wards,
        args.staff_per_ward,
        args.days,
        range(args.seeds),
        methods,
        args.time_limit,
        args.gap,
        args.near,
        args.levels,
    ):
        done += rows
        df = pd.DataFrame([asdict(r) for r in done])
        print(pd.DataFrame([asdict(r) for r in rows]).to_string(index=False), flush=True)
        if args.csv:
            df.to_csv(args.csv, index=False)  # rewritten after every problem


if __name__ == "__main__":
    main()
