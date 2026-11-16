"""Compare CPU (CBC) and GPU (cuOpt) on the same generated problems.

    uv run cuopt-bench --wards 1 5 10 20 --days 28 --seeds 3

Each problem has `--staff-per-ward` employees per ward. The default (34) is deliberately
tight, which makes the problem hard enough to be interesting; around 45 per ward covers all
demand and the problem becomes easy.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import pandas as pd

from cuopt_scheduler.data import generate_problem
from cuopt_scheduler.solver import Backend, cuopt_available, solve
from cuopt_scheduler.validate import coverage_gaps, violations


@dataclass(frozen=True)
class BenchmarkRow:
    wards: int
    employees: int
    days: int
    seed: int
    backend: str
    outcome: str
    seconds: float
    objective: float | None
    uncovered: int | None
    """Required staff slots left empty (None if no solution was found)."""


def run_benchmark(
    wards: Iterable[int],
    staff_per_ward: int,
    days: Iterable[int],
    seeds: Iterable[int],
    backends: Sequence[Backend],
    time_limit: float,
    gap_rel: float | None = None,
) -> list[BenchmarkRow]:
    if Backend.GPU in backends:
        # The first cuOpt call pays for CUDA initialisation; keep that out of the timings.
        solve(generate_problem(10, 7, seed=0), Backend.GPU, time_limit=10)

    rows = []
    for w, d, seed in itertools.product(wards, days, seeds):
        problem = generate_problem(w * staff_per_ward, d, seed=seed, wards=w)
        for backend in backends:
            result = solve(problem, backend, time_limit=time_limit, gap_rel=gap_rel)
            if broken := violations(problem, result.assignments):
                raise AssertionError(f"{backend} returned an invalid schedule: {broken[:3]}")
            rows.append(
                BenchmarkRow(
                    wards=w,
                    employees=len(problem.employees),
                    days=d,
                    seed=seed,
                    backend=str(result.backend),
                    outcome=str(result.outcome),
                    seconds=round(result.solve_seconds, 3),
                    objective=result.objective,
                    uncovered=sum(coverage_gaps(problem, result.assignments).values())
                    if result.feasible
                    else None,
                )
            )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--wards", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--staff-per-ward", type=int, default=34)
    parser.add_argument("--days", type=int, nargs="+", default=[28])
    parser.add_argument("--seeds", type=int, default=1, help="number of seeds per size")
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--gap", type=float, default=None, help="relative gap, e.g. 0.01")
    parser.add_argument("--csv", help="also write the results to this CSV file")
    args = parser.parse_args(argv)

    backends = [Backend.CPU, Backend.GPU] if cuopt_available() else [Backend.CPU]
    if len(backends) == 1:
        print("cuOpt not available: running CPU only.")
    rows = run_benchmark(
        args.wards,
        args.staff_per_ward,
        args.days,
        range(args.seeds),
        backends,
        args.time_limit,
        args.gap,
    )
    df = pd.DataFrame([asdict(r) for r in rows])
    print(df.to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
