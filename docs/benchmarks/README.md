# Benchmark data and cuOpt tuning notes

`runs/w<wards>-l<level>.csv` are raw `cuopt-bench` results, one file per problem (one process
each, to keep memory in check). Regenerate a file by deleting it and running, for example:

```bash
uv run cuopt-bench --wards 80 --levels 3 --days 28 --time-limit 120 --csv docs/benchmarks/runs/w80-l3.csv
uv run cuopt-plots docs/benchmarks/runs/*.csv --out docs/img
```

Machine: RTX 4080 Laptop GPU (12 GB), i9-13980HX (24 cores), 32 GB RAM, cuOpt 26.10.1
(CUDA 13), HiGHS 1.16. All problems: 28 days, 34 employees per ward, seed 0.

## How cuOpt's settings were chosen

Measured on this machine with `generate_problem(34 * wards, 28, seed=0, wards=wards, level=...)`.

### LP relaxation (PDLP), for LP + rounding

Median of 3 solves, LP solve plus rounding:

| `pdlp_solver_mode` | 2,720 employees, level 0 | 5,440 employees, level 0 |
|---|---|---|
| Stable3 (26.10 default) | 0.79 s | 1.21 s |
| **Stable2 (used)** | **0.39 s** | **0.76 s** |
| Fast1 | 0.38 s | 0.74 s |
| Methodical1 | 0.40 s | 0.76 s |
| Stable3, tolerances 1e-2 | 0.78 s | 1.13 s |

Every setting gave the same (optimal) rounded schedule. Stable2 is used rather than Fast1
because it was as fast here and is documented as the more robust mode. Looser tolerances
did not help: most of the time is not in the final digits. Barrier took 9.9 s at 2,720
employees and gave a worse rounding; dual simplex run right after a barrier solve in the
same process failed with `cudaErrorInvalidDevice` (it works on its own).

### MILP

At 1,360 employees, level 3, 90 s limit (time until within 1% of the optimum / total):

| Setting | Within 1% | Total (proven optimal) |
|---|---|---|
| **presolve off (used)** | **14.9 s** | **16.5 s** |
| presolve on (default) | 42 s | 51 s |
| presolve off, symmetry off | 20 s | 23 s |
| presolve off, probing off | 15 s | 16 s |
| presolve off, cut passes 0 | 51 s | 82 s |
| presolve off, batch-PDLP strong branching | 15 s | 17 s |
| presolve off, no MIP scaling | 15 s | 16 s |

cuOpt's log shows where the remaining time goes: its root LP bound is already the optimum,
but the root LP itself is slow inside the MILP solver (PDLP needed 181,000 iterations and
18 s at 2,720 employees, against 0.4 s standalone), and with the default `method`
(concurrent) the race was sometimes won by a 31 s dual simplex. `method = 1` (PDLP) is
therefore set for MILP solves too.

The bigger gain is a starting solution. `cuopt-pdlp+round+mip` solves the LP with PDLP,
rounds it (which already gives an optimal schedule on these problems), and passes that to
the MILP solver with `DataModel.set_initial_primal_solution`, which works for MILPs in 26.10
(it failed in 25.10). Proven optimal, total time:

| Problem | MILP | MILP from rounding |
|---|---|---|
| 1,360 employees, level 3 | 18 s | 8.3 s |
| 2,720 employees, level 0 | 27–29 s | 21 s |

### Proving the rounded schedule optimal with the LP bound

The MILP solver above spent nearly all its time proving a schedule that rounding had
already found. The LP solver's duals give that proof directly: for any row multipliers y,
`cost @ v >= y @ (row bounds) + sum_j min over the column's bounds of (cost - A.T y)_j v_j`
for every valid v (weak duality, the Lagrangian bound), so PDLP's approximate duals are
safe to use; they only need to be good enough to make the bound tight.
`ScheduleModel.lower_bound` computes it, after zeroing multipliers that point at an infinite
row bound and subtracting a floating-point error allowance. Costs are whole numbers, so a
schedule less than 1 above the bound is optimal (the same absolute gap the MILP solvers get).

Gap between the rounded schedule and the bound from cuOpt PDLP's (Stable2) duals:

| Problem | PDLP + rounding + bound | Gap | Before: MILP from rounding, total |
|---|---|---|---|
| 170 employees, level 3 | 0.08 s | 0.27 | 0.39 s |
| 2,720 employees, level 0 | 0.41 s | 0.53 | 21 s |
| 2,720 employees, level 3 | 0.62 s | 0.74 | 30 s |
| 5,440 employees, level 3 | 1.4 s | 0.01 | 121 s |
| 10,880 employees, level 0 | 1.7 s | 0.81 | 123 s |
| 10,880 employees, level 3 | 2.7 s | 0.00 | 123 s |

All under 1, so every benchmark problem is proven optimal without the MILP solver, which
`cuopt-pdlp+round+mip` now runs only as a fallback. The tables below, measured before this
change, still label that method "MILP from rounding".

## cuOpt 26.12 nightly (checked 2026-11-18)

`cuopt-cu13 26.12.00a101` from the RAPIDS nightly index passes the test suite, but is no
faster here, so the project stays on 26.10.1. Total / time until within 1%, 120 s limit (an
empty cell means it never got within 1%):

| Problem | Method | 26.10.1 | 26.12 nightly |
|---|---|---|---|
| 680 employees, level 3 | cuOpt MILP | 6.6 s / 4.6 s | 5.9 s / 3.8 s |
| | MILP from rounding | 6.1 s / 0.16 s | 6.4 s / 0.19 s |
| 2,720 employees, level 0 | cuOpt MILP | 27 s / 26 s | 33 s / 32 s |
| | MILP from rounding | 21 s / 0.38 s | 20 s / 0.36 s |
| 2,720 employees, level 3 | cuOpt MILP | 45 s / 41 s | 53 s / 52 s |
| | MILP from rounding | 30 s / 0.56 s | 27 s / 0.58 s |
| 5,440 employees, level 3 | cuOpt MILP | 121 s / – | 121 s / – |
| | MILP from rounding | 121 s / 1.1 s | 122 s / 1.1 s |
| 10,880 employees, level 0 | cuOpt MILP | 121 s / 90 s | 123 s / – (best 245,289 vs 134,400) |
| | MILP from rounding | 123 s / 1.6 s | 122 s / 1.5 s |

PDLP + rounding times were within 0.03 s of 26.10.1 on every problem. To try a nightly
without touching `uv.lock`, install it into a separate environment:

```bash
uv venv /tmp/cuopt-nightly && VIRTUAL_ENV=/tmp/cuopt-nightly uv pip install --prerelease=allow \
  --index-strategy unsafe-best-match \
  --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
  --extra-index-url https://pypi.nvidia.com -e . "cuopt-cu13>=26.12.0a0,<26.13"
```
