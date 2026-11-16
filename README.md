# ⚡ NVIDIA cuOpt Scheduler

A high-performance, GPU-accelerated staff scheduling application powered by **NVIDIA cuOpt** and **Streamlit**. 

This project solves a nurse rostering problem as a mixed-integer program, on a local NVIDIA GPU with cuOpt or on the CPU with CBC, and includes a benchmark to compare the two honestly (see [GPU vs CPU](#-gpu-vs-cpu)).

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![NVIDIA](https://img.shields.io/badge/NVIDIA-cuOpt-green)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

## 🚀 Features

* **GPU or CPU:** Solves with NVIDIA cuOpt via PuLP, with automatic fallback to CBC on CPU, and can solve the same problem on both side by side.
* **Realistic Simulation:** Generates "fake" but realistic employee profiles (names, skills, availability) using `Faker`.
* **Interactive Visualization:**

    * **Gantt Charts:** Visual timeline of staff assignments.

    * **Heatmaps:** Instant analysis of shift coverage and gaps.

* **Constraint Handling:** Skill requirements, weekly hour caps, availability, one shift per day and a minimum rest between shifts (default 11h).
* **Never just "Infeasible":** When there aren't enough staff, the app still produces the best schedule and lists exactly which shifts are short.
* **Export:** Download optimized schedules as CSV files.

## 🛠️ Prerequisites

* **Python:** 3.12+ and [uv](https://docs.astral.sh/uv/).
* **For the GPU solver (optional):** Linux or WSL2, an NVIDIA GPU (RTX 30/40-series recommended) and drivers supporting CUDA 13. Without one the app runs on the CPU solver (CBC).

## 📦 Installation

```bash
git clone <this repo> && cd cuopt-scheduler
uv sync --extra gpu   # with NVIDIA cuOpt
uv sync               # CPU only
```

The NVIDIA package index is configured in `pyproject.toml`, and `uv.lock` pins every version.

## 🖥️ Usage

```bash
uv run streamlit run app.py
```

* Open `http://localhost:8501`.
* In the **Sidebar**, set the number of wards, employees, schedule length and a random seed (same seed, same data). About 45 employees per ward cover all demand.
* Adjust the minimum rest, time limit and an optional optimality gap.
* Toggle **"Use NVIDIA GPU"**, or tick **compare** to solve on both. Both are disabled when cuOpt is not available.
* Click **Generate & Solve**.

## 📈 GPU vs CPU

```bash
uv run cuopt-bench --wards 1 10 20 --days 28 --time-limit 60
uv run cuopt-bench --help
```

Each run is checked by the independent validator. Typical results (cuOpt's numbers vary between runs because its heuristics run in parallel) on an RTX 4080 Laptop GPU and an i9-13980HX, with cuOpt 25.10.1, CBC 2.10.3, 28 days and 34 employees per ward (deliberately tight):

| Wards | Employees | CBC (CPU) | cuOpt (GPU) |
|---|---|---|---|
| 1 | 34 | 0.10 s, optimal | 0.24 s, optimal |
| 10 | 340 | 0.80 s, optimal | 60 s time limit, within 0.1% |
| 20 | 680 | 1.5–1.9 s, optimal | 60 s time limit; 0–390 slots uncovered, varying between runs |
| 40 | 1360 | 3.4–4.8 s, optimal | 60 s time limit; 539–855 slots uncovered, or no solution |

**For this model, CBC wins at every size tested.** cuOpt finds a near-optimal schedule quickly (within 0.1% after about 1.5 s at 10 wards). But cuOpt 25.10 runs its branch-and-bound proof search on CPU threads, and in these runs it didn't stop at the requested `mip_relative_gap`, so it runs to the time limit. Meanwhile CBC solves this kind of assignment problem almost directly from its initial LP relaxation. Re-run the benchmark after upgrading cuOpt.

## 🧪 Development

```bash
uv run pytest                  # tests; GPU tests skip automatically without cuOpt
uv run ruff format . && uv run ruff check .
uv run pyright
```

CI (`.github/workflows/ci.yml`) runs the same checks on every pull request, on CPU only.

The tests check every solution against `cuopt_scheduler.validate`, an independent rule checker that doesn't reuse the model. It works from clock times, so the rest rule is checked independently of how the model encodes it. They include property-based tests (Hypothesis) over generated problems and headless Streamlit tests (`AppTest`).

## 📂 Project Structure

```text
cuopt-scheduler/
├── app.py                    # Streamlit UI only
├── src/cuopt_scheduler/
│   ├── data.py               # Problem definition and synthetic data generation
│   ├── model.py              # MILP formulation (PuLP)
│   ├── solver.py             # CPU/GPU solving with fallback
│   ├── validate.py           # Independent schedule checker
│   ├── frames.py             # DataFrames for display and CSV export
│   ├── viz.py                # Plotly figures
│   └── benchmark.py          # CPU vs GPU benchmark (`cuopt-bench`)
├── tests/
├── pyproject.toml / uv.lock
└── README.md
```

## 🧠 How It Works

1. **Data Generation**: Creates a synthetic workforce with random availability and skills (RN, LPN, CNA).
2. **Modeling**: Formulates the problem as a Mixed-Integer Linear Program (MILP) with PuLP.
   * *Objective*: First cover as much demand as possible (each uncovered slot costs 1000), then use as few shifts as possible.
   * *Constraints*: Staffing demand per skill (shortfall allowed but penalised), max hours per calendar week, availability, at most one shift per day, and minimum rest between consecutive days' shifts.
3. **Solving**: On GPU, PuLP hands the model to the cuOpt library. On CPU, or if cuOpt is unavailable, it uses CBC. A result counts as a solution only if every variable has a value and every constraint holds, regardless of the solver's status code. It is "optimal" only if the solver also proved it.
4. **Visualization**: Results become pandas DataFrames rendered with Plotly.

## ⚠️ Troubleshooting

* **GPU toggle is disabled**: cuOpt isn't available. Run `uv sync --extra gpu` and check `nvidia-smi`.
* **"required staff slots could not be covered"**: Not enough staff for the demand. Add employees (about 45 per ward), or lower the minimum rest.
* **"Time limit reached ... not proven optimal"**: The schedule is valid but may not be the best. Raise the time limit, set a gap, or try CBC.

## 📄 License

This project is open for educational and testing purposes.