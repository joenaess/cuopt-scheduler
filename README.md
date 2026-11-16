# ⚡ NVIDIA cuOpt Scheduler

A high-performance, GPU-accelerated staff scheduling application powered by **NVIDIA cuOpt** and **Streamlit**. 

This project demonstrates how to solve complex combinatorial optimization problems (like nurse rostering) in seconds using the massive parallel processing power of local NVIDIA GPUs (Ampere/Ada Lovelace architectures).

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![NVIDIA](https://img.shields.io/badge/NVIDIA-cuOpt-green)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

## 🚀 Features

* **GPU Acceleration:** Solves with NVIDIA cuOpt via PuLP, with automatic fallback to CBC on CPU. At the demo's default sizes CBC is still faster; cuOpt pays off on larger problems.
* **Realistic Simulation:** Generates "fake" but realistic employee profiles (names, skills, availability) using `Faker`.
* **Interactive Visualization:**

    * **Gantt Charts:** Visual timeline of staff assignments.

    * **Heatmaps:** Instant analysis of shift coverage and gaps.

* **Constraint Handling:** Respects maximum hours, skill requirements, availability, and "one shift per day" rules.
* **Export:** Download optimized schedules as CSV files.

## 🛠️ Prerequisites

* **Python:** 3.11+ and [uv](https://docs.astral.sh/uv/).
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
* In the **Sidebar**, set the number of employees, the schedule length and a random seed (same seed, same data).
* Toggle **"Use NVIDIA GPU"** to compare against the CPU solver. It is disabled when cuOpt is not available.
* Click **Generate & Solve**.

## 🧪 Development

```bash
uv run pytest                  # tests; GPU tests skip automatically without cuOpt
uv run ruff format . && uv run ruff check .
uv run pyright
```

CI (`.github/workflows/ci.yml`) runs the same checks on every pull request, on CPU only.

The tests check every solution against `cuopt_scheduler.validate.violations`, an independent rule checker that doesn't reuse the model. They include property-based tests (Hypothesis) over generated problems and headless Streamlit tests (`AppTest`).

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
│   └── viz.py                # Plotly figures
├── tests/
├── pyproject.toml / uv.lock
└── README.md
```

## 🧠 How It Works

1. **Data Generation**: Creates a synthetic workforce with random availability and skills (RN, LPN, CNA).
2. **Modeling**: Formulates the problem as a Mixed-Integer Linear Program (MILP) with PuLP.
   * *Objective*: Minimize total assigned shifts.
   * *Constraints*: Staffing demand per skill, max hours, availability, at most one shift per day.
3. **Solving**: On GPU, PuLP hands the model to the cuOpt library. On CPU, or if cuOpt is unavailable, it uses CBC.
4. **Visualization**: Results become pandas DataFrames rendered with Plotly.

## ⚠️ Troubleshooting

* **GPU toggle is disabled**: cuOpt isn't available. Run `uv sync --extra gpu` and check `nvidia-smi`.
* **"Infeasible"**: The constraints are too tight for the generated staff. With 7 days you need roughly 40+ employees, and more for longer schedules, because the hour cap currently applies to the whole period rather than per week.

## 📄 License

This project is open for educational and testing purposes.