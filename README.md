# ⚡ NVIDIA cuOpt Scheduler

A high-performance, GPU-accelerated staff scheduling application powered by **NVIDIA cuOpt** and **Streamlit**. 

This project demonstrates how to solve complex combinatorial optimization problems (like nurse rostering) in seconds using the massive parallel processing power of local NVIDIA GPUs (Ampere/Ada Lovelace architectures).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NVIDIA](https://img.shields.io/badge/NVIDIA-cuOpt-green)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)

## 🚀 Features

* **GPU Acceleration:** Utilizes NVIDIA cuOpt (via PuLP) to solve scheduling problems significantly faster than traditional CPU solvers (CBC/GLPK).
* **Realistic Simulation:** Generates "fake" but realistic employee profiles (names, skills, availability) using `Faker`.
* **Interactive Visualization:**

    * **Gantt Charts:** Visual timeline of staff assignments.

    * **Heatmaps:** Instant analysis of shift coverage and gaps.

* **Constraint Handling:** Respects maximum hours, skill requirements, availability, and "one shift per day" rules.
* **Export:** Download optimized schedules as CSV files.

## 🛠️ Prerequisites

* **Hardware:** NVIDIA GPU (RTX 30-series or 40-series recommended).
* **OS:** Linux (Tested on Pop!_OS) or WSL2.
* **Drivers:** NVIDIA Drivers (535+) and CUDA Toolkit 12/13.
* **Python:** Version 3.10 or higher.

## 📦 Installation

1. **Clone or Create Directory**

    ```bash
    mkdir ~/cuopt-scheduler
    cd ~/cuopt-scheduler
    ```

2. **Set up Virtual Environment**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install Dependencies**
    Note: We must point to the NVIDIA PyPI index to get the optimized `cuopt` binaries.
  
    *Create a `requirements.txt` with the following content:*

    ```text
    streamlit
    pandas
    pulp>=2.8.0
    Faker
    plotly
    --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com)
    cuopt-server-cu13
    cuopt-sh-client
    ```

    *Then run:*

    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

1. **Activate Environment** (if not already active)

    ```bash
    source .venv/bin/activate
    ```

2. **Run the Application**

    ```bash
    streamlit run app.py
    ```

3. **Interact**
    * Open your browser to `http://localhost:8501`.
    * Use the **Sidebar** to adjust the number of employees and simulation duration.
    * Toggle **"Use NVIDIA GPU"** to compare performance against the CPU solver.
    * Click **Generate & Solve**.

## 📂 Project Structure

```text
cuopt-scheduler/
├── app.py              # Main application logic (UI, Solver, Viz)
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── data/               # Directory for generated CSVs (optional)
```

## 🧠 How It Works

1.**Data Generation**: The app creates a synthetic workforce with randomized availability and skills (RN, LPN, CNA).

2.**Modeling**: The problem is formulated as a Mixed-Integer Linear Program (MILP) using PuLP.

* *Objective*: Minimize total assigned shifts (or maximize efficiency).

* *Constraints*: Staffing demand, max hours, availability, no double-booking.

3.**Solving**: * If GPU is selected, PuLP delegates the matrix calculations to cuopt-server, utilizing CUDA cores for rapid convergence.

* *If CPU is selected*, it falls back to the default CBC solver.

4.**Visualization**: Results are parsed into Pandas DataFrames and rendered using Plotly.

## ⚠️ Troubleshooting

"pulp.CUOPT not detected": Ensure you have installed the specific cuopt packages from the NVIDIA index URL. Check that your pulp version is >= 2.8.0.

Optimization Infeasible: If the solver returns "Infeasible," try increasing the Number of Employees in the sidebar. The constraints might be too tight (e.g., needing 5 RNs but only generating 3).

## 📄 License

This project is open for educational and testing purposes.