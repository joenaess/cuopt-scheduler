"""Streamlit UI. All scheduling logic lives in the cuopt_scheduler package."""

import dataclasses

import pandas as pd
import streamlit as st

from cuopt_scheduler import (
    Backend,
    Outcome,
    SolveResult,
    coverage_gaps,
    cuopt_available,
    generate_problem,
    solve,
)
from cuopt_scheduler.data import LEVELS
from cuopt_scheduler.frames import demand_frame, employees_frame, gaps_frame, schedule_frame
from cuopt_scheduler.viz import coverage_figure, gantt_figure

st.set_page_config(page_title="NVIDIA cuOpt Scheduler", layout="wide")
st.title("🏥 Intelligent Staff Scheduler")
st.caption("Powered by NVIDIA cuOpt & Streamlit")

with st.sidebar:
    st.header("1. Data Generation")
    wards = st.slider("Wards", 1, 20, 1, help="Each ward adds 15 staff slots per day.")
    num_emp = st.slider(
        "Number of Employees", 10, 1000, 45, help="About 45 per ward covers all demand."
    )
    duration = st.slider("Schedule Duration (Days)", 7, 28, 7)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.divider()

    st.header("2. Rules")
    min_rest = st.slider(
        "Minimum rest between shifts (hours)",
        0,
        16,
        11,
        help="11h rules out Night→Day and Evening→Day on consecutive days.",
    )
    level = st.select_slider(
        "Difficulty",
        options=sorted(LEVELS),
        value=0,
        help="\n\n".join(f"**{k}**: {v}" for k, v in LEVELS.items()),
    )
    st.caption(LEVELS[level])
    st.caption("Max hours apply per calendar week (24, 32 or 40 per employee).")

    st.divider()

    st.header("3. Optimization Settings")
    gpu_ok = cuopt_available()
    use_gpu = st.toggle(
        "Use NVIDIA GPU (cuOpt)",
        value=gpu_ok,
        disabled=not gpu_ok,
        help=None if gpu_ok else "cuOpt not available. Install with `uv sync --extra gpu`.",
    )
    compare = st.checkbox(
        "Also solve on the other backend and compare",
        disabled=not gpu_ok,
        help="Solves the same problem on CPU and GPU.",
    )
    time_limit = st.slider("Time limit (seconds)", 5, 120, 30)
    gap_pct = st.number_input(
        "Stop within this % of optimal",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help="0 = solver default. A small gap often saves a lot of time.",
    )

    if st.button("Generate & Solve", type="primary"):
        problem = dataclasses.replace(
            generate_problem(num_emp, duration, seed=int(seed), wards=wards, level=level),
            min_rest_hours=min_rest,
        )
        primary = Backend.GPU if use_gpu else Backend.CPU
        backends = [primary]
        if compare and gpu_ok:
            backends.append(Backend.CPU if use_gpu else Backend.GPU)
        gap = gap_pct / 100 or None
        results: list[SolveResult] = []
        for backend in backends:
            with st.spinner(f"Optimizing schedule on {backend.upper()}..."):
                results.append(solve(problem, backend, time_limit=time_limit, gap_rel=gap))
        # Keep results across reruns (e.g. clicking the download button).
        st.session_state["run"] = (problem, results)

run = st.session_state.get("run")
if run is None:
    st.info("👈 Adjust settings in the sidebar and click 'Generate & Solve' to start.")
    st.markdown("### Preview of Realistic Input Data")
    preview = generate_problem(5, 7, seed=0)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Employee Pool (Sample)**")
        st.dataframe(employees_frame(preview)[["Name", "Skill", "Max_Hours_Per_Week"]])
    with col2:
        st.write("**Shift Requirements (staff per shift, per ward)**")
        st.dataframe(demand_frame(preview))
    st.stop()

problem, results = run
result = results[0]


def engine(r: SolveResult) -> str:
    return "🚀 NVIDIA cuOpt" if r.backend is Backend.GPU else "CPU (HiGHS)"


for warning in result.warnings:
    st.warning(warning)

if len(results) > 1:
    st.subheader("CPU vs GPU")
    st.dataframe(
        pd.DataFrame(
            {
                "Solver": engine(r),
                "Outcome": r.outcome.value,
                "Seconds": round(r.solve_seconds, 3),
                "Objective": r.objective,
                "Uncovered slots": sum(coverage_gaps(problem, r.assignments).values())
                if r.feasible
                else None,
            }
            for r in results
        ),
        hide_index=True,
    )

if result.outcome is Outcome.NO_SOLUTION:
    st.error(
        f"No schedule found within {time_limit} seconds ({result.solver_status}). "
        "Try a longer time limit, a gap tolerance, or the other solver."
    )
    st.stop()

took = f"with {engine(result)} in {result.solve_seconds:.3f} seconds"
if result.outcome is Outcome.OPTIMAL:
    st.success(f"Optimal schedule found {took}.")
else:
    st.warning(f"Time limit reached: best schedule found {took}, not proven optimal.")

gaps = coverage_gaps(problem, result.assignments)
if gaps:
    st.warning(
        f"{sum(gaps.values())} required staff slots could not be covered with the available "
        "staff. See **Coverage Analysis**."
    )

schedule = schedule_frame(problem, result.assignments)
tab1, tab2, tab3 = st.tabs(
    ["📅 Visual Schedule (Gantt)", "📊 Coverage Analysis", "📝 Detailed List"]
)

with tab1:
    st.subheader("Employee Schedule Timeline")
    if schedule.empty:
        st.info("Nobody is scheduled.")
    else:
        st.plotly_chart(gantt_figure(schedule), width="stretch")

with tab2:
    if gaps:
        st.subheader("Uncovered Staff Slots")
        st.dataframe(gaps_frame(problem, gaps), hide_index=True, width="stretch")
    if not schedule.empty:
        st.subheader("Shift Coverage Heatmap")
        st.plotly_chart(coverage_figure(schedule, list(problem.shifts)), width="stretch")
        st.markdown("**Skill Breakdown:**")
        st.bar_chart(schedule["Skill"].value_counts())

with tab3:
    export = schedule[
        ["Day", "Shift", "Employee_ID", "Employee", "Skill", "Start_Time", "End_Time"]
    ]
    st.dataframe(export, width="stretch")
    st.download_button(
        "Download Schedule CSV", export.to_csv(index=False), "optimized_schedule.csv", "text/csv"
    )
