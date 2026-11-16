"""Streamlit UI. All scheduling logic lives in the cuopt_scheduler package."""

import streamlit as st

from cuopt_scheduler import Backend, cuopt_available, generate_problem, solve
from cuopt_scheduler.frames import demand_frame, employees_frame, schedule_frame
from cuopt_scheduler.viz import coverage_figure, gantt_figure

st.set_page_config(page_title="NVIDIA cuOpt Scheduler", layout="wide")
st.title("🏥 Intelligent Staff Scheduler")
st.caption("Powered by NVIDIA cuOpt & Streamlit")

with st.sidebar:
    st.header("1. Data Generation")
    num_emp = st.slider("Number of Employees", 10, 100, 25)
    duration = st.slider("Schedule Duration (Days)", 7, 21, 7)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.divider()

    st.header("2. Optimization Settings")
    gpu_ok = cuopt_available()
    use_gpu = st.toggle(
        "Use NVIDIA GPU (cuOpt)",
        value=gpu_ok,
        disabled=not gpu_ok,
        help=None if gpu_ok else "cuOpt not available. Install with `uv sync --extra gpu`.",
    )

    st.info("""
    **What can I alter in this simulation?**

    In a real deployment, you can adjust:
    * **Staffing Rules:** Min/Max staff per shift.
    * **Skills:** Required certifications (RN vs CNA).
    * **Constraints:** Max overtime, union rules, consecutive work days.
    * **Objectives:** Minimize cost vs. Maximize employee preference.
    """)

    if st.button("Generate & Solve", type="primary"):
        problem = generate_problem(num_emp, duration, seed=int(seed))
        with st.spinner("Optimizing schedule..."):
            result = solve(problem, Backend.GPU if use_gpu else Backend.CPU)
        # Keep results across reruns (e.g. clicking the download button).
        st.session_state["run"] = (problem, result)

run = st.session_state.get("run")
if run is None:
    st.info("👈 Adjust settings in the sidebar and click 'Generate & Solve' to start.")
    st.markdown("### Preview of Realistic Input Data")
    preview = generate_problem(5, 7, seed=0)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Employee Pool (Sample)**")
        st.dataframe(employees_frame(preview)[["Name", "Skill", "Max_Hours"]])
    with col2:
        st.write("**Shift Requirements (staff per shift)**")
        st.dataframe(demand_frame(preview))
    st.stop()

problem, result = run
for warning in result.warnings:
    st.warning(warning)

if not result.feasible:
    st.error(
        f"Optimization failed ({result.status}). The constraints might be too tight, "
        "e.g. not enough staff for the required shifts. Try increasing the employee count."
    )
    st.stop()

engine = "🚀 NVIDIA cuOpt" if result.backend is Backend.GPU else "CPU (CBC)"
st.success(f"Solved with {engine} in {result.solve_seconds:.4f} seconds!")

schedule = schedule_frame(problem, result.assignments)
tab1, tab2, tab3 = st.tabs(
    ["📅 Visual Schedule (Gantt)", "📊 Coverage Analysis", "📝 Detailed List"]
)

with tab1:
    st.subheader("Employee Schedule Timeline")
    st.plotly_chart(gantt_figure(schedule), width="stretch")

with tab2:
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
