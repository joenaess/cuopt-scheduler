"""Streamlit UI. All scheduling logic lives in the cuopt_scheduler package."""

import dataclasses

import numpy as np
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
from cuopt_scheduler.explain import (
    Explanation,
    cost_breakdown,
    decision_grid,
    explain,
    is_proven,
    schedule_vector,
)
from cuopt_scheduler.frames import demand_frame, employees_frame, gaps_frame, schedule_frame
from cuopt_scheduler.lp_round import LPEngine
from cuopt_scheduler.viz import coverage_figure, decision_grid_figure, gantt_figure

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
        with st.spinner("Preparing the explanation (LP relaxation)..."):
            explanation = explain(problem, primary)
        # Keep results across reruns (e.g. clicking the download button).
        st.session_state["run"] = (problem, results, explanation)

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

problem, results, explanation = run
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
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📅 Visual Schedule (Gantt)",
        "📊 Coverage Analysis",
        "📝 Detailed List",
        "🎓 How it was solved",
    ]
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


def how_it_was_solved(result: SolveResult, ex: Explanation) -> None:
    model = ex.model
    st.markdown(
        "This schedule was found by solving a **mixed-integer linear program (MILP)**: "
        "a list of yes/no **decisions**, **rules** they must obey, and a **cost** to make "
        "as small as possible. Here is each part, for the problem you just solved."
    )

    st.subheader("1 · Decisions")
    st.markdown(
        "Each cell is one decision: *does this person work this shift on this day?* "
        "**1 = yes**, **0 = no**. ✕ marks shifts the person can't work, so that decision is "
        "fixed to no."
    )
    c = st.columns(3)
    c[0].metric("Yes/no decisions", f"{ex.num_decisions:,}")
    c[1].metric("Fixed to no (unavailable)", f"{ex.num_fixed:,}")
    c[2].metric(
        "Shortfall counters",
        f"{model.num_cols - ex.num_decisions:,}",
        help=(
            "One per day, shift and skill: how many required staff are missing. "
            "They let the model say 'not enough staff' instead of having no answer."
        ),
    )
    values = schedule_vector(model, result.assignments)
    grid = decision_grid(problem, model, values)
    days = len(grid.columns.get_level_values("Day").unique())
    shown = (
        f"first {len(grid)} of {len(problem.employees)} people, "
        f"first {days} of {problem.num_days} days"
    )
    st.plotly_chart(
        decision_grid_figure(
            grid, f"The schedule as decisions ({shown}; D/E/N = Day/Evening/Night)"
        ),
        width="stretch",
    )

    st.subheader("2 · Rules")
    st.markdown(
        "Every rule becomes a set of **linear constraints**: sums of decisions that must "
        "stay within a limit. For example, *at most one shift per person per day* is "
        "one constraint per person per day: Day + Evening + Night ≤ 1."
    )
    rules = pd.DataFrame(model.rules, columns=pd.Index(["Rule", "Constraints"]))
    st.dataframe(rules, hide_index=True, width="stretch")
    st.caption(f"{ex.num_constraints:,} constraints in total.")

    st.subheader("3 · Cost")
    st.markdown(
        "The solver minimises one number. A missing staff member costs far more than a "
        "shift worked, so covering demand always comes first; then it uses as few (and as "
        "few disliked) shifts as it can."
    )
    breakdown = cost_breakdown(model, values)
    st.dataframe(breakdown, hide_index=True, width="stretch")
    objective = float(model.cost @ values)  # = the sum of the parts above
    st.caption(f"Total cost: **{objective:,.0f}**")

    st.subheader("4 · The shortcut: relax, round, prove")
    on_gpu = ex.engine is LPEngine.CUOPT_PDLP
    st.markdown(
        "Searching every combination of yes/no answers is hopeless "
        f"(2^{ex.num_decisions - ex.num_fixed:,} of them). Instead, first **relax** the "
        "problem: allow fractions, like *half a shift*. That is a plain linear program (LP), "
        + (
            "which the GPU solves fast with cuOpt's PDLP (a first-order method built for GPUs)."
            if on_gpu
            else "which the CPU solves with HiGHS's simplex method."
        )
    )
    if ex.share_already_whole >= 0.9:
        st.markdown("Here the LP's answer is already almost all 0s and 1s.")
    else:
        st.markdown(
            "Here the LP's answer is full of fractions. Many people are interchangeable "
            "(same skill, free that day), so the LP splits a shift between them, like "
            "0.33 each for three equally good choices. That's fine: what matters is that "
            "its cost is as low as any real schedule's can be, and rounding picks one of "
            "the equally good options."
        )
    c = st.columns(3)
    c[0].metric("LP solved in", f"{ex.lp_seconds:.2f} s", help=f"Engine: {ex.engine}")
    c[1].metric("LP decisions already 0 or 1", f"{ex.share_already_whole:.0%}")
    changed = int(np.sum(np.abs(ex.rounded - ex.lp)[: ex.num_decisions] > 1e-3))
    c[2].metric("Decisions rounding changed", f"{changed:,}")
    st.plotly_chart(
        decision_grid_figure(
            decision_grid(problem, model, ex.lp), "Relaxed (LP): fractions allowed"
        ),
        width="stretch",
    )
    st.plotly_chart(
        decision_grid_figure(
            decision_grid(problem, model, ex.rounded), "Rounded: a valid schedule"
        ),
        width="stretch",
    )
    st.markdown(
        "**Rounding** turns the fractions into yes/no, filling each shift with the people "
        "the LP liked most, and never adds anyone a rule doesn't allow. The result is always "
        "a valid schedule."
    )

    st.markdown(
        "**Proving it is the best.** The LP also gives a *lower bound*: no schedule, "
        "however clever, can cost less. If our schedule's cost is within 1 of it (costs "
        "are whole numbers), nothing cheaper exists."
    )
    c = st.columns(3)
    c[0].metric("This schedule costs", f"{objective:,.0f}")
    c[1].metric("No schedule can cost less than", f"{ex.bound:,.2f}")
    c[2].metric("Gap", f"{objective - ex.bound:,.2f}")
    if is_proven(objective, ex.bound):
        st.success("Gap below 1: the LP bound proves this schedule optimal. No search needed.")
    elif result.outcome is Outcome.OPTIMAL:
        st.info(
            "The LP bound alone leaves a gap here, so the MILP solver closed it by "
            "**search** (branch and bound: trying yes and no for undecided cells and "
            "discarding branches that can't beat the best schedule found)."
        )
    else:
        st.warning(
            "Not proven: the solver stopped at the time limit with this gap. A cheaper "
            "schedule may exist, but it can't be cheaper than the bound."
        )


with tab4:
    how_it_was_solved(result, explanation)
