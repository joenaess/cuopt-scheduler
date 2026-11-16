from cuopt_scheduler import generate_problem, solve
from cuopt_scheduler.frames import (
    day_label,
    demand_frame,
    employees_frame,
    gaps_frame,
    schedule_frame,
)
from cuopt_scheduler.viz import coverage_table, gantt_figure


def test_day_label_starts_on_monday() -> None:
    assert day_label(0) == "Day 1 (Mon)"
    assert day_label(9) == "Day 10 (Wed)"


def test_input_frames() -> None:
    p = generate_problem(5, 7, seed=0)
    assert list(employees_frame(p).index) == [e.id for e in p.employees]
    assert list(demand_frame(p).index) == ["Day", "Evening", "Night"]


def test_schedule_and_coverage_in_chronological_order() -> None:
    p = generate_problem(100, 10, seed=4)
    result = solve(p)
    assert result.feasible
    schedule = schedule_frame(p, result.assignments)
    assert len(schedule) == len(result.assignments)
    assert (schedule["End_Time"] - schedule["Start_Time"]).dt.total_seconds().eq(8 * 3600).all()

    cov = coverage_table(schedule, list(p.shifts))
    # Day 10 must come last, not between Day 1 and Day 2 (lexicographic sort bug).
    assert list(cov.columns) == [day_label(d) for d in range(10)]
    assert list(cov.index) == ["Day", "Evening", "Night"]
    assert cov.loc["Day"].min() >= 7  # 2 RN + 2 LPN + 3 CNA

    traces = gantt_figure(schedule).to_dict()["data"]
    assert {t["name"] for t in traces} == {"Day", "Evening", "Night"}


def test_gaps_frame() -> None:
    p = generate_problem(5, 7, seed=0)
    df = gaps_frame(p, {(9, "Night", "RN"): 1, (0, "Day", "CNA"): 2})
    assert df.to_dict("records") == [
        {
            "Day": "Day 1 (Mon)",
            "Shift": "Day",
            "Skill": "CNA (Cert. Nursing Asst)",
            "Required": 3,
            "Missing": 2,
        },
        {
            "Day": "Day 10 (Wed)",
            "Shift": "Night",
            "Skill": "RN (Registered Nurse)",
            "Required": 1,
            "Missing": 1,
        },
    ]
    assert gaps_frame(p, {}).empty
