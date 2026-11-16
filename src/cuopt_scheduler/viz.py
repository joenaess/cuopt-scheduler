"""Plotly figures for a schedule DataFrame (see frames.schedule_frame)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SHIFT_COLORS = {"Day": "#00CC96", "Evening": "#AB63FA", "Night": "#19D3F3"}


def gantt_figure(schedule: pd.DataFrame) -> go.Figure:
    fig = px.timeline(
        schedule,
        x_start="Start_Time",
        x_end="End_Time",
        y="Employee",
        color="Shift",
        hover_data=["Employee_ID", "Skill", "Day"],
        title="Staff Shifts Overview",
        color_discrete_map=SHIFT_COLORS,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def coverage_table(schedule: pd.DataFrame, shifts: list[str]) -> pd.DataFrame:
    """Staff count per shift (rows) and day (columns), in chronological order."""
    day_order = schedule.sort_values("Day_Index")["Day"].unique().tolist()
    return (
        schedule.pivot_table(index="Shift", columns="Day", values="Employee_ID", aggfunc="count")
        .reindex(index=shifts, columns=day_order)
        .fillna(0)
        .astype(int)
    )


def coverage_figure(schedule: pd.DataFrame, shifts: list[str]) -> go.Figure:
    return px.imshow(
        coverage_table(schedule, shifts),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Number of Staff Assigned per Shift",
    )
