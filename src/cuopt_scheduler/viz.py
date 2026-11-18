"""Plotly figures for a schedule DataFrame (see frames.schedule_frame)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cuopt_scheduler.frames import WEEKDAYS

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


def decision_grid_figure(grid: pd.DataFrame, title: str) -> go.Figure:
    """Heatmap of yes/no decisions (see explain.decision_grid): dark = yes, light = no,
    fractions printed in the cell, and ✕ where the person is unavailable."""
    days = grid.columns.get_level_values("Day")
    shifts = grid.columns.get_level_values("Shift")
    labels = [f"{WEEKDAYS[d % 7]} {d + 1}<br>{s[0]}" for d, s in zip(days, shifts, strict=True)]
    z = grid.to_numpy(dtype=float)
    text = [
        ["✕" if np.isnan(v) else (f"{v:.2f}" if 0.005 <= v <= 0.995 else "") for v in row]
        for row in z
    ]
    hover = [
        [
            f"{name}<br>Day {d + 1} ({WEEKDAYS[d % 7]}), {s}<br>"
            + ("unavailable: fixed to 0" if np.isnan(v) else f"value {v:.2f}")
            for d, s, v in zip(days, shifts, row, strict=True)
        ]
        for name, row in zip(grid.index, z, strict=True)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=list(grid.index),
            zmin=0,
            zmax=1,
            colorscale=[[0, "#eef4fb"], [1, "#1c5aa6"]],
            text=text,
            texttemplate="%{text}",
            hovertext=hover,
            hoverinfo="text",
            xgap=2,
            ygap=2,
            colorbar={"title": "value", "tickvals": [0, 0.5, 1]},
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title, height=120 + 28 * len(grid), margin={"t": 60, "b": 20})
    return fig
