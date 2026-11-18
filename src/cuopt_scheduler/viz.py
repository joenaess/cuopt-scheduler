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


# What kind of work a phase is, by its label's start; colours from the README charts'
# validated palette, so each kind keeps its colour wherever it appears.
PHASE_KINDS = (
    ("Building", "Build the model", "#898781"),
    ("LP relaxation", "LP relaxation", "#2a78d6"),
    ("Rounding", "Rounding", "#1baf7a"),
    ("Lower bound", "Proof check (LP bound)", "#eda100"),
    ("Branch and bound", "Branch and bound (MILP search)", "#eb6834"),
)


def _phase_kind(label: str) -> tuple[str, str]:
    for prefix, name, color in PHASE_KINDS:
        if label.startswith(prefix):
            return name, color
    return "Other", "#52514e"


def phase_timeline_figure(runs: list[tuple[str, list[tuple[str, float]]]]) -> go.Figure:
    """One horizontal bar per run, split into its phases (label, seconds) end to end."""
    fig = go.Figure()
    shown: set[str] = set()
    for name, phases in runs:
        start = 0.0
        for label, seconds in phases:
            kind, color = _phase_kind(label)
            fig.add_bar(
                y=[name],
                x=[seconds],
                base=[start],
                orientation="h",
                marker={"color": color, "line": {"width": 1, "color": "white"}},
                name=kind,
                legendgroup=kind,
                showlegend=kind not in shown,
                hovertemplate=f"{label}<br>{seconds:.3f} s<extra></extra>",
            )
            shown.add(kind)
            start += seconds
        fig.add_annotation(y=name, x=start, text=f" {start:.2f} s", showarrow=False, xanchor="left")
    fig.update_layout(
        barmode="overlay",
        title="Where the time went",
        xaxis_title="seconds",
        height=140 + 70 * len(runs),
        margin={"t": 60, "b": 40, "r": 60},
        legend={"orientation": "h", "y": -0.35 if len(runs) > 1 else -0.6},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def convergence_figure(
    runs: list[tuple[str, tuple[tuple[float, float], ...], float, bool]],
) -> go.Figure:
    """Best schedule cost over time, one step line per run (name, incumbents, total seconds,
    proven). Log scales: costs fall by orders of magnitude as uncovered slots are filled."""
    fig = go.Figure()
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    span: list[float] = []
    for (name, incumbents, total, proven), color in zip(runs, colors, strict=False):
        if not incumbents:
            continue
        times = [max(t, 1e-3) for t, _ in incumbents]
        costs = [c for _, c in incumbents]
        span += [times[0], max(total, times[-1])]
        best = [min(costs[: i + 1]) for i in range(len(costs))]
        fig.add_scatter(
            x=[*times, max(total, times[-1])],
            y=[*best, best[-1]],
            mode="lines+markers",
            line={"shape": "hv", "color": color, "width": 2},
            marker={"size": 8},
            name=name,
            hovertemplate="%{x:.3f} s: cost %{y:,.0f}<extra>" + name + "</extra>",
        )
        fig.add_annotation(
            x=np.log10(max(total, times[-1])),
            y=np.log10(best[-1]),
            text="✔ proven optimal" if proven else "stopped: not proven",
            showarrow=True,
            arrowhead=0,
            arrowcolor=color,
            standoff=6,
            ax=0,
            ay=-30,
            font={"color": color},
        )
    ticks = [m * 10.0**e for e in range(-3, 3) for m in (1, 2, 5)]
    if span:
        lo, hi = np.log10(min(span)) - 0.15, np.log10(max(span)) + 0.25
        ticks = [v for v in ticks if lo <= np.log10(v) <= hi] or ticks
        fig.update_xaxes(range=[lo, hi])
    fig.update_xaxes(
        type="log",
        title="seconds into the solve",
        tickvals=ticks,
        ticktext=[f"{v:.3g} s" for v in ticks],
    )
    fig.update_yaxes(
        type="log", title="cost of the best schedule so far", dtick=1, tickformat=",.0f"
    )
    fig.update_layout(
        title="Best schedule found over time",
        height=380,
        margin={"t": 60},
        legend={"orientation": "h", "y": -0.25},
    )
    return fig
