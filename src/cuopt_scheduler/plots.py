"""Charts for the README from `cuopt-bench` CSVs.

    uv run cuopt-plots docs/benchmarks/scale.csv docs/benchmarks/levels.csv --out docs/img

Each chart is written twice, `<name>-light.png` and `<name>-dark.png`, for GitHub's
`<picture>` theme switch. Needs matplotlib (in the dev dependency group).

Colour follows the method (fixed order, validated for colour-vision deficiency); line style
is a second encoding of the device (GPU solid, CPU dashed), and every line is labelled
directly, so nothing depends on colour alone.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# The four methods the charts follow, in fixed colour order. Tables in the README cover all.
SERIES: dict[str, str] = {
    "highs-mip": "HiGHS MILP (CPU)",
    "cuopt-mip": "cuOpt MILP (GPU)",
    "highs-pdlp+round": "HiGHS PDLP + rounding (CPU)",
    "cuopt-pdlp+round": "cuOpt PDLP + rounding (GPU)",
}


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]


LIGHT = Theme(
    "light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
)
DARK = Theme(
    "dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
)


def load(paths: Sequence[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df.drop_duplicates(["wards", "level", "days", "seed", "method"], keep="last")


def gpu_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Per problem: best CPU method's time to a good schedule / best GPU method's.

    > 1 means the GPU got there first. A method that never got within reach counts as
    taking the whole time limit, so the ratio is then a lower (or upper) bound.
    """
    keys = ["wards", "employees", "level", "days", "seed"]
    t = df.assign(t=df["near_seconds"].fillna(df["seconds"]))
    best = pd.pivot_table(t, values="t", index=keys, columns="device", aggfunc="min")
    best = best.dropna(subset=["cpu", "gpu"])
    return best.assign(speedup=best["cpu"] / best["gpu"]).reset_index()


def _style(ax, theme: Theme) -> None:
    ax.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
    ax.tick_params(colors=theme.muted, labelcolor=theme.ink_secondary, which="both")
    ax.grid(True, which="major", color=theme.grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(theme.ink_secondary)
    ax.yaxis.label.set_color(theme.ink_secondary)
    ax.title.set_color(theme.ink)


def _log_axis(axis, unit: str = "", prefix: str = "") -> None:
    """Log scale with plain labels at 1-2-5 steps (no 2x10^2 notation)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{prefix}{v:,.{0 if v >= 1 else 1}f}{unit}")
    )
    axis.set_minor_formatter(NullFormatter())


def _end_labels(ax, ends: list[tuple[float, float, str]], theme: Theme) -> None:
    """Label lines at their right end, nudged apart so labels never overlap."""
    if not ends:
        return
    to_px = ax.transData.transform
    gap = 15.0  # px between label baselines
    placed = sorted((to_px((x, y))[1], x, y, text) for x, y, text in ends)
    last = -1e9
    for py, x, y, text in placed:
        target = max(py, last + gap)
        last = target
        ax.annotate(
            text,
            (x, y),
            xytext=(6, (target - py) * 72 / ax.figure.dpi),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=theme.ink_secondary,
        )


def _figure(theme: Theme, ncols: int = 1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, ncols, figsize=(5.2 * ncols + 1.6, 4.2), sharey=True, squeeze=False, dpi=150
    )
    fig.patch.set_facecolor(theme.surface)
    for ax in axes[0]:
        _style(ax, theme)
    return fig, list(axes[0])


def _method_lines(ax, data: pd.DataFrame, x: str, theme: Theme, limit: float) -> None:
    """One line per method: median time to a good schedule over seeds. Misses sit on the
    time-limit line as hollow markers."""
    ends: list[tuple[float, float, str]] = []
    for colour, (method, label) in zip(theme.series, SERIES.items(), strict=True):
        rows = data.loc[data["method"] == method]
        if rows.empty:
            continue
        med = pd.Series(rows.groupby(x)["near_seconds"].median()).sort_index()
        style = "-" if method.startswith(("cuopt", "cupdlpx")) else "--"
        hit = med.dropna()
        ax.plot(hit.index, hit.values, style, color=colour, linewidth=2, marker="o", markersize=5)
        missed = [i for i, v in med.items() if pd.isna(v)]
        ax.plot(
            missed,
            [limit] * len(missed),
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=theme.surface,
            markeredgecolor=colour,
            markeredgewidth=1.6,
        )
        if not hit.empty:
            ends.append((float(hit.index.to_numpy()[-1]), float(hit.iloc[-1]), label))
    ax.set_yscale("log")
    _log_axis(ax.yaxis, unit=" s")
    ax.set_ylim(top=limit * 1.6)
    ax.figure.canvas.draw()  # so data-to-pixel transforms are final for label placement
    _end_labels(ax, ends, theme)
    ax.axhline(limit, color=theme.axis, linewidth=1, linestyle=":")
    ax.annotate(
        f"{limit:g} s limit (hollow = not within 1% by then)",
        (0.01, limit),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7,
        color=theme.muted,
    )


def plot_time_to_good(df: pd.DataFrame, out: Path, theme: Theme, limit: float) -> Path:
    levels = sorted(df["level"].unique())
    shown = [levels[0], levels[-1]] if len(levels) > 1 else levels
    fig, axes = _figure(theme, len(shown))
    for ax, level in zip(axes, shown, strict=True):
        ax.set_xscale("log")
        _log_axis(ax.xaxis)
        _method_lines(ax, df.loc[df["level"] == level], "employees", theme, limit)
        ax.set_xlabel("Employees (34 per ward, 28 days)")
        ax.set_title(f"Difficulty level {level}", fontsize=10, loc="left")
    axes[0].set_ylabel("Seconds to a schedule within 1% of the best")
    fig.suptitle("Time to a good schedule as the problem grows", color=theme.ink, x=0.06, ha="left")
    return _save(fig, out / f"time-to-good-{theme.name}.png", theme)


def plot_levels(df: pd.DataFrame, out: Path, theme: Theme, limit: float, wards: int) -> Path:
    fig, (ax,) = _figure(theme)
    ax.set_xticks(sorted(df["level"].unique()))
    _method_lines(ax, df.loc[df["wards"] == wards], "level", theme, limit)
    ax.set_xlabel("Difficulty level (each adds rules)")
    ax.set_ylabel("Seconds to a schedule within 1% of the best")
    employees = int(df.loc[df["wards"] == wards, "employees"].iloc[0])
    fig.suptitle(
        f"Harder rules, same size ({employees:,} employees)",
        color=theme.ink,
        x=0.06,
        ha="left",
    )
    return _save(fig, out / f"levels-{theme.name}.png", theme)


def plot_speedup(df: pd.DataFrame, out: Path, theme: Theme) -> Path:
    s = gpu_speedup(df)
    fig, (ax,) = _figure(theme)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_axis(ax.xaxis)
    _log_axis(ax.yaxis, prefix="×")
    ends = []
    for colour, level in zip(theme.series, sorted(s["level"].unique()), strict=False):
        rows = pd.Series(
            s.loc[s["level"] == level].groupby("employees")["speedup"].median()
        ).sort_index()
        ax.plot(rows.index, rows.values, "-", color=colour, linewidth=2, marker="o", markersize=5)
        ends.append((float(rows.index.to_numpy()[-1]), float(rows.iloc[-1]), f"level {level}"))
    fig.canvas.draw()
    _end_labels(ax, ends, theme)
    ax.axhline(1, color=theme.muted, linewidth=1, linestyle="--")
    ax.annotate(
        "break-even: above this line the GPU is faster",
        (0.01, 1),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7,
        color=theme.muted,
    )
    ax.set_xlabel("Employees (34 per ward, 28 days)")
    ax.set_ylabel("Best CPU time ÷ best GPU time")
    fig.suptitle("GPU speed-up to a good schedule", color=theme.ink, x=0.06, ha="left")
    return _save(fig, out / f"speedup-{theme.name}.png", theme)


def _save(fig, path: Path, theme: Theme) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)
    return path


def plot_all(
    paths: Sequence[Path], out: Path, limit: float = 120, level_wards: int = 80
) -> list[Path]:
    df = load(paths)
    written = []
    for theme in (LIGHT, DARK):
        written.append(plot_time_to_good(df, out, theme, limit))
        written.append(plot_speedup(df, out, theme))
        if df.loc[df["wards"] == level_wards, "level"].nunique() > 1:
            written.append(plot_levels(df, out, theme, limit, level_wards))
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    parser.add_argument("--time-limit", type=float, default=120)
    parser.add_argument("--level-wards", type=int, default=80)
    args = parser.parse_args(argv)
    for path in plot_all(args.csv, args.out, args.time_limit, args.level_wards):
        print(path)


if __name__ == "__main__":
    main()
