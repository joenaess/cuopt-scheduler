from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cuopt_scheduler.plots import gpu_speedup, markdown_table, plot_all

pytest.importorskip("matplotlib")


def rows() -> pd.DataFrame:
    out = []
    for wards, employees in ((5, 170), (20, 680)):
        for level in (0, 3):
            for method, device, near in (
                ("highs-mip", "cpu", 2.0 * wards),
                ("cuopt-mip", "gpu", None),  # never within 1%
                ("highs-pdlp+round", "cpu", 0.2 * wards),
                ("cuopt-pdlp+round", "gpu", 0.1 * wards),
            ):
                out.append(
                    {
                        "wards": wards,
                        "employees": employees,
                        "level": level,
                        "days": 28,
                        "seed": 0,
                        "method": method,
                        "device": device,
                        "outcome": "feasible",
                        "seconds": 120.0 if near is None else near,
                        "near_seconds": near,
                    }
                )
    return pd.DataFrame(out)


def test_gpu_speedup_compares_each_devices_best() -> None:
    s = gpu_speedup(rows())
    # Best CPU is PDLP (0.2 * wards), best GPU is PDLP (0.1 * wards): 2x everywhere.
    assert list(s["speedup"]) == pytest.approx([2.0] * 4)


def test_plot_all_writes_light_and_dark_charts(tmp_path: Path) -> None:
    csv = tmp_path / "bench.csv"
    rows().to_csv(csv, index=False)
    written = plot_all([csv], tmp_path / "img", level_wards=20)
    names = sorted(p.name for p in written)
    assert names == sorted(
        f"{chart}-{theme}.png"
        for chart in ("time-to-good", "speedup", "levels")
        for theme in ("light", "dark")
    )
    assert all(p.stat().st_size > 10_000 for p in written)


def test_markdown_table_marks_the_fastest_misses_and_failures() -> None:
    df = rows()
    failed = (df["wards"] == 20) & (df["level"] == 3) & (df["method"] == "cuopt-mip")
    df.loc[failed, "outcome"] = "failed: cuOpt failed (Memory allocation failed)"
    lines = markdown_table(df).splitlines()
    assert lines[0].startswith("| Employees | Level | HiGHS MILP (CPU) | cuOpt MILP (GPU) |")
    assert len(lines) == 2 + 4  # header, rule, one row per problem
    # 170 employees, level 0: HiGHS MILP 10 s, cuOpt MILP missed, rounding 1 s and 0.5 s.
    assert lines[2] == "| 170 | 0 | 10 | – |  |  | 1 | **0.5** |"
    assert "failed (GPU memory)" in lines[5]


def test_markdown_table_of_time_to_a_proven_optimum() -> None:
    df = rows()
    proven = df["method"] == "highs-mip"
    df.loc[proven, "outcome"] = "optimal"
    lines = markdown_table(df, proof=True).splitlines()
    # Only HiGHS MILP proved anything; the rest have a schedule but no proof.
    assert lines[2] == "| 170 | 0 | **10** | – |  |  | – | – |"


def test_a_method_that_never_got_close_counts_as_the_longest_run() -> None:
    df = rows()
    one = (df["wards"] == 5) & (df["level"] == 0)
    # HiGHS PDLP + rounding finished first, but with a poor schedule.
    poor = one & (df["method"] == "highs-pdlp+round")
    df.loc[poor, ["near_seconds", "seconds"]] = [np.nan, 0.01]
    s = gpu_speedup(df)
    row = s[(s["wards"] == 5) & (s["level"] == 0)].iloc[0]
    assert row["cpu"] == 10.0  # HiGHS MILP's time, not the poor run's 0.01 s
