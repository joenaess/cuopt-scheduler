from pathlib import Path

import pandas as pd
import pytest

from cuopt_scheduler.plots import gpu_speedup, plot_all

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
