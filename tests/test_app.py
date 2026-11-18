"""Smoke tests that drive the Streamlit app headlessly."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def force_cpu(at: AppTest) -> None:
    # Without cuOpt the toggle is already off and disabled.
    if not at.toggle[0].disabled:
        at.toggle[0].set_value(False)


def slider(at: AppTest, label: str):
    return next(s for s in at.slider if s.label.startswith(label))


def test_initial_page_shows_preview() -> None:
    at = app()
    assert not at.exception
    assert len(at.dataframe) == 2


def test_default_settings_cover_all_demand() -> None:
    at = app()
    force_cpu(at)
    at.button[0].click().run()
    assert not at.exception
    assert at.success
    assert "Optimal schedule found with CPU (HiGHS)" in at.success[0].value
    assert not at.warning
    assert len(at.tabs) == 4
    # The "How it was solved" tab: the LP bound proves this schedule optimal.
    assert any("bound proves this schedule optimal" in s.value for s in at.success)
    assert any("Rule" in df.value.columns for df in at.dataframe)


def test_understaffed_shows_uncovered_slots() -> None:
    at = app()
    slider(at, "Number of Employees").set_value(20)
    force_cpu(at)
    at.button[0].click().run()
    assert not at.exception
    assert any("could not be covered" in w.value for w in at.warning)
    assert any("Missing" in df.value.columns for df in at.dataframe)
    # The explanation tab says how optimality was proven: by the LP bound or by search.
    verdicts = [m.value for m in [*at.success, *at.info]]
    assert any("bound proves" in v or "closed it by **search**" in v for v in verdicts)


def test_compare_backends_when_gpu_available() -> None:
    import pytest

    from cuopt_scheduler import cuopt_available

    if not cuopt_available():
        pytest.skip("cuOpt / NVIDIA GPU not available")
    at = app()
    at.checkbox[0].check()
    at.button[0].click().run()
    assert not at.exception
    table = next(df.value for df in at.dataframe if "Solver" in df.value.columns)
    assert len(table) == 2


def test_hardest_level_solves() -> None:
    at = app()
    at.select_slider[0].set_value(3)
    force_cpu(at)
    at.button[0].click().run()
    assert not at.exception
    assert "dislikes" in " ".join(c.value for c in at.caption)
    assert at.success
