"""Smoke tests that drive the Streamlit app headlessly."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")


def force_cpu(at: AppTest) -> None:
    # Without cuOpt the toggle is already off and disabled.
    if not at.toggle[0].disabled:
        at.toggle[0].set_value(False)


def test_initial_page_shows_preview() -> None:
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert len(at.dataframe) == 2


def test_generate_and_solve_on_cpu() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.slider[0].set_value(80)
    force_cpu(at)
    at.button[0].click().run()
    assert not at.exception
    assert at.success and "CPU (CBC)" in at.success[0].value
    assert len(at.tabs) == 3


def test_infeasible_shows_error() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.slider[0].set_value(10)
    force_cpu(at)
    at.button[0].click().run()
    assert not at.exception
    assert at.error and "Infeasible" in at.error[0].value
