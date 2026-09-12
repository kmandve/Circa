"""Every CLI command, against an empty database.

These are the commands a person reaches for when something is already wrong, so
they have to work when there is no data, no token and no network - and say
something useful rather than traceback.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from circa.cli import app

runner = CliRunner()

# Commands that must be safe to run with no data, no credentials and no network.
OFFLINE = ["init", "status", "retention", "renormalize", "backtest"]


@pytest.mark.parametrize("command", OFFLINE)
def test_command_runs_on_an_empty_database(command):
    result = runner.invoke(app, [command])
    assert result.exit_code == 0, (
        f"`circa {command}` exited {result.exit_code}\n{result.output}\n{result.exception!r}"
    )
    assert "Traceback" not in result.output


def test_doctor_reports_problems_without_crashing():
    """`doctor` exits non-zero when it finds issues - that is its job - but it
    must still produce a readable report rather than a traceback."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1), result.output
    assert "Traceback" not in result.output
    assert "not authenticated" in result.output
    assert "tier 0" in result.output


@pytest.mark.parametrize("command", OFFLINE + ["doctor", "auth", "sync", "probe", "run", "calendars", "serve"])
def test_help_works_for_every_command(command):
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output


def test_run_without_credentials_fails_cleanly(monkeypatch):
    """No token yet is the normal state before setup, not a crash."""
    result = runner.invoke(app, ["run", "--no-calendar"])
    assert result.exit_code in (0, 1), result.output
    assert "Traceback" not in result.output


def test_status_reports_the_empty_state_in_words():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert result.output.strip(), "status printed nothing at all"


def test_doctor_does_not_print_secrets(monkeypatch):
    """`doctor` is the command people paste into a chat window."""
    monkeypatch.setenv("CIRCA_GOOGLE_CLIENT_SECRET", "super-secret-value-xyz")
    from circa.config import get_settings

    get_settings.cache_clear()
    result = runner.invoke(app, ["doctor"])
    assert "super-secret-value-xyz" not in result.output


def test_unknown_command_is_rejected():
    result = runner.invoke(app, ["definitely-not-a-command"])
    assert result.exit_code != 0


def test_doctor_notices_the_example_coordinates_are_still_in_place(db, monkeypatch):
    """The light proxy clamps step->lux to the clear-sky irradiance at your
    latitude, so shipping with the example city is silent and wrong: sunrise
    lands at the wrong hour and every phase-delaying evening is misjudged."""
    from circa.cli import EXAMPLE_COORDINATES
    from circa.config import get_settings

    lat, lon = EXAMPLE_COORDINATES
    monkeypatch.setenv("CIRCA_LATITUDE", str(lat))
    monkeypatch.setenv("CIRCA_LONGITUDE", str(lon))
    get_settings.cache_clear()

    out = CliRunner().invoke(app, ["doctor"]).output
    assert "still the example values" in out, out


def test_doctor_is_happy_with_real_coordinates(db, monkeypatch):
    from circa.config import get_settings

    monkeypatch.setenv("CIRCA_LATITUDE", "35.6895")
    monkeypatch.setenv("CIRCA_LONGITUDE", "139.6917")
    get_settings.cache_clear()

    out = CliRunner().invoke(app, ["doctor"]).output
    assert "still the example values" not in out, out
