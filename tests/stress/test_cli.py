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


def test_doctor_notices_the_modelling_stack_is_missing(db, monkeypatch):
    """`pip install -e .` without `[science]` gives a collector that fills the
    database forever and estimates nothing. Doctor used to report that setup as
    healthy; the first symptom was a ModuleNotFoundError out of a scheduler job
    hours later."""
    from circa import cli

    monkeypatch.setattr(cli, "missing_science", lambda: ["numpy", "scipy"])
    out = CliRunner().invoke(app, ["doctor"]).output
    assert "Modelling stack not installed" in out, out
    # The fix line is meant to be copied, so it has to survive table wrapping.
    assert '.[science]' in out, out


def test_the_modelling_stack_is_present_in_this_environment():
    """And the check itself has to be right: these tests import numpy, so a
    non-empty answer here would mean the detection is broken, not the venv."""
    from circa.cli import missing_science

    assert missing_science() == []


def test_every_documented_install_command_includes_the_modelling_stack():
    """Both published install paths omitted `[science]`, so anyone following the
    README got a model that could not run."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in ("README.md", "deploy/setup-vm.sh"):
        text = (root / name).read_text()
        for line in re.findall(r"^.*pip install.*$", text, re.M):
            if "-e" not in line:
                continue
            assert "science" in line, f"{name}: {line.strip()}"


def test_the_vm_setup_script_has_no_command_substitution_in_its_heredocs():
    """`ENVEOF` is deliberately unquoted so $DATA_DIR and $SECRET expand - which
    means backticks and $(...) inside it are executed too, by a script running
    as root. A comment mentioning 'circa doctor' in backticks was run for real
    on the first deployment; it was harmless, and the next one might not be.
    """
    import re
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[2] / "deploy" / "setup-vm.sh"
    ).read_text()

    # Every heredoc body, paired with whether its delimiter was quoted. A quoted
    # delimiter ('LOGEOF') disables expansion, so those bodies are safe.
    pattern = re.compile(r"<<(?P<q>'?)(?P<tag>\w+)(?P=q)\n(?P<body>.*?)\n(?P=tag)\n", re.S)
    offenders = []
    for m in pattern.finditer(script):
        if m.group("q"):
            continue                      # quoted delimiter: nothing expands
        for line in m.group("body").splitlines():
            if "`" in line or "$(" in line:
                offenders.append(f"{m.group('tag')}: {line.strip()}")
    assert not offenders, offenders


def test_config_reads_the_deployed_env_file_as_well_as_a_local_one(tmp_path, monkeypatch):
    """On the VM the config lives at /etc/circa/circa.env, which systemd reads
    itself - so `circa auth` and `circa doctor` run by hand saw no client id, no
    timezone and no coordinates, and called a correctly configured machine
    broken. Both paths are consulted now, with the local checkout winning."""
    from circa.config import Settings

    assert "/etc/circa/circa.env" in Settings.model_config["env_file"]
    assert ".env" in Settings.model_config["env_file"]
    # Local last, so working in a checkout still overrides a deployed file.
    assert list(Settings.model_config["env_file"]).index(".env") == 1


def test_a_local_env_file_overrides_the_deployed_one(tmp_path, monkeypatch):
    from circa.config import Settings

    deployed = tmp_path / "circa.env"
    deployed.write_text("CIRCA_TIMEZONE=Europe/London\nCIRCA_LATITUDE=51.5072\n")
    local = tmp_path / ".env"
    local.write_text("CIRCA_TIMEZONE=America/Chicago\n")

    for key in ("CIRCA_TIMEZONE", "CIRCA_LATITUDE"):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=(str(deployed), str(local)))
    assert s.timezone == "America/Chicago", "local .env should win"
    assert s.latitude == 51.5072, "values only in the deployed file still apply"
