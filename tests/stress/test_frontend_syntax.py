"""The inline JavaScript has to parse.

A syntax error in a template's <script> block is completely invisible from the
server: the page still returns 200, the HTML is still well formed, and every
assertion about status codes and absent tracebacks passes. The only symptom is
that the chart never draws - which is exactly what happened when an edit
replaced the closing of an IIFE without replacing its opening.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEMPLATES = Path(__file__).resolve().parents[2] / "circa" / "web" / "templates"
SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)

PAGES = ["/", "/trends", "/model", "/settings"]


@pytest.fixture
def client():
    from circa.web.app import create_app

    return TestClient(create_app(with_scheduler=False))


def _inline_scripts(html: str) -> list[str]:
    return [m.group(1) for m in SCRIPT.finditer(html) if m.group(1).strip()]


def _balanced(source: str) -> bool:
    """Crude delimiter check, used when node is unavailable.

    Ignores strings, template literals, regexes and comments well enough for the
    unbalanced-paren case this exists to catch.
    """
    stripped = re.sub(r"//[^\n]*", "", source)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S)
    stripped = re.sub(r"'(?:\\.|[^'\\])*'", "''", stripped)
    stripped = re.sub(r'"(?:\\.|[^"\\])*"', '""', stripped)
    stripped = re.sub(r"`(?:\\.|[^`\\])*`", "``", stripped)
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in stripped:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs and (not stack or stack.pop() != pairs[ch]):
            return False
    return not stack


@pytest.mark.parametrize("path", PAGES)
def test_inline_scripts_parse(client, path):
    html = client.get(path).text
    scripts = _inline_scripts(html)
    node = shutil.which("node")
    for i, source in enumerate(scripts):
        if node:
            result = subprocess.run(
                [node, "--input-type=module", "--check"],
                input=source, capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, (
                f"{path} inline script #{i} does not parse:\n{result.stderr[:800]}"
            )
        else:
            assert _balanced(source), f"{path} inline script #{i} has unbalanced delimiters"


@pytest.mark.parametrize("path", PAGES)
def test_pages_have_no_unrendered_template_syntax(client, path):
    html = client.get(path).text
    for marker in ("{{", "{%", "Undefined", "jinja2."):
        assert marker not in html, f"{path} leaked {marker!r}"


def test_the_chart_bootstrap_is_actually_invoked(client):
    """A function that parses but is never called draws nothing."""
    html = client.get("/").text
    script = "\n".join(_inline_scripts(html))
    assert "function draw" in script
    assert re.search(r"^\s*draw\(\);", script, re.M), "draw() is defined but never called"


def test_every_template_file_parses_as_javascript():
    """Catches a broken script in a page a route test does not happen to hit."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    for template in TEMPLATES.rglob("*.html"):
        raw = template.read_text()
        for i, source in enumerate(_inline_scripts(raw)):
            # Strip Jinja expressions; they are server-side and never reach the
            # browser, but they are not valid JavaScript on their own.
            cleaned = re.sub(r"\{\{.*?\}\}", "0", source, flags=re.S)
            cleaned = re.sub(r"\{%.*?%\}", "", cleaned, flags=re.S)
            result = subprocess.run(
                [node, "--input-type=module", "--check"],
                input=cleaned, capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, (
                f"{template.name} script #{i}:\n{result.stderr[:800]}"
            )


def test_every_css_variable_a_chart_reads_is_actually_defined():
    """`--energy` was never defined. `getPropertyValue` answered "", ECharts
    drew the first frame off its own default palette so it looked fine, then
    derived an emphasis colour from "" on hover and drew nothing - the curve
    vanished the moment you pointed at it, and came back when you moved away.

    The templates now fall back to a visible grey, but a fallback is a
    consolation prize: the variable should exist.
    """
    import re
    from pathlib import Path

    web = Path(__file__).resolve().parents[2] / "circa" / "web"
    css = (web / "static" / "app.css").read_text()
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", css))

    missing: list[str] = []
    for template in sorted((web / "templates").glob("*.html")):
        for name in set(re.findall(r"C\(\s*'(--[a-z0-9-]+)'", template.read_text())):
            if name not in defined:
                missing.append(f"{template.name} reads {name}")
    assert not missing, missing


def test_the_now_marker_ticks_even_with_reduced_motion():
    """A clock is not decoration. Behind `if (!REDUCED)` the "now" line froze at
    page load for anyone who has asked for less motion, and then quietly lied
    for as long as the tab stayed open."""
    from pathlib import Path

    template = (
        Path(__file__).resolve().parents[2]
        / "circa" / "web" / "templates" / "today.html"
    ).read_text()

    tick = template.index("setInterval(")
    guard = template.rfind("if (!REDUCED)", 0, tick)
    assert guard == -1 or template.index("}", guard) < tick, (
        "the marker tick is inside a reduced-motion guard"
    )
