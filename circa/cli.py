"""Circa command line."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from circa.config import get_settings
from circa.logging_setup import configure_logging
from circa.timefmt import clock_from, stamp

app = typer.Typer(
    add_completion=False,
    help="Circa - a personal circadian phase engine for Fitbit Air.",
    no_args_is_help=True,
)
console = Console()


def _bootstrap() -> None:
    configure_logging()
    from circa.db.session import init_db

    init_db()


# ---------------------------------------------------------------------------


@app.command()
def init() -> None:
    """Create the database and show where Circa keeps its data."""
    _bootstrap()
    settings = get_settings()
    console.print(
        Panel(
            f"[bold]data dir[/bold]  {settings.data_dir}\n"
            f"[bold]database[/bold]  {settings.database_url}\n"
            f"[bold]timezone[/bold]  {settings.timezone}\n"
            f"[bold]redirect[/bold]  {settings.redirect_uri}",
            title="Circa initialised",
            border_style="green",
        )
    )
    if not settings.google_client_id:
        console.print(
            "\n[yellow]No OAuth client configured.[/yellow] "
            "See [bold]docs/SETUP.md[/bold], then run [bold]circa auth[/bold]."
        )


@app.command()
def auth(
    no_browser: bool = typer.Option(False, "--no-browser", help="Print the URL instead of opening it."),
    only: str = typer.Option(None, "--only", help="Authorise just 'health' or 'calendar'."),
    broad_calendar: bool = typer.Option(
        False,
        "--broad-calendar",
        help="Request full calendar access instead of app-created-calendars-only. "
        "Only needed if Google rejects the narrower scope.",
    ),
) -> None:
    """Authorise Circa against your Google account.

    Runs two consent flows. The Google Health API rejects any token that also
    carries Calendar scopes (403 DISALLOWED_OAUTH_SCOPES), so the two families
    must be authorised separately and their tokens kept apart.
    """
    _bootstrap()
    from circa.ingest.oauth import PURPOSES, AuthError, run_local_auth_flow

    purposes = [only] if only else list(PURPOSES)
    for purpose in purposes:
        if purpose not in PURPOSES:
            console.print(f"[red]Unknown purpose '{purpose}'.[/red] Use health or calendar.")
            raise typer.Exit(1)

    if len(purposes) > 1:
        console.print(
            "[bold]Two authorisations are needed.[/bold] Google's Health API refuses "
            "tokens that also carry Calendar scopes, so they are granted separately.\n"
        )

    for i, purpose in enumerate(purposes, 1):
        console.print(f"[cyan]({i}/{len(purposes)})[/cyan] {purpose}")
        try:
            bundle = run_local_auth_flow(
                purpose, open_browser=not no_browser, broad_calendar=broad_calendar
            )
        except AuthError as exc:
            console.print(f"[red]Authorisation failed for {purpose}:[/red] {exc}")
            raise typer.Exit(1) from exc
        console.print(f"[green]{purpose} connected.[/green] Scopes:")
        for scope in bundle.scopes.split():
            console.print(f"  • {scope}")
        console.print()

    console.print(
        "[yellow]Important:[/yellow] if your OAuth consent screen is still in "
        "'Testing' status, these refresh tokens expire in 7 days.\n"
        "Publish the app to 'In production' in Google Cloud Console. "
        "Run [bold]circa doctor[/bold] to check."
    )


# The modelling stack is an optional dependency group, so a `pip install -e .`
# that omits it produces a collector that fills the database forever and never
# estimates anything. The first symptom was a raw ModuleNotFoundError traceback
# out of a scheduler job, hours after a install that looked successful.
SCIENCE_MODULES = ("numpy", "scipy", "statsmodels", "circadian", "pvlib", "astral")


def missing_science() -> list[str]:
    from importlib.util import find_spec

    missing = []
    for name in SCIENCE_MODULES:
        try:
            if find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


# What `.env.example` and the VM setup script ship with. Shared so the check
# below cannot drift away from the thing it is checking.
EXAMPLE_COORDINATES = (51.5072, -0.1276)


@app.command()
def doctor() -> None:
    """Check configuration, credentials, data freshness and coverage."""
    _bootstrap()
    from sqlalchemy import func, select

    from circa.db.models import (
        DailyMetric,
        HeartRateMinute,
        HrvSample,
        RawDataPoint,
        SleepSession,
        SyncState,
    )
    from circa.db.session import session_scope
    from circa.ingest.oauth import token_status

    settings = get_settings()
    problems: list[str] = []

    table = Table(title="Circa doctor", show_header=False, box=None, padding=(0, 2))

    if settings.google_client_id and settings.google_client_secret:
        table.add_row("[green]ok[/green]", "OAuth client configured")
    else:
        table.add_row("[red]!![/red]", "OAuth client id/secret missing (see docs/SETUP.md)")
        problems.append("oauth-client")

    absent = missing_science()
    if absent:
        # Two rows rather than one long one: the fix has to survive the table's
        # wrapping intact, because it is meant to be copied.
        table.add_row(
            "[red]!![/red]",
            f"Modelling stack not installed ({', '.join(absent)}). "
            "Collection works; nothing will be estimated.",
        )
        # escape(): Rich reads "[science]" as a style tag and silently eats it,
        # which turned the fix line into `uv pip install -e "."` - advice that
        # reproduces the problem it is meant to solve.
        table.add_row("", escape('Fix: uv pip install -e ".[science]"'))
        problems.append("science")
    else:
        table.add_row("[green]ok[/green]", "Modelling stack installed")

    # The light proxy clamps its step->lux estimate to the clear-sky irradiance
    # at your latitude. Deploying with the example coordinates still left in
    # place is silent and wrong: the sun rises at the wrong time and every
    # phase-delaying evening hour is misjudged.
    if (round(settings.latitude, 4), round(settings.longitude, 4)) == EXAMPLE_COORDINATES:
        table.add_row(
            "[yellow]??[/yellow]",
            "Latitude/longitude are still the example values - set "
            "CIRCA_LATITUDE and CIRCA_LONGITUDE to your own city",
        )
        problems.append("coordinates")
    else:
        table.add_row("[green]ok[/green]", f"Location {settings.timezone} for the light proxy")

    status = token_status()
    for purpose, info in status.get("per_purpose", {}).items():
        if info.get("authenticated"):
            age = info.get("refresh_token_age_days")
            label = f"{purpose}: authenticated ({len(info['scopes'])} scopes)"
            if age is not None:
                label += f", token {age:.1f}d old"
            table.add_row("[green]ok[/green]", label)
            if info.get("testing_status_suspected"):
                table.add_row(
                    "[yellow]??[/yellow]",
                    f"{purpose} refresh token is over 6.5 days old. If auth breaks "
                    "around day 7, the consent screen is still in 'Testing'.",
                )
        else:
            table.add_row("[red]!![/red]", f"{purpose}: not authenticated - run `circa auth`")
            problems.append(f"auth:{purpose}")

    with session_scope() as s:
        counts = {
            "raw payloads": s.scalar(select(func.count()).select_from(RawDataPoint)),
            "HR minutes": s.scalar(select(func.count()).select_from(HeartRateMinute)),
            "HRV samples": s.scalar(select(func.count()).select_from(HrvSample)),
            "sleep sessions": s.scalar(select(func.count()).select_from(SleepSession)),
            "daily metrics": s.scalar(select(func.count()).select_from(DailyMetric)),
        }
        nights = s.scalar(
            select(func.count(func.distinct(SleepSession.sleep_date)))
            .select_from(SleepSession)
            .where(SleepSession.is_main_sleep.is_(True))
        ) or 0
        latest_sleep = s.scalar(select(func.max(SleepSession.end_ts)))
        states = list(s.scalars(select(SyncState)))

    table.add_row("", "")
    for label, value in counts.items():
        table.add_row("[dim]--[/dim]", f"{label}: {value:,}")

    tier, tier_name = _confidence_tier(nights)
    table.add_row("", "")
    table.add_row(
        "[cyan]>>[/cyan]",
        f"{nights} main-sleep nights -> confidence tier {tier} ({tier_name})",
    )

    if latest_sleep:
        age_h = (datetime.now(UTC) - latest_sleep).total_seconds() / 3600
        style = "green" if age_h < 36 else "yellow"
        table.add_row(f"[{style}]--[/{style}]", f"Most recent sleep ended {age_h:.1f}h ago")

    failing = [st for st in states if st.consecutive_failures]
    if failing:
        for st in failing:
            table.add_row(
                "[red]!![/red]",
                f"{st.data_type}: {st.consecutive_failures} consecutive failures - {st.last_error}",
            )
        problems.append("sync")

    console.print(table)
    if problems:
        console.print(f"\n[yellow]{len(problems)} issue(s) need attention.[/yellow]")
        raise typer.Exit(1)
    console.print("\n[green]All checks passed.[/green]")


def _confidence_tier(nights: int) -> tuple[int, str]:
    if nights < 7:
        return 0, "cold start"
    if nights < 14:
        return 1, "provisional"
    if nights < 30:
        return 2, "personalized"
    return 3, "mature"


@app.command()
def probe(
    out: Path = typer.Option(
        None, "--out", "-o", help="Write full sample payloads to this JSON file."
    ),
) -> None:
    """Fetch one small page of every data type and report its real shape.

    Google's v4 schemas are still evolving and its docs contradict themselves in
    places, so this is how we verify assumptions against the live API instead of
    trusting documentation. Run it before relying on any normaliser.
    """
    _bootstrap()
    from circa.ingest.datatypes import ordered
    from circa.ingest.health_api import HealthApiError, HealthClient

    results: dict[str, dict] = {}
    table = Table(title="Google Health API probe")
    table.add_column("data type")
    table.add_column("status")
    table.add_column("pts", justify="right")
    table.add_column("filter")
    table.add_column("top-level keys")

    now = datetime.now(UTC)
    with HealthClient() as client:
        for dt in ordered():
            entry: dict = {"kind": dt.kind.value, "filter_field": dt.time_filter}
            try:
                page = client.sample_page(dt, page_size=3)
                points = client._extract_points(page)
                entry["sample"] = points[:2]
                entry["envelope_keys"] = sorted(page.keys())

                # Does the documented filter expression actually work?
                filter_ok = "n/a"
                if dt.time_filter:
                    result = client.fetch(dt, now - timedelta(days=2), now, page_size=5, max_pages=1)
                    filter_ok = "[green]ok[/green]" if result.server_filtered else "[yellow]rejected[/yellow]"
                    entry["server_filtered"] = result.server_filtered
                    entry["filter_error"] = result.filter_error

                keys = sorted(points[0].keys()) if points else []
                entry["point_keys"] = keys
                results[dt.name] = entry
                table.add_row(
                    dt.name,
                    "[green]200[/green]",
                    str(len(points)),
                    filter_ok,
                    ", ".join(keys[:4]) or "[dim]empty[/dim]",
                )
            except HealthApiError as exc:
                entry["error"] = f"{exc.status}: {exc}"
                results[dt.name] = entry
                table.add_row(dt.name, f"[red]{exc.status}[/red]", "-", "-", str(exc)[:50])
            except Exception as exc:  # noqa: BLE001
                entry["error"] = str(exc)
                results[dt.name] = entry
                table.add_row(dt.name, "[red]err[/red]", "-", "-", str(exc)[:50])

    console.print(table)
    if out:
        out.write_text(json.dumps(results, indent=2, default=str))
        console.print(f"\nFull payloads written to [bold]{out}[/bold]")
    else:
        console.print(
            "\n[dim]Re-run with --out probe.json to capture full payloads "
            "for correcting the parsers.[/dim]"
        )


@app.command()
def sync(
    force: bool = typer.Option(False, "--force", help="Ignore the device lastSyncTime gate."),
    only: list[str] = typer.Option(None, "--only", help="Restrict to these data types."),
    since: str = typer.Option(None, "--since", help="Backfill from this date (YYYY-MM-DD)."),
) -> None:
    """Run one collection pass."""
    _bootstrap()
    from circa.ingest.poller import sync_once

    since_dt = None
    if since:
        since_dt = datetime.fromisoformat(since).replace(tzinfo=UTC)

    report = sync_once(force=force, only=list(only) if only else None, since=since_dt)

    table = Table(title="Sync report")
    for col, justify in [
        ("data type", "left"), ("fetched", "right"), ("new", "right"),
        ("dupes", "right"), ("normalized", "right"), ("filter", "left"),
    ]:
        table.add_column(col, justify=justify)

    for t in report.types:
        table.add_row(
            t.data_type,
            str(t.fetched),
            f"[green]{t.inserted}[/green]" if t.inserted else "0",
            str(t.duplicates),
            str(t.normalized),
            "server" if t.server_filtered else "[yellow]client[/yellow]",
        )
    console.print(table)

    if report.skipped_high_frequency:
        console.print("[dim]High-frequency types skipped: no new device sync.[/dim]")
    if report.device_synced_at:
        age = (datetime.now(UTC) - report.device_synced_at).total_seconds() / 60
        console.print(f"[dim]Watch last synced {age:.0f} min ago.[/dim]")
    for err in report.errors:
        console.print(f"[red]{err}[/red]")


@app.command()
def status() -> None:
    """Show per-data-type sync state."""
    _bootstrap()
    from sqlalchemy import select

    from circa.db.models import SyncState
    from circa.db.session import session_scope

    with session_scope() as s:
        states = list(s.scalars(select(SyncState).order_by(SyncState.data_type)))

    if not states:
        console.print("[yellow]No sync has run yet. Try `circa sync`.[/yellow]")
        return

    # The user's own clock, matching the Details page. Printing UTC here and
    # local there is a five-hour disagreement about the same row.
    tz = ZoneInfo(get_settings().timezone)

    def local(ts: datetime | None) -> str:
        if ts is None:
            return "-"
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return stamp(ts.astimezone(tz))

    table = Table(title="Sync state")
    for col in ("data type", "watermark", "last success", "ingested", "fails"):
        table.add_column(col)
    for st in states:
        table.add_row(
            st.data_type,
            local(st.watermark),
            local(st.last_success_at),
            f"{st.points_ingested or 0:,}",
            f"[red]{st.consecutive_failures}[/red]" if st.consecutive_failures else "0",
        )
    console.print(table)


@app.command()
def renormalize(
    only: list[str] = typer.Option(None, "--only", help="Restrict to these data types."),
) -> None:
    """Re-derive every normalised table from the stored raw payloads.

    Raw payloads are kept precisely so a parser can be corrected and history
    re-derived. Also repairs `point_time`, which earlier parser versions may
    have failed to extract.
    """
    _bootstrap()
    from sqlalchemy import select

    from circa.db.models import RawDataPoint
    from circa.db.session import session_scope
    from circa.ingest.datatypes import BY_NAME, ordered
    from circa.ingest.parsing import extract_point_time
    from circa.normalize.dispatch import normalize_type

    names = list(only) if only else [d.name for d in ordered()]

    table = Table(title="Re-normalisation")
    for col in ("data type", "raw rows", "timestamps repaired", "normalized"):
        table.add_column(col, justify="right" if col != "data type" else "left")

    for name in names:
        dt = BY_NAME.get(name)
        if dt is None:
            continue
        with session_scope() as s:
            rows = list(
                s.scalars(
                    select(RawDataPoint)
                    .where(RawDataPoint.data_type == name)
                    .order_by(RawDataPoint.id)
                )
            )
            repaired = 0
            for row in rows:
                if row.point_time is None:
                    ts = extract_point_time(row.payload or {}, dt.payload_key)
                    if ts is not None:
                        row.point_time = ts
                        repaired += 1
            s.flush()
            written = normalize_type(s, name, rows) if rows else 0
        if rows:
            table.add_row(
                name, f"{len(rows):,}",
                f"[green]{repaired}[/green]" if repaired else "0",
                f"{written:,}",
            )

    console.print(table)


@app.command()
def retention(vacuum: bool = typer.Option(False, "--vacuum", help="Reclaim disk after pruning.")) -> None:
    """Prune aged-out raw payloads and 5-second HR samples."""
    _bootstrap()
    from circa.ingest.retention import run_retention

    report = run_retention(vacuum=vacuum)
    before = (report.bytes_before or 0) / 1e6
    after = (report.bytes_after or 0) / 1e6
    console.print(
        f"Removed [bold]{report.raw_deleted:,}[/bold] raw payloads and "
        f"[bold]{report.hr_samples_deleted:,}[/bold] HR samples.\n"
        f"Database: {before:.1f} MB -> {after:.1f} MB"
    )


@app.command()
def run(
    no_calendar: bool = typer.Option(False, "--no-calendar", help="Compute but do not touch Google Calendar."),
) -> None:
    """Estimate phase, build the alertness curve, and push calendar blocks."""
    _bootstrap()
    from datetime import timedelta

    from circa.db.session import session_scope
    from circa.pipeline import run as run_pipeline

    with session_scope() as s:
        report = run_pipeline(s, push_calendar=not no_calendar)

    if report.skipped_reason:
        console.print(f"[yellow]Skipped:[/yellow] {report.skipped_reason}")
        raise typer.Exit(0)
    if report.phase is None:
        for err in report.errors:
            console.print(f"[red]{err}[/red]")
        raise typer.Exit(1)

    p = report.phase
    off = p.utc_offset_seconds

    def local(ts):
        return clock_from(ts + timedelta(seconds=off))

    console.print(
        Panel(
            f"[bold]DLMO[/bold]      {local(p.dlmo_ts)}   "
            f"80% CI {local(p.ci80[0])}-{local(p.ci80[1])}  (±{p.sd_minutes:.0f} min)\n"
            f"[bold]CBTmin[/bold]    {local(p.cbtmin_ts)}\n"
            f"[bold]Tier[/bold]      {p.confidence.tier} ({p.confidence.tier_name})  "
            f"quality {p.confidence.q}\n"
            f"[bold]tau[/bold]       {p.posterior.tau_mean:.2f} ± {p.posterior.tau_sd:.2f} h"
            + (f"\n[yellow]flags[/yellow]     {', '.join(p.confidence.ood_flags)}"
               if p.confidence.ood_flags else ""),
            title="Circadian estimate",
            border_style="cyan",
        )
    )

    if report.blocks:
        table = Table(title=f"{len(report.blocks)} blocks")
        for col in ("calendar", "when", "block"):
            table.add_column(col)
        for b in report.blocks:
            start = b.start + timedelta(seconds=off)
            end = b.end + timedelta(seconds=off)
            table.add_row(
                b.category,
                f"{start:%a} {clock_from(start)}-{clock_from(end)}",
                b.title,
            )
        console.print(table)
    else:
        console.print("[yellow]No blocks generated at this confidence level.[/yellow]")

    if report.calendar:
        c = report.calendar
        console.print(
            f"Calendar: [green]{c.created} created[/green], {c.updated} updated, "
            f"{c.unchanged} unchanged, {c.deleted} removed"
        )
        for err in c.errors:
            console.print(f"[red]{err}[/red]")
    for err in report.errors:
        console.print(f"[yellow]{err}[/yellow]")


@app.command()
def backtest(
    max_days: int = typer.Option(14, help="Held-out days to evaluate."),
) -> None:
    """Run the held-out-day ablation ladder."""
    _bootstrap()
    from circa.db.session import session_scope
    from circa.validate import backtest as bt

    with session_scope() as s:
        report = bt.run(s, max_days=max_days)

    if not report.ablations:
        for note in report.notes:
            console.print(f"[yellow]{note}[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Ablation ladder ({report.n_days} held-out days)")
    for col in ("model", "adds", "MAE", "P60", "CI80 width", "coverage"):
        table.add_column(col)
    for a in report.ablations:
        if a.metrics is None:
            table.add_row(a.name, a.description, f"[red]{a.error}[/red]", "", "", "")
            continue
        m = a.metrics.as_dict()
        table.add_row(
            a.name, a.description,
            f"{m['mae_minutes']:.0f} min",
            f"{m['p60']:.0%}",
            f"{m.get('mean_ci80_width_minutes', 0):.0f} min",
            f"{m['coverage_80']:.0%}" if m["coverage_80"] is not None else "-",
        )
    console.print(table)

    if report.probe_consistency:
        console.print(f"\n[bold]Passive probe check[/bold]: {report.probe_consistency}")
    for note in report.notes:
        console.print(f"[dim]{note}[/dim]")


@app.command()
def calendars(
    delete: bool = typer.Option(False, "--delete", help="Remove Circa's calendars entirely."),
) -> None:
    """List (or delete) the Google Calendars Circa manages."""
    _bootstrap()
    from sqlalchemy import select

    from circa.db.models import CalendarLink
    from circa.db.session import session_scope
    from circa.gcal.client import CalendarClient
    from circa.settings_store import load_settings

    with session_scope() as s:
        settings = load_settings(s)
        if delete:
            if not typer.confirm(
                "Delete every Circa calendar and all of its events from your Google "
                "account? Your own calendars and events are untouched."
            ):
                raise typer.Exit(0)
            with CalendarClient() as client:
                for link in s.scalars(select(CalendarLink)):
                    if not link.calendar_id:
                        continue
                    try:
                        client.delete_calendar(link.calendar_id)
                        console.print(f"[green]deleted[/green] {link.summary}")
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"[red]{link.summary}: {exc}[/red]")
                    link.calendar_id = None
            raise typer.Exit(0)

        with CalendarClient() as client:
            from circa.gcal.sync import provision

            provision(s, client, settings)

        table = Table(title="Circa calendars")
        for col in ("category", "calendar", "enabled", "id"):
            table.add_column(col)
        for link in s.scalars(select(CalendarLink)):
            table.add_row(
                link.category, link.summary,
                "[green]yes[/green]" if link.enabled else "[dim]no[/dim]",
                (link.calendar_id or "-")[:42],
            )
        console.print(table)


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind address (defaults to CIRCA_WEB_HOST)."),
    port: int = typer.Option(None, help="Port (defaults to CIRCA_WEB_PORT)."),
    no_scheduler: bool = typer.Option(False, "--no-scheduler", help="Web only, no polling."),
) -> None:
    """Run the web app and the background poller."""
    _bootstrap()
    import uvicorn

    settings = get_settings()
    from circa.web.app import create_app

    application = create_app(with_scheduler=not no_scheduler)
    uvicorn.run(
        application,
        host=host or settings.web_host,
        port=port or settings.web_port,
        log_config=None,
    )


if __name__ == "__main__":
    app()
