"""Concurrent access to a single SQLite file.

The web app, the scheduler and a manual CLI run all touch the same database,
and a long write transaction held across a network call has already caused
"database is locked" in production. These tests reproduce the shape of that
contention rather than trusting that it was fixed.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from circa.db.models import HeartRateMinute, Setting, SyncState
from circa.db.session import session_scope


def _write_hr(offset_minutes: int, n: int = 40) -> None:
    base = datetime.now(UTC) - timedelta(minutes=offset_minutes * 1000)
    with session_scope() as s:
        for i in range(n):
            s.add(HeartRateMinute(
                ts=base + timedelta(minutes=i), bpm_median=60.0,
                bpm_min=55, bpm_max=70, n_samples=12, active_fraction=0.0,
            ))


def _run_threads(targets, timeout=30):
    errors: list[BaseException] = []

    def wrap(fn):
        def inner():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        return inner

    threads = [threading.Thread(target=wrap(t)) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
        assert not t.is_alive(), "a thread deadlocked on the database"
    return errors


def test_concurrent_writers_do_not_hit_database_is_locked(db):
    errors = _run_threads([lambda i=i: _write_hr(i) for i in range(8)])
    locked = [e for e in errors if "database is locked" in str(e).lower()]
    assert not locked, f"{len(locked)} writers hit a lock: {locked[:2]}"
    assert not errors, errors[:3]

    with db() as s:
        assert s.scalar(select(func.count()).select_from(HeartRateMinute)) == 8 * 40


def test_readers_are_not_blocked_by_a_writer(db):
    """WAL exists so a long write does not stall the web app."""
    started = threading.Event()
    finish = threading.Event()
    read_ok = threading.Event()

    def slow_writer():
        with session_scope() as s:
            s.add(Setting(key="slow", value={"n": 1}))
            s.flush()
            started.set()
            finish.wait(10)

    def reader():
        assert started.wait(10)
        with session_scope() as s:
            s.scalar(select(func.count()).select_from(HeartRateMinute))
        read_ok.set()
        finish.set()

    errors = _run_threads([slow_writer, reader])
    finish.set()
    assert not errors, errors
    assert read_ok.is_set(), "a reader was blocked behind an open write"


def test_a_failed_write_rolls_back_completely(db):
    """A half-applied poll would corrupt the watermark's meaning."""
    class Boom(Exception):
        pass

    with pytest.raises(Boom), session_scope() as s:
        s.add(SyncState(data_type="heart-rate", watermark=datetime.now(UTC)))
        s.flush()
        raise Boom()

    with db() as s:
        assert s.get(SyncState, "heart-rate") is None


def test_engine_reset_under_load_does_not_corrupt(db):
    """The web app rebuilds the engine on config reload."""
    from circa.db.session import init_db, reset_engine

    _write_hr(0, 20)
    reset_engine()
    init_db()
    _write_hr(1, 20)
    with db() as s:
        assert s.scalar(select(func.count()).select_from(HeartRateMinute)) == 40


def test_a_full_day_of_5_second_heart_rate_normalises(db):
    """SQLite caps bound parameters at 32,766 per statement.

    A day of 5-second heart rate is ~17,000 samples, so any insert path that
    binds the whole batch at once raises "too many SQL variables" - which is
    exactly what happened the first time a real day arrived.
    """
    from circa.db.models import HeartRateSample, RawDataPoint
    from circa.normalize.dispatch import normalize_type
    from tests.conftest import hr_payload, make_raw

    base = datetime.now(UTC) - timedelta(days=1)
    with db() as s:
        for i in range(17280):  # 24 h at one sample every 5 s
            ts = base + timedelta(seconds=5 * i)
            s.add(make_raw("heart-rate", hr_payload(ts, 60 + i % 20), point_time=ts))

    with db() as s:
        raws = list(s.scalars(select(RawDataPoint)))
        normalize_type(s, "heart-rate", raws)

    with db() as s:
        assert s.scalar(select(func.count()).select_from(HeartRateSample)) == 17280


# --- schema readiness -------------------------------------------------------


def test_a_database_missing_a_new_column_is_migrated_on_first_use(tmp_path, monkeypatch):
    """Schema readiness must not depend on remembering to call init_db.

    `create_app()` under uvicorn, a script, or a test builds sessions without
    going through the CLI's bootstrap. Against a database created before a model
    gained a column, every request then failed with "no such column" - which is
    exactly what happened to the real database when `notify_minutes` was added.
    """
    import sqlite3

    from circa.db.session import get_engine, init_db, reset_engine

    init_db()
    reset_engine()

    # Simulate a database that predates the column.
    db_path = tmp_path / "circa.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE calendar_block DROP COLUMN notify_minutes")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(calendar_block)")}
    assert "notify_minutes" not in cols

    # Touching the engine at all must repair it - no init_db call.
    from sqlalchemy import inspect

    reset_engine()
    cols = {c["name"] for c in inspect(get_engine()).get_columns("calendar_block")}
    assert "notify_minutes" in cols


def test_every_model_column_exists_in_a_fresh_database(db):
    """Catches a column the auto-migration cannot add (NOT NULL, no default)."""
    from sqlalchemy import inspect

    from circa.db.models import Base
    from circa.db.session import get_engine

    inspector = inspect(get_engine())
    tables = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        assert table.name in tables, f"table {table.name} was never created"
        present = {c["name"] for c in inspector.get_columns(table.name)}
        missing = {c.name for c in table.columns} - present
        assert not missing, f"{table.name} is missing {sorted(missing)}"
