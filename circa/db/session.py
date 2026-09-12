"""Engine and session management.

SQLite is the right call for a single-user, single-writer service: no hosted
database to pay for, no network hop, and the whole dataset is ~50 MB/year once
retention is applied. WAL mode lets the web app read while the poller writes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import structlog
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from circa.config import get_settings
from circa.db.models import Base

log = structlog.get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _configure_sqlite(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    # Concurrent reader (web app) alongside the writer (poller).
    cur.execute("PRAGMA journal_mode=WAL")
    # Durable enough for this workload; a lost final transaction just re-syncs.
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    # Wait rather than immediately raising "database is locked".
    cur.execute("PRAGMA busy_timeout=10000")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.close()


def get_engine() -> Engine:
    """The process-wide engine, with its schema brought up to date once.

    The schema check runs here rather than only in `init_db` so that it cannot
    be skipped. Any entry point that builds a session without going through the
    CLI - `create_app()` under uvicorn, a script, a test - would otherwise query
    a column the models declare and the database does not have, and fail with
    "no such column" at request time rather than at startup.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        engine = create_engine(url, future=True, pool_pre_ping=True)
        if url.startswith("sqlite"):
            event.listen(engine, "connect", _configure_sqlite)
        # Assign before migrating: _add_missing_columns opens its own
        # connection, and leaving _engine unset would build a second engine.
        _engine = engine
        _ensure_schema(engine)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Explicitly prepare the schema. `get_engine` already does this."""
    _ensure_schema(get_engine())


def _ensure_schema(engine) -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine) -> None:
    """Add columns present on the models but missing from the live database.

    `create_all` creates missing tables but never alters existing ones, so a
    model that gains a column would raise `no such column` against any database
    created before it - which is every database that already holds real data.
    There is no Alembic history here, and SQLite's ADD COLUMN is cheap, so the
    additive case is handled directly. Renames, drops and type changes are not,
    and still need a real migration.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                if not column.nullable and column.default is None:
                    log.warning(
                        "db.column_needs_migration",
                        table=table.name, column=column.name,
                        reason="not nullable and has no default",
                    )
                    continue
                ddl = column.type.compile(engine.dialect)
                conn.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl}')
                )
                log.info("db.column_added", table=table.name, column=column.name)


def reset_engine() -> None:
    """Drop cached engine/factory — used by tests that swap the database URL."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
