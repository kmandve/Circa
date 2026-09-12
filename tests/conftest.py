from __future__ import annotations

from datetime import UTC, datetime

import pytest


@pytest.fixture(autouse=True)
def temp_env(tmp_path, monkeypatch):
    """Give every test its own data dir, database and encryption key."""
    monkeypatch.setenv("CIRCA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CIRCA_TIMEZONE", "America/Chicago")
    monkeypatch.setenv("CIRCA_GOOGLE_CLIENT_ID", "test-client")
    monkeypatch.setenv("CIRCA_GOOGLE_CLIENT_SECRET", "test-secret")
    monkeypatch.delenv("CIRCA_DB_URL", raising=False)

    from circa.config import get_settings
    from circa.db import session as db_session

    get_settings.cache_clear()
    db_session.reset_engine()
    db_session.init_db()
    yield

    # A page load with no stored forecast kicks off a background recompute. Let
    # it finish before the temporary database is torn down underneath it -
    # otherwise it surfaces later as a stray "no such table" from a thread that
    # belongs to a test that has already ended.
    from circa.web import app as web_app

    thread = getattr(web_app, "_REFRESH_THREAD", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=30)
    web_app._REFRESH_THREAD = None

    db_session.reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def db():
    from circa.db.session import session_scope

    return session_scope


def utc(y, m, d, hh=0, mm=0, ss=0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=UTC)


def make_raw(data_type: str, payload: dict, point_time: datetime | None = None):
    """Build a RawDataPoint without going through the API."""
    from circa.db.models import RawDataPoint
    from circa.ingest.store import content_hash

    return RawDataPoint(
        data_type=data_type,
        source="test",
        point_time=point_time,
        fetched_at=datetime.now(UTC),
        content_hash=content_hash(payload),
        payload=payload,
    )


def sleep_payload(
    start: datetime,
    end: datetime,
    stages: list[tuple[datetime, datetime, str]] | None = None,
    external_id: str = "sleep/1",
) -> dict:
    """Mimic the v4 sleep session shape."""
    body: dict = {
        "name": external_id,
        "interval": {
            "startTime": start.isoformat().replace("+00:00", "Z"),
            "endTime": end.isoformat().replace("+00:00", "Z"),
        },
        "type": "STAGES",
    }
    if stages:
        body["stages"] = [
            {
                "interval": {
                    "startTime": s.isoformat().replace("+00:00", "Z"),
                    "endTime": e.isoformat().replace("+00:00", "Z"),
                },
                "type": name,
            }
            for s, e, name in stages
        ]
    return {"sleep": body}


def hr_payload(ts: datetime, bpm: int, context: str = "SEDENTARY") -> dict:
    return {
        "heartRate": {
            "sampleTime": ts.isoformat().replace("+00:00", "Z"),
            "beatsPerMinute": bpm,
            "metadata": {"motionContext": context},
        }
    }


def steps_payload(start: datetime, end: datetime, count: int) -> dict:
    return {
        "steps": {
            "interval": {
                "startTime": start.isoformat().replace("+00:00", "Z"),
                "endTime": end.isoformat().replace("+00:00", "Z"),
            },
            "count": count,
        }
    }
