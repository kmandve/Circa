"""OAuth scope separation.

Regression guard for a bug that cost a full auth round-trip to diagnose: the
Google Health API returns

    403 PERMISSION_DENIED  DISALLOWED_OAUTH_SCOPES
    disallowed_scopes: cl_app_created,cl_readonly

when the access token *also* carries Calendar scopes. Not a missing scope - a
disallowed one. Health and Calendar therefore need separate tokens, and the
auth URL must not use incremental authorisation, which would silently merge
them back together.
"""

from __future__ import annotations

import urllib.parse

import pytest

from circa.ingest.oauth import (
    CALENDAR_SCOPES,
    HEALTH_SCOPES,
    PURPOSES,
    SCOPE_SETS,
    build_auth_url,
    get_access_token,
    save_tokens,
    token_status,
)


def _scopes_in(url: str) -> set[str]:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return set(query["scope"][0].split())


def test_health_and_calendar_scope_sets_are_disjoint():
    """The whole point: neither token may carry the other's scopes."""
    assert set(HEALTH_SCOPES).isdisjoint(CALENDAR_SCOPES)
    assert all("googlehealth" in s for s in HEALTH_SCOPES)
    assert all("calendar" in s for s in CALENDAR_SCOPES)


def test_health_auth_url_requests_no_calendar_scopes():
    url = build_auth_url("state", "challenge", SCOPE_SETS["health"])
    scopes = _scopes_in(url)
    assert scopes == set(HEALTH_SCOPES)
    assert not any("calendar" in s for s in scopes)


def test_calendar_auth_url_requests_no_health_scopes():
    url = build_auth_url("state", "challenge", SCOPE_SETS["calendar"])
    scopes = _scopes_in(url)
    assert scopes == set(CALENDAR_SCOPES)
    assert not any("googlehealth" in s for s in scopes)


def test_auth_url_does_not_use_incremental_authorisation():
    """`include_granted_scopes=true` merges prior scopes onto the new token.

    That single parameter is enough to reintroduce the bug even with correctly
    separated scope sets, because the second flow's token would inherit the
    first flow's scopes.
    """
    url = build_auth_url("state", "challenge", SCOPE_SETS["health"])
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert "include_granted_scopes" not in query
    # Offline access with forced consent is still required for a refresh token.
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]


def test_tokens_are_stored_and_retrieved_per_purpose():
    from datetime import UTC, datetime, timedelta

    from circa.ingest.oauth import TokenBundle

    expires = datetime.now(UTC) + timedelta(hours=1)
    save_tokens(
        TokenBundle("health-access", "health-refresh", expires, " ".join(HEALTH_SCOPES)),
        "health",
    )
    save_tokens(
        TokenBundle("cal-access", "cal-refresh", expires, " ".join(CALENDAR_SCOPES)),
        "calendar",
    )

    assert get_access_token("health") == "health-access"
    assert get_access_token("calendar") == "cal-access"


def test_unknown_purpose_is_rejected():
    with pytest.raises(ValueError, match="unknown token purpose"):
        get_access_token("nonsense")


def test_combined_status_requires_both_tokens():
    """Collection needs Health; publishing needs Calendar. Half-authorised is not authorised."""
    from datetime import UTC, datetime, timedelta

    from circa.ingest.oauth import TokenBundle

    expires = datetime.now(UTC) + timedelta(hours=1)
    assert token_status()["authenticated"] is False

    save_tokens(TokenBundle("a", "r", expires, " ".join(HEALTH_SCOPES)), "health")
    status = token_status()
    assert status["authenticated"] is False
    assert status["missing"] == ["calendar"]

    save_tokens(TokenBundle("b", "r2", expires, " ".join(CALENDAR_SCOPES)), "calendar")
    status = token_status()
    assert status["authenticated"] is True
    assert status["missing"] == []


def test_every_purpose_has_a_scope_set():
    assert set(SCOPE_SETS) == set(PURPOSES)
