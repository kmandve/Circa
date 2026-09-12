"""Configuration for Circa.

Settings come from environment variables (prefix ``CIRCA_``) or a ``.env`` file.
Anything secret (OAuth client secret, encryption key) should come from the
environment in production; on a dev machine we fall back to files in the data
directory with 0600 permissions.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    """`/var/lib/circa` when it exists and is writable (the VM), else `~/.circa`."""
    system = Path("/var/lib/circa")
    if system.is_dir() and os.access(system, os.W_OK):
        return system
    return Path.home() / ".circa"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CIRCA_",
        # `.env` for a checkout you are working in; `/etc/circa/circa.env` for a
        # deployed one. systemd reads the latter itself, but nothing else did -
        # so `circa auth` and `circa doctor` run by hand on the VM saw no client
        # id, no timezone and no coordinates, and reported a correctly
        # configured machine as broken. Later entries win, and a missing file is
        # simply skipped.
        env_file=("/etc/circa/circa.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage -----------------------------------------------------------
    data_dir: Path = Field(default_factory=_default_data_dir)
    db_url: str | None = None

    # --- Google OAuth ------------------------------------------------------
    # Created in Google Cloud Console -> APIs & Services -> Credentials.
    # The consent screen MUST be published to "In production" or refresh tokens
    # silently expire after 7 days. See docs/SETUP.md.
    google_client_id: str = ""
    google_client_secret: str = ""
    oauth_host: str = "localhost"
    oauth_port: int = 8721

    # --- secrets -----------------------------------------------------------
    # Fernet key used to encrypt the stored refresh token at rest.
    secret_key: str = ""

    # --- location (a static setting, not location tracking) ----------------
    # Used only for solar elevation / clear-sky irradiance in the light proxy.
    # These are placeholders, not a sensible default: `circa doctor` says so
    # while they are still in place, because a light ceiling computed for the
    # wrong meridian is wrong quietly.
    timezone: str = "Europe/London"
    latitude: float = 51.5072
    longitude: float = -0.1276

    # --- polling -----------------------------------------------------------
    poll_interval_minutes: int = 15
    # How far back an initial (empty-database) sync reaches.
    initial_backfill_days: int = 90
    # Never ask the API for more than this in one request window.
    max_fetch_window_days: int = 7

    # --- retention (keeps the whole database around ~50 MB/year) -----------
    raw_payload_retention_days: int = 30
    hr_sample_retention_days: int = 90

    # --- web ---------------------------------------------------------------
    web_host: str = "127.0.0.1"
    web_port: int = 8720

    log_level: str = "INFO"

    @field_validator("data_dir", mode="after")
    @classmethod
    def _ensure_data_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    # --- derived -----------------------------------------------------------
    @property
    def database_url(self) -> str:
        if self.db_url:
            return self.db_url
        return f"sqlite+pysqlite:///{self.data_dir / 'circa.db'}"

    @property
    def redirect_uri(self) -> str:
        return f"http://{self.oauth_host}:{self.oauth_port}/oauth/callback"

    def fernet_key(self) -> bytes:
        """Return the at-rest encryption key, generating one on first use.

        Environment always wins. The generated fallback lives in the data dir
        with 0600 permissions so a dev machine works with no setup, but on the
        VM you should set CIRCA_SECRET_KEY so the key is not next to the DB.
        """
        if self.secret_key:
            return self.secret_key.encode()

        keyfile = self.data_dir / "secret.key"
        if keyfile.exists():
            return keyfile.read_bytes().strip()

        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        keyfile.write_bytes(key)
        keyfile.chmod(0o600)
        return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
