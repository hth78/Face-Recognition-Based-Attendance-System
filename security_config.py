"""Utilities for loading secrets without hardcoding credentials."""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

SecretProvider = Callable[[str], Optional[str]]


def _read_secret(name: str, provider: Optional[SecretProvider] = None) -> str:
    """Read a secret from env vars or an optional provider and fail fast if missing."""
    value: Optional[str] = None
    if provider is not None:
        value = provider(name)
    if not value:
        value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required secret '{name}'. Set it as an environment variable "
            "or provide a secret provider callback."
        )
    return value


def load_email_credentials(provider: Optional[SecretProvider] = None) -> Tuple[str, str]:
    """Load SMTP/IMAP credentials."""
    username = _read_secret("FRAS_EMAIL_USERNAME", provider)
    password = _read_secret("FRAS_EMAIL_PASSWORD", provider)
    return username, password


def load_admin_credentials(provider: Optional[SecretProvider] = None) -> Tuple[str, str]:
    """Load admin UI login credentials."""
    username = _read_secret("FRAS_ADMIN_USERNAME", provider)
    password = _read_secret("FRAS_ADMIN_PASSWORD", provider)
    return username, password


def load_mail_hosts() -> Tuple[str, int, str, int]:
    """Load host settings with safe defaults for Gmail."""
    imap_host = os.getenv("FRAS_IMAP_HOST", "imap.gmail.com")
    imap_port = int(os.getenv("FRAS_IMAP_PORT", "993"))
    smtp_host = os.getenv("FRAS_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("FRAS_SMTP_PORT", "587"))
    return imap_host, imap_port, smtp_host, smtp_port
