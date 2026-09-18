"""Fernet-backed storage for repository credentials."""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken

MASTER_KEY_VARIABLE = "GRAPHRAG_ENCRYPTION_MASTER_KEY"
MISSING_KEY = (
    "GRAPHRAG_ENCRYPTION_MASTER_KEY is not set; refusing to encrypt/decrypt secrets."
)
INVALID_KEY = (
    "GRAPHRAG_ENCRYPTION_MASTER_KEY is invalid — expected a urlsafe-base64 "
    "32-byte Fernet key."
)
DECRYPT_FAILED = "could not decrypt token (wrong key or corrupt data)."


class VaultError(RuntimeError):
    """A vault configuration or cryptographic failure safe to report."""


def is_configured() -> bool:
    try:
        _fernet()
    except VaultError:
        return False
    return True


def _fernet() -> Fernet:
    raw = os.environ.get(MASTER_KEY_VARIABLE, "").strip()
    if not raw:
        raise VaultError(MISSING_KEY)
    try:
        return Fernet(raw.encode("ascii"))
    except (ValueError, TypeError, UnicodeEncodeError) as exc:
        raise VaultError(INVALID_KEY) from exc


def encrypt(plain_text: str) -> str:
    if not isinstance(plain_text, str) or not plain_text:
        raise VaultError("token must be a non-empty string.")
    return _fernet().encrypt(plain_text.encode("utf-8")).decode("ascii")


def decrypt(cipher_text: str) -> str:
    if not isinstance(cipher_text, str) or not cipher_text:
        raise VaultError("token must be a non-empty string.")
    try:
        return _fernet().decrypt(cipher_text.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeEncodeError, UnicodeDecodeError) as exc:
        raise VaultError(DECRYPT_FAILED) from exc
