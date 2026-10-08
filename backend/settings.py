from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppSettings:
    database_url: str = os.getenv(
        "ARIA_DATABASE_URL",
        "postgresql+psycopg://aria:aria@127.0.0.1:55432/aria",
    )
    migration_database_url: str = os.getenv(
        "ARIA_MIGRATION_DATABASE_URL",
        "postgresql+psycopg://aria:aria@127.0.0.1:55432/aria",
    )
    public_origin: str = os.getenv("ARIA_PUBLIC_ORIGIN", "http://localhost:5173")
    api_origin: str = os.getenv("ARIA_API_ORIGIN", "http://localhost:8000")
    google_client_id: str = os.getenv("ARIA_GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("ARIA_GOOGLE_CLIENT_SECRET", "")
    google_redirect_uri: str = os.getenv(
        "ARIA_GOOGLE_REDIRECT_URI",
        "http://localhost:8000/api/auth/google/callback",
    )
    jwt_private_key_path: Path = ROOT / os.getenv(
        "ARIA_JWT_PRIVATE_KEY_PATH", ".aria-private/jwt-private.pem"
    )
    jwt_public_key_path: Path = ROOT / os.getenv(
        "ARIA_JWT_PUBLIC_KEY_PATH", ".aria-private/jwt-public.pem"
    )
    oauth_encryption_key: str = os.getenv("ARIA_OAUTH_ENCRYPTION_KEY", "")
    storage_root: Path = ROOT / os.getenv("ARIA_STORAGE_ROOT", "data/private_documents")
    deletion_ledger_path: Path = ROOT / "data/deletion-ledger.jsonl"
    cookie_secure: bool = _bool("ARIA_COOKIE_SECURE", False)
    consent_version: str = os.getenv("ARIA_CONSENT_VERSION", "aria-privacy-v1")
    jwt_issuer: str = os.getenv("ARIA_JWT_ISSUER", "aria-api")
    jwt_audience: str = os.getenv("ARIA_JWT_AUDIENCE", "aria-web")
    access_ttl_seconds: int = 600
    auth_session_ttl_seconds: int = 7 * 24 * 60 * 60
    oauth_ttl_seconds: int = 600
    max_pdf_bytes: int = 10 * 1024 * 1024
    max_pdf_pages: int = 50
    max_extracted_characters: int = 200_000
    max_answer_characters: int = 20_000
    max_audio_bytes: int = 20 * 1024 * 1024
    connection_lease_seconds: int = 45

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.public_origin, self.api_origin)))


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
