from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import secrets
import uuid
from urllib.parse import urlencode

from cryptography.fernet import Fernet, InvalidToken
import httpx
import jwt
from fastapi import Depends, HTTPException, Request, Response, status
from jwt import PyJWKClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db
from backend.models import (
    AuthSession,
    OAuthIdentity,
    OAuthTransaction,
    RefreshToken,
    User,
    utcnow,
)
from backend.settings import AppSettings, get_settings


ACCESS_COOKIE = "aria_access"
REFRESH_COOKIE = "aria_refresh"
OAUTH_BINDING_COOKIE = "aria_oauth_binding"
GOOGLE_ISSUER = "https://accounts.google.com"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"


def digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def token_urlsafe() -> str:
    return secrets.token_urlsafe(48)


def request_hash(*values: str) -> str:
    joined = "\x1f".join(values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _fernet(settings: AppSettings) -> Fernet:
    if not settings.oauth_encryption_key:
        raise HTTPException(503, "OAuth encryption is not configured")
    try:
        return Fernet(settings.oauth_encryption_key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise HTTPException(503, "OAuth encryption key is invalid") from exc


def _jwt_key(path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise HTTPException(503, f"JWT {label} key is not configured") from exc


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID
    auth_session_id: uuid.UUID
    email: str
    display_name: str | None


def issue_access_token(user_id: uuid.UUID, auth_session_id: uuid.UUID, settings: AppSettings) -> str:
    now = utcnow()
    return jwt.encode(
        {
            "sub": str(user_id),
            "sid": str(auth_session_id),
            "jti": str(uuid.uuid4()),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(seconds=settings.access_ttl_seconds),
        },
        _jwt_key(settings.jwt_private_key_path, "private"),
        algorithm="RS256",
        headers={"kid": "aria-local-v1"},
    )


def decode_access_token(token: str, settings: AppSettings) -> tuple[uuid.UUID, uuid.UUID]:
    try:
        claims = jwt.decode(
            token,
            _jwt_key(settings.jwt_public_key_path, "public"),
            algorithms=["RS256"],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={"require": ["sub", "sid", "jti", "iss", "aud", "iat", "nbf", "exp"]},
            leeway=30,
        )
        return uuid.UUID(claims["sub"]), uuid.UUID(claims["sid"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required") from exc


def set_auth_cookies(response: Response, access_token: str, refresh_token: str, settings: AppSettings) -> None:
    common = dict(httponly=True, secure=settings.cookie_secure, samesite="lax")
    response.set_cookie(
        ACCESS_COOKIE,
        access_token,
        max_age=settings.access_ttl_seconds,
        path="/",
        **common,
    )
    response.set_cookie(
        REFRESH_COOKIE,
        refresh_token,
        max_age=settings.auth_session_ttl_seconds,
        path="/api/auth",
        **common,
    )


def clear_auth_cookies(response: Response, settings: AppSettings) -> None:
    response.delete_cookie(ACCESS_COOKIE, path="/", secure=settings.cookie_secure, samesite="lax")
    response.delete_cookie(
        REFRESH_COOKIE, path="/api/auth", secure=settings.cookie_secure, samesite="lax"
    )


async def get_principal(
    request: Request,
    db: AsyncSession = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> Principal:
    token = request.cookies.get(ACCESS_COOKIE)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    user_id, auth_session_id = decode_access_token(token, settings)
    row = await db.execute(
        select(User, AuthSession)
        .join(AuthSession, AuthSession.user_id == User.id)
        .where(
            User.id == user_id,
            User.status == "active",
            AuthSession.id == auth_session_id,
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > utcnow(),
        )
    )
    result = row.one_or_none()
    if result is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    user, _ = result
    return Principal(user.id, auth_session_id, user.email, user.display_name)


async def require_csrf(
    request: Request,
    principal: Principal = Depends(get_principal),
    db: AsyncSession = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
) -> Principal:
    if request.headers.get("origin") not in settings.allowed_origins:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Request origin is not allowed")
    supplied = request.headers.get("x-csrf-token", "")
    auth_session = await db.get(AuthSession, principal.auth_session_id)
    if not supplied or not auth_session or not auth_session.csrf_digest:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF token is required")
    if not secrets.compare_digest(digest(supplied), auth_session.csrf_digest):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF token is invalid")
    return principal


async def create_oauth_transaction(
    db: AsyncSession, return_path: str, settings: AppSettings
) -> tuple[str, str, str, str]:
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(503, "Google login is not configured")
    if not return_path.startswith("/") or return_path.startswith("//"):
        return_path = "/"
    state, nonce, binding = token_urlsafe(), token_urlsafe(), token_urlsafe()
    verifier = token_urlsafe()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    tx = OAuthTransaction(
        state_digest=digest(state),
        nonce_digest=digest(nonce),
        browser_binding_digest=digest(binding),
        pkce_verifier_ciphertext=_fernet(settings).encrypt(verifier.encode()),
        return_path=return_path,
        expires_at=utcnow() + timedelta(seconds=settings.oauth_ttl_seconds),
    )
    db.add(tx)
    await db.commit()
    return state, nonce, binding, challenge


async def exchange_google_code(code: str, verifier: str, nonce: str, settings: AppSettings) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": verifier,
            },
        )
        response.raise_for_status()
        token = response.json()
    id_token = token.get("id_token")
    if not id_token:
        raise HTTPException(401, "Google did not return an identity token")

    def verify() -> dict:
        key = PyJWKClient(GOOGLE_JWKS_URL).get_signing_key_from_jwt(id_token).key
        return jwt.decode(
            id_token,
            key,
            algorithms=["RS256"],
            audience=settings.google_client_id,
            issuer=[GOOGLE_ISSUER, "accounts.google.com"],
            options={"require": ["sub", "iss", "aud", "exp", "iat", "nonce", "email"]},
        )

    try:
        claims = await asyncio.to_thread(verify)
    except (jwt.PyJWTError, httpx.HTTPError) as exc:
        raise HTTPException(401, "Google identity validation failed") from exc
    if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        raise HTTPException(401, "Google login nonce is invalid")
    if claims.get("email_verified") is not True:
        raise HTTPException(401, "A verified Google email is required")
    return claims


async def consume_oauth_transaction(
    db: AsyncSession, state: str, binding: str, settings: AppSettings
) -> tuple[OAuthTransaction, str, str]:
    result = await db.execute(
        select(OAuthTransaction)
        .where(OAuthTransaction.state_digest == digest(state))
        .with_for_update()
    )
    tx = result.scalar_one_or_none()
    if (
        tx is None
        or tx.consumed_at is not None
        or tx.expires_at <= utcnow()
        or not secrets.compare_digest(tx.browser_binding_digest, digest(binding))
    ):
        raise HTTPException(401, "Google login transaction is invalid or expired")
    tx.consumed_at = utcnow()
    try:
        verifier = _fernet(settings).decrypt(tx.pkce_verifier_ciphertext).decode()
    except InvalidToken as exc:
        raise HTTPException(503, "Google login transaction cannot be decrypted") from exc
    # Nonce is intentionally unrecoverable in the schema, so the raw nonce is
    # carried in the bound HttpOnly cookie beside the browser binding value.
    await db.commit()
    return tx, verifier, tx.return_path


async def establish_identity(
    db: AsyncSession, claims: dict, settings: AppSettings
) -> tuple[User, AuthSession, str]:
    result = await db.execute(
        select(OAuthIdentity).where(
            OAuthIdentity.issuer == claims["iss"], OAuthIdentity.subject == claims["sub"]
        )
    )
    identity = result.scalar_one_or_none()
    if identity:
        user = await db.get(User, identity.user_id, with_for_update=True)
        if not user or user.status != "active":
            raise HTTPException(403, "This account is unavailable")
        user.email = claims["email"]
        user.display_name = claims.get("name")
        user.last_login_at = utcnow()
        identity.provider_email = claims["email"]
        identity.email_verified = True
    else:
        user = User(email=claims["email"], display_name=claims.get("name"))
        db.add(user)
        await db.flush()
        db.add(
            OAuthIdentity(
                user_id=user.id,
                provider="google",
                issuer=claims["iss"],
                subject=claims["sub"],
                provider_email=claims["email"],
                email_verified=True,
            )
        )
    auth_session = AuthSession(
        user_id=user.id,
        expires_at=utcnow() + timedelta(seconds=settings.auth_session_ttl_seconds),
    )
    db.add(auth_session)
    await db.flush()
    raw_refresh = token_urlsafe()
    db.add(
        RefreshToken(
            auth_session_id=auth_session.id,
            token_digest=digest(raw_refresh),
            expires_at=auth_session.expires_at,
        )
    )
    await db.commit()
    return user, auth_session, raw_refresh


async def rotate_refresh_token(
    db: AsyncSession, raw_token: str, settings: AppSettings
) -> tuple[User, AuthSession, str]:
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_digest == digest(raw_token)).with_for_update()
    )
    current = result.scalar_one_or_none()
    if current is None:
        raise HTTPException(401, "Refresh token is invalid")
    auth_session = await db.get(AuthSession, current.auth_session_id, with_for_update=True)
    now = utcnow()
    if not auth_session or auth_session.revoked_at or auth_session.expires_at <= now:
        raise HTTPException(401, "Login session is expired")
    if current.consumed_at is not None:
        auth_session.revoked_at = now
        auth_session.revocation_reason = "refresh_reuse"
        await db.commit()
        raise HTTPException(401, "Refresh token reuse detected")
    if current.expires_at <= now:
        raise HTTPException(401, "Refresh token is expired")
    current.consumed_at = now
    replacement = token_urlsafe()
    db.add(
        RefreshToken(
            auth_session_id=auth_session.id,
            token_digest=digest(replacement),
            expires_at=auth_session.expires_at,
        )
    )
    user = await db.get(User, auth_session.user_id)
    if not user or user.status != "active":
        raise HTTPException(403, "This account is unavailable")
    await db.commit()
    return user, auth_session, replacement


def google_authorization_url(state: str, nonce: str, challenge: str, settings: AppSettings) -> str:
    return f"{GOOGLE_AUTH_URL}?{urlencode({'client_id': settings.google_client_id, 'redirect_uri': settings.google_redirect_uri, 'response_type': 'code', 'scope': 'openid email profile', 'state': state, 'nonce': nonce, 'code_challenge': challenge, 'code_challenge_method': 'S256', 'prompt': 'select_account'})}"
