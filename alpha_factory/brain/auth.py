"""Thread-safe WorldQuant BRAIN authentication, session cache, and headless biometric flow."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

DEFAULT_BRAIN_API_URL = "https://api.worldquantbrain.com"
SESSION_TOKEN_HEADER = "X-WQB-Session-Token"


class BrainAPIError(Exception):
    """Base exception for all WorldQuant BRAIN API errors."""


class AuthenticationError(BrainAPIError):
    """Authentication failure (HTTP 401 or missing credentials)."""

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__(detail or "Authentication failed")


class RecaptchaRequiredError(AuthenticationError):
    """401 response requiring a reCAPTCHA verification challenge."""


class WebAuthnRequiredError(AuthenticationError):
    """401 response requiring a WebAuthn / FIDO2 credential challenge."""

    def __init__(self, options: Any = None) -> None:
        self.options = options
        super().__init__("WebAuthn authentication required")


class BiometricAuthTimeoutError(AuthenticationError, TimeoutError):
    """Headless Persona biometric verification exceeded biometric_max_wait_s."""


class RateLimitError(BrainAPIError):
    """HTTP 429 rate-limit response exceeded configured retry budget."""

    def __init__(
        self,
        message: str = "BRAIN API rate limit exceeded",
        retry_after: Optional[float] = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message)


def _get_header(headers: Optional[Mapping[str, Any]], name: str) -> Optional[str]:
    """Return a header value using case-insensitive lookup on real or mock header dicts."""
    if not headers:
        return None
    if name in headers:
        val = headers[name]
        return None if val is None else str(val)
    target = name.lower()
    for key, val in headers.items():
        if str(key).lower() == target:
            return None if val is None else str(val)
    return None


def _pop_header(headers: Any, name: str) -> None:
    """Remove a header from a mutable mapping case-insensitively."""
    if not headers or not hasattr(headers, "pop"):
        return
    target = name.lower()
    matching = [key for key in list(headers.keys()) if str(key).lower() == target]
    for key in matching:
        headers.pop(key, None)


def _parse_json_safe(resp: Any) -> Any:
    """Parse JSON from a response object, returning None if body is empty or non-JSON."""
    try:
        return resp.json()
    except (ValueError, TypeError):
        return None


def _parse_json_dict(resp: Any) -> dict[str, Any]:
    """Parse a JSON dict from a response object, falling back to {} when absent."""
    parsed = _parse_json_safe(resp)
    return parsed if isinstance(parsed, dict) else {}


def _parse_retry_after_positive(headers: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Parse a positive finite Retry-After float in seconds, or return None."""
    raw = _get_header(headers, "Retry-After")
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val) or val <= 0:
        return None
    return val


class BrainAuthManager:
    """Manages BRAIN credentials, session cookies, session token caching, and re-auth."""

    def __init__(
        self,
        api_url: str = DEFAULT_BRAIN_API_URL,
        credentials_path: str | Path = "~/secrets/platform-brain.json",
        session_cache_path: str | Path = "~/secrets/alpha-factory-session.json",
        relogin_ttl_threshold_s: float = 1800,
        biometric_poll_interval_s: float = 15,
        biometric_max_wait_s: float = 1800,
        connect_timeout_s: float = 10,
        read_timeout_s: float = 60,
        alert_callback: Optional[Callable[[str], Any]] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.api_url = (api_url or DEFAULT_BRAIN_API_URL).rstrip("/")
        self.credentials_path = Path(credentials_path).expanduser()
        self.session_cache_path = Path(session_cache_path).expanduser()
        self.relogin_ttl_threshold_s = float(relogin_ttl_threshold_s)
        self.biometric_poll_interval_s = float(biometric_poll_interval_s)
        self.biometric_max_wait_s = float(biometric_max_wait_s)
        self.timeout: tuple[float, float] = (float(connect_timeout_s), float(read_timeout_s))
        self.alert_callback = alert_callback
        self._explicit_email = email
        self._explicit_password = password
        self._lock = threading.RLock()
        self._auth_generation = 0

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        alert_callback: Optional[Callable[[str], Any]] = None,
    ) -> BrainAuthManager:
        """Construct a BrainAuthManager from loaded settings or its ``brain`` section."""
        brain_cfg = settings.get("brain", settings)
        if not isinstance(brain_cfg, Mapping):
            brain_cfg = {}
        return cls(
            api_url=str(brain_cfg.get("api_url", DEFAULT_BRAIN_API_URL)),
            credentials_path=brain_cfg.get("credentials_path", "~/secrets/platform-brain.json"),
            session_cache_path=brain_cfg.get("session_cache_path", "~/secrets/alpha-factory-session.json"),
            relogin_ttl_threshold_s=float(brain_cfg.get("relogin_ttl_threshold_s", 1800)),
            biometric_poll_interval_s=float(brain_cfg.get("biometric_poll_interval_s", 15)),
            biometric_max_wait_s=float(brain_cfg.get("biometric_max_wait_s", 1800)),
            connect_timeout_s=float(brain_cfg.get("connect_timeout_s", 10)),
            read_timeout_s=float(brain_cfg.get("read_timeout_s", 60)),
            alert_callback=alert_callback,
        )

    @property
    def lock(self) -> threading.RLock:
        """Return the shared re-entrant lock protecting all auth and cache state."""
        return self._lock

    @property
    def auth_generation(self) -> int:
        """Monotonic counter incremented on every fresh authentication."""
        with self._lock:
            return self._auth_generation

    def load_credentials(self) -> tuple[str, str]:
        """Load email/password from explicit args, environment vars, or credentials JSON."""
        if self._explicit_email and self._explicit_password:
            return str(self._explicit_email), str(self._explicit_password)

        env_email = os.environ.get("BRAIN_CREDENTIAL_EMAIL", "").strip()
        env_password = os.environ.get("BRAIN_CREDENTIAL_PASSWORD", "")
        if env_email and env_password:
            return env_email, env_password

        if self.credentials_path.is_file():
            try:
                raw = json.loads(self.credentials_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AuthenticationError(
                    f"Failed to read BRAIN credentials file {self.credentials_path}: {exc}"
                ) from exc
            if isinstance(raw, dict):
                email = str(raw.get("email") or raw.get("username") or "").strip()
                password = str(raw.get("password") or "")
                if email and password:
                    return email, password

        raise AuthenticationError(
            "Missing BRAIN credentials: set BRAIN_CREDENTIAL_EMAIL and "
            f"BRAIN_CREDENTIAL_PASSWORD or populate {self.credentials_path}"
        )

    @staticmethod
    def _extract_cookies(session: Any) -> list[dict[str, Any]]:
        cookies = getattr(session, "cookies", None)
        if cookies is None:
            return []
        if isinstance(cookies, dict):
            return [{"name": str(k), "value": str(v)} for k, v in cookies.items()]

        serialized: list[dict[str, Any]] = []
        try:
            for cookie in cookies:
                if hasattr(cookie, "name") and hasattr(cookie, "value"):
                    serialized.append(
                        {
                            "name": str(cookie.name),
                            "value": "" if cookie.value is None else str(cookie.value),
                            "domain": getattr(cookie, "domain", None),
                            "path": getattr(cookie, "path", "/"),
                            "secure": bool(getattr(cookie, "secure", False)),
                            "expires": getattr(cookie, "expires", None),
                        }
                    )
        except TypeError:
            return []
        return serialized

    @staticmethod
    def _apply_cookies(session: Any, cookies_data: list[dict[str, Any]]) -> None:
        cookies = getattr(session, "cookies", None)
        if cookies is None:
            return
        for item in cookies_data:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            value = "" if item.get("value") is None else str(item["value"])
            if isinstance(cookies, dict):
                cookies[name] = value
            elif hasattr(cookies, "set"):
                kwargs: dict[str, Any] = {}
                if item.get("domain"):
                    kwargs["domain"] = item["domain"]
                if item.get("path"):
                    kwargs["path"] = item["path"]
                if item.get("secure") is not None:
                    kwargs["secure"] = bool(item["secure"])
                if item.get("expires") is not None:
                    kwargs["expires"] = item["expires"]
                cookies.set(name, value, **kwargs)

    @staticmethod
    def _session_has_auth(session: Any) -> bool:
        headers = getattr(session, "headers", None)
        if _get_header(headers, SESSION_TOKEN_HEADER):
            return True
        cookies = getattr(session, "cookies", None)
        if cookies is None:
            return False
        try:
            return len(cookies) > 0
        except TypeError:
            return False

    @staticmethod
    def _capture_session_token(resp: Any, session: Any) -> None:
        token = _get_header(getattr(resp, "headers", None), SESSION_TOKEN_HEADER)
        if token:
            headers = getattr(session, "headers", None)
            if headers is not None:
                _pop_header(headers, SESSION_TOKEN_HEADER)
                headers[SESSION_TOKEN_HEADER] = token

    def save_session_cache(self, session: Any) -> None:
        """Persist session cookies and X-WQB-Session-Token atomically with 0o600 permissions."""
        with self._lock:
            self.session_cache_path.parent.mkdir(parents=True, exist_ok=True)
            headers = getattr(session, "headers", None)
            session_token = _get_header(headers, SESSION_TOKEN_HEADER) or ""
            payload = {
                "cookies": self._extract_cookies(session),
                "session_token": session_token,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            encoded = json.dumps(payload, indent=2)

            fd, tmp_name = tempfile.mkstemp(
                prefix=".session_cache_",
                suffix=".tmp",
                dir=str(self.session_cache_path.parent),
            )
            tmp_path = Path(tmp_name)
            try:
                os.chmod(tmp_path, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(encoded)
                os.replace(tmp_path, self.session_cache_path)
                os.chmod(self.session_cache_path, 0o600)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

    def load_session_cache(self, session: Any) -> bool:
        """Restore cached cookies and X-WQB-Session-Token onto ``session`` if present."""
        with self._lock:
            if not self.session_cache_path.is_file():
                return False
            try:
                raw = json.loads(self.session_cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Ignoring unreadable session cache %s: %s", self.session_cache_path, exc)
                return False

            if not isinstance(raw, dict):
                return False

            cookies_data = raw.get("cookies")
            restored = False
            if isinstance(cookies_data, list) and cookies_data:
                self._apply_cookies(session, cookies_data)
                restored = True

            token = raw.get("session_token")
            if isinstance(token, str) and token:
                headers = getattr(session, "headers", None)
                if headers is not None:
                    _pop_header(headers, SESSION_TOKEN_HEADER)
                    headers[SESSION_TOKEN_HEADER] = token
                    restored = True
            return restored

    def clear_session_cache(self, session: Optional[Any] = None) -> None:
        """Clear cached auth artifacts from disk and optionally from ``session``."""
        with self._lock:
            if session is not None:
                _pop_header(getattr(session, "headers", None), SESSION_TOKEN_HEADER)
                cookies = getattr(session, "cookies", None)
                if cookies is not None and hasattr(cookies, "clear"):
                    cookies.clear()
            try:
                self.session_cache_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove session cache %s: %s", self.session_cache_path, exc)

    def check_authenticated(self, session: Any) -> Optional[dict[str, Any]]:
        """Query GET /authentication; return user/token JSON on 200 or None on 204."""
        with self._lock:
            url = f"{self.api_url}/authentication"
            resp = session.get(url, timeout=self.timeout)
            if resp.status_code == 204:
                return None
            if resp.status_code == 429:
                retry_after = _parse_retry_after_positive(getattr(resp, "headers", None))
                raise RateLimitError(
                    "Rate limited during GET /authentication",
                    retry_after=retry_after,
                )
            resp.raise_for_status()
            self._capture_session_token(resp, session)
            return _parse_json_dict(resp)

    def _needs_relogin(self, auth_info: dict[str, Any]) -> bool:
        token = auth_info.get("token")
        if not isinstance(token, dict):
            return False
        expiry = token.get("expiry")
        if expiry is None:
            return False

        remaining_s: Optional[float] = None
        if isinstance(expiry, (int, float)):
            remaining_s = float(expiry)
        elif isinstance(expiry, str):
            try:
                remaining_s = float(expiry)
            except ValueError:
                try:
                    dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    remaining_s = (dt - datetime.now(timezone.utc)).total_seconds()
                except ValueError:
                    return False

        if remaining_s is None:
            return False
        return remaining_s <= 0 or remaining_s < self.relogin_ttl_threshold_s

    @staticmethod
    def _is_recaptcha_challenge(body: Any) -> bool:
        if not isinstance(body, dict) or "recaptcha" not in body:
            return False
        val = body["recaptcha"]
        if val is True:
            return True
        if isinstance(val, str):
            return "required" in val.lower()
        if isinstance(val, list):
            return any("required" in str(item).lower() for item in val)
        return False

    @staticmethod
    def _is_webauthn_challenge(body: Any) -> bool:
        if not isinstance(body, dict):
            return False
        if "publicKey" in body or "credentialOptions" in body:
            return True
        credentials = body.get("credentials")
        if isinstance(credentials, dict) and "createOptions" in credentials:
            return True
        return str(body.get("error", "")).lower() == "webauthn"

    def _handle_biometric_challenge(self, session: Any, resp: Any, location: str) -> dict[str, Any]:
        base_url = getattr(resp, "url", None) or f"{self.api_url}/authentication"
        verify_url = urljoin(base_url, location)
        logger.warning("Persona biometric verification required. Verification URL: %s", verify_url)

        if self.alert_callback is not None:
            try:
                self.alert_callback(verify_url)
            except Exception:
                logger.exception("alert_callback failed while delivering verification URL %s", verify_url)

        started = time.monotonic()
        deadline = started + self.biometric_max_wait_s
        simulated_elapsed = 0.0
        interval = max(0.0, self.biometric_poll_interval_s)

        while True:
            wall_elapsed = time.monotonic() - started
            remaining = self.biometric_max_wait_s - max(simulated_elapsed, wall_elapsed)
            if remaining <= 0:
                raise BiometricAuthTimeoutError(
                    f"Persona biometric authentication timed out after {self.biometric_max_wait_s}s: {verify_url}"
                )

            sleep_s = min(interval, remaining) if interval > 0 else 0.0
            if sleep_s > 0:
                time.sleep(sleep_s)
                simulated_elapsed += sleep_s
            elif time.monotonic() >= deadline:
                raise BiometricAuthTimeoutError(
                    f"Persona biometric authentication timed out after {self.biometric_max_wait_s}s: {verify_url}"
                )

            poll_resp = session.post(verify_url, timeout=self.timeout)
            if poll_resp.status_code == 201:
                self._capture_session_token(poll_resp, session)
                self.save_session_cache(session)
                self._auth_generation += 1
                logger.info("Persona biometric verification completed successfully.")
                return _parse_json_dict(poll_resp)

            if poll_resp.status_code == 429:
                retry_after = _parse_retry_after_positive(getattr(poll_resp, "headers", None))
                if retry_after is not None:
                    extra_remaining = self.biometric_max_wait_s - max(
                        simulated_elapsed, time.monotonic() - started
                    )
                    if extra_remaining <= 0:
                        raise BiometricAuthTimeoutError(
                            f"Persona biometric authentication timed out while rate-limited: {verify_url}"
                        )
                    pause = min(retry_after, extra_remaining)
                    if pause > 0:
                        time.sleep(pause)
                        simulated_elapsed += pause
                continue

            if poll_resp.status_code in (401, 403, 200, 202, 204) or poll_resp.status_code >= 500:
                continue

            poll_resp.raise_for_status()

    def ensure_authenticated(self, session: Any, force: bool = False) -> dict[str, Any]:
        """Return active auth payload, restoring cache or logging in under a single RLock."""
        with self._lock:
            if not force:
                if not self._session_has_auth(session):
                    self.load_session_cache(session)
                try:
                    info = self.check_authenticated(session)
                except requests.HTTPError as exc:
                    resp = getattr(exc, "response", None)
                    if resp is not None and resp.status_code == 401:
                        info = None
                    else:
                        raise

                if info is not None and not self._needs_relogin(info):
                    self._capture_session_token_from_cache_refresh(session)
                    return info

            email, password = self.load_credentials()
            _pop_header(getattr(session, "headers", None), SESSION_TOKEN_HEADER)
            cookies = getattr(session, "cookies", None)
            if cookies is not None and hasattr(cookies, "clear"):
                cookies.clear()

            auth_url = f"{self.api_url}/authentication"
            resp = session.post(auth_url, auth=(email, password), timeout=self.timeout)

            if resp.status_code in (200, 201):
                self._capture_session_token(resp, session)
                self.save_session_cache(session)
                self._auth_generation += 1
                return _parse_json_dict(resp)

            if resp.status_code == 401:
                headers = getattr(resp, "headers", None)
                www_auth = (_get_header(headers, "WWW-Authenticate") or "").lower()
                location = _get_header(headers, "Location")
                if ("persona" in www_auth or "inquiry" in www_auth) and location:
                    return self._handle_biometric_challenge(session, resp, location)

                body = _parse_json_safe(resp)
                if self._is_recaptcha_challenge(body):
                    raise RecaptchaRequiredError("reCAPTCHA verification required during login")
                if self._is_webauthn_challenge(body):
                    raise WebAuthnRequiredError(options=body)

                detail = ""
                if isinstance(body, dict):
                    detail = str(body.get("detail") or body.get("code") or body.get("error") or body)
                elif getattr(resp, "text", ""):
                    detail = resp.text[:300]
                raise AuthenticationError(detail=detail)

            if resp.status_code == 429:
                retry_after = _parse_retry_after_positive(getattr(resp, "headers", None))
                raise RateLimitError(
                    "Rate limited during POST /authentication",
                    retry_after=retry_after,
                )

            resp.raise_for_status()
            return _parse_json_dict(resp)

    def _capture_session_token_from_cache_refresh(self, session: Any) -> None:
        """Persist updated token/cookies if session cache doesn't exist yet."""
        if not self.session_cache_path.is_file() and self._session_has_auth(session):
            self.save_session_cache(session)

    def logout(self, session: Any) -> bool:
        """Terminate current BRAIN session via DELETE /authentication."""
        with self._lock:
            url = f"{self.api_url}/authentication"
            resp = session.delete(url, timeout=self.timeout)
            if 200 <= resp.status_code < 300:
                self.clear_session_cache(session)
                self._auth_generation += 1
                return True
            if resp.status_code == 401:
                logger.info("Logout no-op: BRAIN session already invalid (401).")
                self.clear_session_cache(session)
                return False
            resp.raise_for_status()
            return False


__all__ = [
    "AuthenticationError",
    "BiometricAuthTimeoutError",
    "BrainAPIError",
    "BrainAuthManager",
    "DEFAULT_BRAIN_API_URL",
    "RateLimitError",
    "RecaptchaRequiredError",
    "SESSION_TOKEN_HEADER",
    "WebAuthnRequiredError",
]
