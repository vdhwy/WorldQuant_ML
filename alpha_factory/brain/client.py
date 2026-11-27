"""Unified thread-safe WorldQuant BRAIN API client with rate-limit coordination and polling."""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Any, Callable, Mapping, Optional, Union
from urllib.parse import urljoin

import requests

from alpha_factory.brain.auth import (
    DEFAULT_BRAIN_API_URL,
    AuthenticationError,
    BiometricAuthTimeoutError,
    BrainAPIError,
    BrainAuthManager,
    RateLimitError,
    RecaptchaRequiredError,
    WebAuthnRequiredError,
    _get_header,
    _parse_json_dict,
    _parse_json_safe,
    _parse_retry_after_positive,
)

logger = logging.getLogger(__name__)

SIMULATION_TERMINAL_OK = ("COMPLETE", "WARNING")
SIMULATION_TERMINAL_FAILED = ("ERROR", "FAIL", "TIMEOUT", "CANCELLED")
SIMULATION_IN_PROGRESS = ("WAITING", "SIMULATING")

MIN_MULTI_SIMULATIONS = 2
MAX_MULTI_SIMULATIONS = 10
VALID_CORRELATION_KINDS = ("self", "prod", "power-pool")


class SimulationValidationError(BrainAPIError):
    """HTTP 400 validation failure on POST /simulations with flattened field errors."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors) if self.errors else "Simulation validation failed")


class SimulationError(BrainAPIError):
    """Domain-level simulation failure (ERROR, FAIL, TIMEOUT, CANCELLED, or non-terminal stop)."""

    def __init__(
        self,
        status: str,
        message: str = "",
        location: Optional[dict[str, Any]] = None,
        simulation_id: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self.status = status
        self.message = message or ""
        self.location = location
        self.simulation_id = simulation_id or ""
        self.payload = payload or {}
        suffix = f" (simulation id: {self.simulation_id})" if self.simulation_id else ""
        if self.location is not None:
            loc = (
                f" at property={self.location.get('property')!r} "
                f"line={self.location.get('line')} start={self.location.get('start')} "
                f"end={self.location.get('end')}"
            )
        else:
            loc = ""
        super().__init__(f"Simulation {self.status}: {self.message}{loc}{suffix}")


class PollingTimeoutError(BrainAPIError, TimeoutError):
    """Polling exceeded its hard wall-clock time ceiling."""


def simulation_id_from_url(url: str) -> str:
    """Extract the simulation ID segment from a Location progress URL."""
    return url.rstrip("/").rsplit("/", 1)[-1]


def validate_simulation_payload(payload: Union[dict[str, Any], list[dict[str, Any]]]) -> None:
    """Validate a single simulation dict or a multi-simulation list of 2..10 dicts."""
    if isinstance(payload, dict):
        return
    if isinstance(payload, list):
        if not (MIN_MULTI_SIMULATIONS <= len(payload) <= MAX_MULTI_SIMULATIONS):
            raise ValueError(
                f"Multi-simulation requires between {MIN_MULTI_SIMULATIONS} and "
                f"{MAX_MULTI_SIMULATIONS} simulation objects, got {len(payload)}"
            )
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Multi-simulation item at index {idx} must be a dict")
        return
    raise ValueError("Simulation payload must be a dict or a list of 2..10 dicts")


def flatten_validation_errors(body: Any, prefix: str = "") -> list[str]:
    """Recursively flatten nested 400 validation error bodies into dotted strings."""
    errors: list[str] = []
    if isinstance(body, dict):
        for key, value in body.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                errors.extend(flatten_validation_errors(value, prefix=path))
            elif isinstance(value, list):
                for entry in value:
                    if isinstance(entry, (dict, list)):
                        errors.extend(flatten_validation_errors(entry, prefix=path))
                    else:
                        errors.append(f"{path}: {entry}")
            else:
                errors.append(f"{path}: {value}")
    elif isinstance(body, list):
        for entry in body:
            if isinstance(entry, (dict, list)):
                errors.extend(flatten_validation_errors(entry, prefix=prefix))
            else:
                errors.append(f"{prefix}: {entry}" if prefix else str(entry))
    elif isinstance(body, str):
        errors.append(f"{prefix}: {body}" if prefix else body)
    elif body is not None:
        errors.append(f"{prefix}: {body}" if prefix else str(body))
    return errors


def ensure_simulation_finished(
    payload: dict[str, Any],
    simulation_id: str = "",
) -> dict[str, Any]:
    """Validate terminal simulation status; raise SimulationError on failure or non-terminal."""
    status = str(payload.get("status", ""))
    sim_id = simulation_id or str(payload.get("id", ""))

    if status in SIMULATION_TERMINAL_FAILED:
        logger.error(
            "Simulation failed [sim_id=%s] status=%s message=%s",
            sim_id,
            status,
            payload.get("message", ""),
        )
        raise SimulationError(
            status=status,
            message=str(payload.get("message", "")),
            location=payload.get("location"),
            simulation_id=sim_id,
            payload=payload,
        )

    if status not in SIMULATION_TERMINAL_OK:
        logger.error(
            "Simulation stopped in non-terminal state [sim_id=%s] status=%r",
            sim_id,
            status,
        )
        raise SimulationError(
            status=status or "UNKNOWN",
            message=f"Simulation did not reach a terminal state (status={status!r})",
            location=payload.get("location"),
            simulation_id=sim_id,
            payload=payload,
        )

    if status == "WARNING":
        logger.warning(
            "Simulation completed with WARNING [sim_id=%s]%s",
            sim_id,
            f": {payload.get('message')}" if payload.get("message") else "",
        )
    return payload


def parse_recordset(payload: dict[str, Any]) -> dict[str, Any]:
    """Zip ``schema.properties`` column names onto ``records`` rows."""
    properties = (payload.get("schema") or {}).get("properties") or []
    names = [prop.get("name") for prop in properties]
    records = payload.get("records") or []
    rows = [dict(zip(names, record)) for record in records]
    return {
        "names": names,
        "properties": properties,
        "records": records,
        "rows": rows,
        "min": payload.get("min"),
        "max": payload.get("max"),
    }


def _handle_response(
    resp: Any,
    context: str = "",
    entity_id: str = "",
) -> Any:
    """Log entity context and raise HTTPError for 4xx/5xx responses."""
    if resp.status_code >= 400:
        tag = f" [{entity_id}]" if entity_id else ""
        preview = str(getattr(resp, "text", ""))[:300]
        logger.error("HTTP %s on %s%s: %s", resp.status_code, context, tag, preview)
        resp.raise_for_status()
    return resp


class BrainClient:
    """Unified thread-safe WorldQuant BRAIN REST API client."""

    def __init__(
        self,
        auth_manager: Optional[BrainAuthManager] = None,
        session: Optional[requests.Session] = None,
        api_url: str = DEFAULT_BRAIN_API_URL,
        connect_timeout_s: Optional[float] = None,
        read_timeout_s: Optional[float] = None,
        max_wait_s: float = 3600,
        recordset_max_wait_s: float = 600,
        default_interval_s: float = 5,
        max_http_retries: int = 3,
    ) -> None:
        if auth_manager is not None and api_url == DEFAULT_BRAIN_API_URL:
            api_url = auth_manager.api_url

        default_connect = auth_manager.timeout[0] if auth_manager is not None else 10.0
        default_read = auth_manager.timeout[1] if auth_manager is not None else 60.0
        resolved_connect = float(default_connect if connect_timeout_s is None else connect_timeout_s)
        resolved_read = float(default_read if read_timeout_s is None else read_timeout_s)

        self.api_url = (api_url or DEFAULT_BRAIN_API_URL).rstrip("/")
        self.connect_timeout_s = resolved_connect
        self.read_timeout_s = resolved_read
        self.timeout: tuple[float, float] = (self.connect_timeout_s, self.read_timeout_s)
        self.max_wait_s = float(max_wait_s)
        self.recordset_max_wait_s = float(recordset_max_wait_s)
        self.default_interval_s = float(default_interval_s)
        self.max_http_retries = int(max_http_retries)

        self.session: Any = session if session is not None else requests.Session()
        if auth_manager is not None:
            if connect_timeout_s is not None or read_timeout_s is not None:
                auth_manager.timeout = self.timeout
            self.auth_manager = auth_manager
        else:
            self.auth_manager = BrainAuthManager(
                api_url=self.api_url,
                connect_timeout_s=self.connect_timeout_s,
                read_timeout_s=self.read_timeout_s,
            )

        self._rate_limit_lock = threading.Lock()
        self._rate_limit_until = 0.0

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        session: Optional[requests.Session] = None,
        alert_callback: Optional[Callable[[str], Any]] = None,
    ) -> BrainClient:
        """Create a BrainClient configured from loaded settings YAML."""
        brain_cfg = settings.get("brain", settings)
        if not isinstance(brain_cfg, Mapping):
            brain_cfg = {}
        polling_cfg = settings.get("polling", {})
        if not isinstance(polling_cfg, Mapping):
            polling_cfg = {}

        auth_manager = BrainAuthManager.from_settings(settings, alert_callback=alert_callback)
        return cls(
            auth_manager=auth_manager,
            session=session,
            api_url=str(brain_cfg.get("api_url", DEFAULT_BRAIN_API_URL)),
            connect_timeout_s=float(brain_cfg.get("connect_timeout_s", 10)),
            read_timeout_s=float(brain_cfg.get("read_timeout_s", 60)),
            max_wait_s=float(polling_cfg.get("max_wait_s", 3600)),
            recordset_max_wait_s=float(polling_cfg.get("recordset_max_wait_s", 600)),
            default_interval_s=float(polling_cfg.get("default_interval_s", 5)),
            max_http_retries=int(polling_cfg.get("max_http_retries", 3)),
        )

    validate_simulation_payload = staticmethod(validate_simulation_payload)
    flatten_validation_errors = staticmethod(flatten_validation_errors)
    ensure_simulation_finished = staticmethod(ensure_simulation_finished)
    parse_recordset = staticmethod(parse_recordset)

    def ensure_authenticated(self, force: bool = False) -> dict[str, Any]:
        """Delegate session authentication to ``BrainAuthManager``."""
        return self.auth_manager.ensure_authenticated(self.session, force=force)

    def check_authenticated(self) -> Optional[dict[str, Any]]:
        """Delegate GET /authentication check to ``BrainAuthManager``."""
        return self.auth_manager.check_authenticated(self.session)

    def logout(self) -> bool:
        """Delegate DELETE /authentication logout to ``BrainAuthManager``."""
        return self.auth_manager.logout(self.session)

    def _build_url(self, url_or_path: str) -> str:
        if url_or_path.startswith(("http://", "https://")):
            return url_or_path
        return urljoin(self.api_url + "/", url_or_path.lstrip("/"))

    def _dispatch_session(self, method: str, url: str, **kwargs: Any) -> Any:
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("timeout", self.timeout)
        method_lower = method.lower()
        method_fn = getattr(self.session, method_lower, None)
        if callable(method_fn):
            return method_fn(url, **call_kwargs)
        return self.session.request(method.upper(), url, **call_kwargs)

    def _wait_for_rate_limit(self) -> None:
        while True:
            with self._rate_limit_lock:
                target = self._rate_limit_until
            wait_for = target - time.monotonic()
            if wait_for <= 0:
                return
            time.sleep(wait_for)
            with self._rate_limit_lock:
                if self._rate_limit_until <= target:
                    if time.monotonic() < target:
                        self._rate_limit_until = 0.0
                    return

    def _compute_backoff(self, attempt: int) -> float:
        base = max(1.0, self.default_interval_s)
        return min(60.0, base * (2**attempt)) * random.uniform(0.75, 1.25)

    def _record_rate_limit(self, headers: Optional[Mapping[str, Any]], attempt: int) -> float:
        delay = _parse_retry_after_positive(headers)
        if delay is None:
            delay = self._compute_backoff(attempt)
        until = time.monotonic() + delay
        with self._rate_limit_lock:
            if until > self._rate_limit_until:
                self._rate_limit_until = until
        return delay

    def _single_http_call(
        self,
        method: str,
        url: str,
        *,
        allow_reauth: bool = True,
        **kwargs: Any,
    ) -> Any:
        full_url = self._build_url(url)
        gen_before = self.auth_manager.auth_generation
        resp = self._dispatch_session(method, full_url, **kwargs)

        if resp.status_code == 401 and allow_reauth:
            logger.info("HTTP 401 on %s %s; forcing re-authentication and retrying once.", method, full_url)
            with self.auth_manager.lock:
                if self.auth_manager.auth_generation == gen_before:
                    self.auth_manager.ensure_authenticated(self.session, force=True)
            resp = self._dispatch_session(method, full_url, **kwargs)
        return resp

    def _request(
        self,
        method: str,
        url: str,
        *,
        allow_reauth: bool = True,
        max_http_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        full_url = self._build_url(url)
        retries = self.max_http_retries if max_http_retries is None else int(max_http_retries)

        for attempt in range(retries + 1):
            self._wait_for_rate_limit()
            try:
                resp = self._single_http_call(method, full_url, allow_reauth=allow_reauth, **kwargs)
            except requests.RequestException as exc:
                if attempt < retries:
                    delay = self._compute_backoff(attempt)
                    logger.warning(
                        "Transient request error on %s %s (%s/%s), retrying in %.2fs: %s",
                        method,
                        full_url,
                        attempt + 1,
                        retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                raise

            if resp.status_code == 429:
                delay = self._record_rate_limit(getattr(resp, "headers", None), attempt)
                if attempt < retries:
                    logger.warning(
                        "HTTP 429 on %s %s (%s/%s), pausing all threads for %.2fs",
                        method,
                        full_url,
                        attempt + 1,
                        retries,
                        delay,
                    )
                    continue
                raise RateLimitError(
                    f"HTTP 429 rate limit exceeded on {method.upper()} {full_url}",
                    retry_after=delay,
                )

            if resp.status_code >= 500 and attempt < retries:
                delay = self._compute_backoff(attempt)
                logger.warning(
                    "HTTP %s on %s %s (%s/%s), retrying in %.2fs",
                    resp.status_code,
                    method,
                    full_url,
                    attempt + 1,
                    retries,
                    delay,
                )
                time.sleep(delay)
                continue

            return resp

        raise BrainAPIError(f"Exhausted retries on {method.upper()} {full_url}")

    def poll(
        self,
        url: str,
        max_wait_s: Optional[float] = None,
        default_interval_s: Optional[float] = None,
        max_http_retries: Optional[int] = None,
    ) -> Any:
        """Poll a URL respecting Retry-After until terminal response or hard time ceiling."""
        full_url = self._build_url(url)
        ceiling = self.max_wait_s if max_wait_s is None else float(max_wait_s)
        fallback_interval = (
            self.default_interval_s if default_interval_s is None else float(default_interval_s)
        )
        retries = self.max_http_retries if max_http_retries is None else int(max_http_retries)

        started = time.monotonic()
        simulated_elapsed = 0.0
        http_failures = 0

        def _sleep_within_ceiling(wait_s: float) -> None:
            nonlocal simulated_elapsed
            elapsed = max(simulated_elapsed, time.monotonic() - started)
            if elapsed + wait_s > ceiling:
                raise PollingTimeoutError(f"Polling {full_url} exceeded {ceiling}s")
            time.sleep(wait_s)
            simulated_elapsed += wait_s

        while True:
            if simulated_elapsed > ceiling or (
                ceiling > 0 and (time.monotonic() - started) >= ceiling
            ):
                raise PollingTimeoutError(f"Polling {full_url} exceeded {ceiling}s")

            self._wait_for_rate_limit()

            try:
                resp = self._single_http_call("GET", full_url)
            except requests.RequestException as exc:
                if http_failures < retries:
                    http_failures += 1
                    logger.warning(
                        "Transient polling exception for %s (%s/%s), retrying in %ss: %s",
                        full_url,
                        http_failures,
                        retries,
                        fallback_interval,
                        exc,
                    )
                    _sleep_within_ceiling(max(0.0, fallback_interval))
                    continue
                raise

            if resp.status_code == 429:
                delay = self._record_rate_limit(getattr(resp, "headers", None), http_failures)
                if http_failures < retries:
                    http_failures += 1
                    elapsed = max(simulated_elapsed, time.monotonic() - started)
                    if elapsed + delay > ceiling:
                        raise PollingTimeoutError(
                            f"Polling {full_url} exceeded {ceiling}s while rate-limited"
                        )
                    logger.warning(
                        "HTTP 429 polling %s (%s/%s), pausing for %.2fs",
                        full_url,
                        http_failures,
                        retries,
                        delay,
                    )
                    self._wait_for_rate_limit()
                    simulated_elapsed += delay
                    continue
                raise RateLimitError(
                    f"HTTP 429 rate limit exceeded polling {full_url}",
                    retry_after=delay,
                )

            if resp.status_code >= 500:
                if http_failures < retries:
                    http_failures += 1
                    logger.warning(
                        "HTTP %s polling %s (%s/%s), retrying in %ss",
                        resp.status_code,
                        full_url,
                        http_failures,
                        retries,
                        fallback_interval,
                    )
                    _sleep_within_ceiling(max(0.0, fallback_interval))
                    continue
                resp.raise_for_status()

            if resp.status_code >= 400:
                resp.raise_for_status()

            http_failures = 0

            retry_raw = _get_header(getattr(resp, "headers", None), "Retry-After")
            if retry_raw is None:
                return resp

            try:
                wait_s = float(retry_raw)
            except (TypeError, ValueError):
                wait_s = fallback_interval
            else:
                if wait_s == 0:
                    return resp
                if math.isnan(wait_s) or math.isinf(wait_s) or wait_s < 0:
                    wait_s = fallback_interval

            _sleep_within_ceiling(wait_s)

    def submit_simulation(
        self,
        payload: Union[dict[str, Any], list[dict[str, Any]]],
    ) -> str:
        """Validate payload, POST /simulations, and return Location progress URL."""
        validate_simulation_payload(payload)
        resp = self._request("POST", "/simulations", json=payload)

        if resp.status_code == 400:
            body = _parse_json_safe(resp)
            if body is None:
                body = getattr(resp, "text", "")
            errors = flatten_validation_errors(body)
            logger.error("Simulation submission rejected (400): %s", errors)
            raise SimulationValidationError(errors)

        _handle_response(resp, context="POST /simulations")

        location = _get_header(getattr(resp, "headers", None), "Location")
        if not location:
            raise BrainAPIError(
                f"Simulation accepted ({resp.status_code}) without a Location header"
            )
        return location

    def poll_simulation(
        self,
        location_url: str,
        max_wait_s: Optional[float] = None,
    ) -> dict[str, Any]:
        """Poll a simulation URL to completion and verify terminal status."""
        sim_id = simulation_id_from_url(location_url)
        resp = self.poll(location_url, max_wait_s=max_wait_s)
        payload = _parse_json_dict(resp)
        return ensure_simulation_finished(payload, simulation_id=sim_id)

    def poll_simulation_children(
        self,
        parent_payload: dict[str, Any],
        max_wait_s: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Poll every child simulation in a multi-simulation batch individually."""
        children = parent_payload.get("children") or []
        results: list[dict[str, Any]] = []
        for child in children:
            child_str = str(child)
            child_url = (
                child_str
                if child_str.startswith(("http://", "https://"))
                else f"{self.api_url}/simulations/{child_str}"
            )
            results.append(self.poll_simulation(child_url, max_wait_s=max_wait_s))
        return results

    def get_alpha(self, alpha_id: str) -> dict[str, Any]:
        """Fetch GET /alphas/{alpha_id}."""
        resp = self._request("GET", f"/alphas/{alpha_id}")
        _handle_response(resp, context="GET /alphas/<id>", entity_id=alpha_id)
        return _parse_json_dict(resp)

    def get_alpha_check(
        self,
        alpha_id: str,
        max_wait_s: Optional[float] = None,
    ) -> dict[str, Any]:
        """Poll GET /alphas/{alpha_id}/check until submission check results are ready."""
        ceiling = self.recordset_max_wait_s if max_wait_s is None else float(max_wait_s)
        resp = self.poll(f"/alphas/{alpha_id}/check", max_wait_s=ceiling)
        _handle_response(resp, context="GET /alphas/<id>/check", entity_id=alpha_id)
        return _parse_json_dict(resp)

    def get_alpha_correlations(
        self,
        alpha_id: str,
        kind: str,
        max_wait_s: Optional[float] = None,
    ) -> dict[str, Any]:
        """Poll GET /alphas/{alpha_id}/correlations/{kind} for self, prod, or power-pool."""
        if kind not in VALID_CORRELATION_KINDS:
            raise ValueError(
                f"Invalid correlation kind {kind!r}; expected one of {VALID_CORRELATION_KINDS}"
            )
        ceiling = self.recordset_max_wait_s if max_wait_s is None else float(max_wait_s)
        resp = self.poll(f"/alphas/{alpha_id}/correlations/{kind}", max_wait_s=ceiling)
        _handle_response(
            resp,
            context=f"GET /alphas/<id>/correlations/{kind}",
            entity_id=alpha_id,
        )
        if resp.status_code == 204:
            return {}
        return _parse_json_dict(resp)

    def fetch_recordset(
        self,
        alpha_id: str,
        name: str,
        max_wait_s: Optional[float] = None,
    ) -> dict[str, Any]:
        """Poll GET /alphas/{alpha_id}/recordsets/{name} until computed."""
        ceiling = self.recordset_max_wait_s if max_wait_s is None else float(max_wait_s)
        resp = self.poll(f"/alphas/{alpha_id}/recordsets/{name}", max_wait_s=ceiling)
        _handle_response(
            resp,
            context=f"GET /alphas/<id>/recordsets/{name}",
            entity_id=alpha_id,
        )
        if resp.status_code == 204:
            return {}
        return _parse_json_dict(resp)

    def patch_alpha_properties(
        self,
        alpha_id: str,
        color: Optional[str] = None,
        tags: Optional[list[str]] = None,
        name: Optional[str] = None,
        regular_desc: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update alpha metadata via PATCH /alphas/{alpha_id}."""
        payload: dict[str, Any] = {}
        if color is not None:
            payload["color"] = color
        if tags is not None:
            payload["tags"] = list(tags)
        if name is not None:
            payload["name"] = name
        if regular_desc is not None:
            payload["regular"] = {"description": regular_desc}

        resp = self._request("PATCH", f"/alphas/{alpha_id}", json=payload)
        _handle_response(resp, context="PATCH /alphas/<id>", entity_id=alpha_id)
        if resp.status_code == 204:
            return {}
        return _parse_json_dict(resp)

    def submit_alpha(self, alpha_id: str, max_wait_s: float = 900) -> dict[str, Any]:
        """Submit an alpha via POST /alphas/{alpha_id}/submit and poll until finished."""
        submit_path = f"/alphas/{alpha_id}/submit"
        resp = self._request("POST", submit_path)

        if resp.status_code == 409:
            body = _parse_json_safe(resp)
            logger.info("Alpha %s already submitted (HTTP 409).", alpha_id)
            return {
                "alpha_id": alpha_id,
                "status": "ALREADY_SUBMITTED",
                "already_submitted": True,
                "detail": body if body is not None else getattr(resp, "text", ""),
            }

        _handle_response(resp, context="POST /alphas/<id>/submit", entity_id=alpha_id)

        location = _get_header(getattr(resp, "headers", None), "Location")
        if location:
            poll_resp = self.poll(location, max_wait_s=max_wait_s)
            _handle_response(poll_resp, context="GET /alphas/<id>/submit", entity_id=alpha_id)
            return _parse_json_dict(poll_resp)

        if resp.status_code == 200:
            retry_raw = _get_header(getattr(resp, "headers", None), "Retry-After")
            body = _parse_json_safe(resp)
            if retry_raw is None and isinstance(body, dict) and body:
                return body

        poll_resp = self.poll(submit_path, max_wait_s=max_wait_s)
        _handle_response(poll_resp, context="GET /alphas/<id>/submit", entity_id=alpha_id)
        return _parse_json_dict(poll_resp)

    def list_user_alphas(
        self,
        status: str = "ACTIVE",
        region: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List authenticated user's alphas via GET /users/self/alphas."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if status:
            params["status"] = status
        if region:
            params["settings.region"] = region

        resp = self._request("GET", "/users/self/alphas", params=params)
        _handle_response(resp, context="GET /users/self/alphas")
        return _parse_json_dict(resp)

    def get_datasets(
        self,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Fetch paginated dataset catalog via GET /data-sets."""
        params: dict[str, Any] = {
            "instrumentType": instrument_type,
            "region": region,
            "delay": delay,
            "universe": universe,
            "limit": limit,
            "offset": offset,
        }
        resp = self._request("GET", "/data-sets", params=params)
        _handle_response(resp, context="GET /data-sets")
        return _parse_json_dict(resp)

    def get_datafields_page(
        self,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        dataset_id: str = "",
        limit: int = 50,
        offset: int = 0,
        data_type: Optional[str] = None,
        search: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch one page of data fields via GET /data-fields."""
        params: dict[str, Any] = {
            "instrumentType": instrument_type,
            "region": region,
            "delay": delay,
            "universe": universe,
            "limit": limit,
            "offset": offset,
        }
        if dataset_id:
            params["dataset.id"] = dataset_id
        if data_type and data_type != "ALL":
            params["type"] = data_type
        if search:
            params["search"] = search

        resp = self._request("GET", "/data-fields", params=params)
        _handle_response(resp, context="GET /data-fields")
        return _parse_json_dict(resp)

    def get_operators(self) -> list[dict[str, Any]]:
        """Fetch supported FastExpr operators via GET /operators."""
        resp = self._request("GET", "/operators")
        _handle_response(resp, context="GET /operators")
        parsed = _parse_json_safe(resp)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            results = parsed.get("results")
            if isinstance(results, list):
                return results
        return []


__all__ = [
    "AuthenticationError",
    "BiometricAuthTimeoutError",
    "BrainAPIError",
    "BrainClient",
    "MAX_MULTI_SIMULATIONS",
    "MIN_MULTI_SIMULATIONS",
    "PollingTimeoutError",
    "RateLimitError",
    "RecaptchaRequiredError",
    "SIMULATION_IN_PROGRESS",
    "SIMULATION_TERMINAL_FAILED",
    "SIMULATION_TERMINAL_OK",
    "SimulationError",
    "SimulationValidationError",
    "VALID_CORRELATION_KINDS",
    "WebAuthnRequiredError",
    "ensure_simulation_finished",
    "flatten_validation_errors",
    "parse_recordset",
    "simulation_id_from_url",
    "validate_simulation_payload",
]
