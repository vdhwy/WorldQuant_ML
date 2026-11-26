"""
Centralized WorldQuant BRAIN API response handling.

Implements the documented API contract for:
  - /authentication                (GET/POST/DELETE)
  - /simulations                   (POST submit, GET poll, multi-sim children)
  - /alphas/<id>                   (fetch)
  - /alphas/<id>/recordsets/<name> (poll + generic schema parsing)
  - /users/<id>/activities/diversity

Design rules enforced here (see docs / brain-api-agent-instructions.md):
  * Transport/HTTP errors (4xx/5xx) are raised via raise_for_status and are
    distinct from domain errors (a 200/201 body whose `status` field encodes
    failure, e.g. simulation status ERROR), which are raised as typed
    exceptions -- never returned as ordinary values.
  * Polling is server-driven (`Retry-After` header). A missing header and a
    `Retry-After: 0` header BOTH mean "finished". Every poll has a hard
    ceiling (max_wait_s) so a stuck simulation can never hang the caller.
  * The simulation id always comes from the `Location` response header,
    never from a response body.
  * Every error is logged with the alpha/simulation id so failures stay
    traceable to a specific alpha.

Assumption (documented): a final simulation `status` of WARNING is treated as
success-with-alpha (the platform returns an alpha id for COMPLETE/WARNING);
the exact recovery semantics for WARNING are not documented, so callers get
the payload plus a logged warning rather than special-cased behavior.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)

brain_api_url = "https://api.worldquantbrain.com"

# Documented simulation terminal states.
SIMULATION_TERMINAL_OK = ("COMPLETE", "WARNING")
SIMULATION_TERMINAL_FAILED = ("ERROR", "FAIL", "TIMEOUT", "CANCELLED")
SIMULATION_IN_PROGRESS = ("WAITING", "SIMULATING")

MAX_MULTI_SIMULATIONS = 10
MIN_MULTI_SIMULATIONS = 2


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BrainAPIError(Exception):
    """Base class for all BRAIN API errors."""


class AuthenticationError(BrainAPIError):
    """Generic authentication failure (401 from /authentication)."""

    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(detail or "Authentication failed")


class RecaptchaRequiredError(AuthenticationError):
    """401 whose body demands a recaptcha token (distinct from bad credentials)."""


class WebAuthnRequiredError(AuthenticationError):
    """401 whose body is a WebAuthn credential-create/options challenge."""

    def __init__(self, options: Any = None):
        self.options = options
        super().__init__("WebAuthn authentication required")


class SimulationValidationError(BrainAPIError):
    """400 Bad Request on POST /simulations with flattened validation errors."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("; ".join(errors) if errors else "Simulation validation failed")


class SimulationError(BrainAPIError):
    """Domain error: a simulation finished with a non-success `status`.

    Carries the API `message` and, when present, the `location` object
    (property/line/start/end) so calling code can point at the exact
    offending line of alpha code.
    """

    def __init__(
        self,
        status: str,
        message: str = "",
        location: Optional[Dict[str, Any]] = None,
        simulation_id: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ):
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
        super().__init__(f"Simulation {status}: {self.message}{loc}{suffix}")


class PollingTimeoutError(BrainAPIError, TimeoutError):
    """Polling exceeded its hard time ceiling."""


# ---------------------------------------------------------------------------
# Centralized HTTP response handling
# ---------------------------------------------------------------------------

def _handle_response(
    resp: requests.Response,
    context: str = "",
    entity_id: str = "",
) -> requests.Response:
    """Centralized transport-level status handling for every endpoint call.

    Raises requests.HTTPError for 4xx/5xx (transport errors). Domain errors
    carried in 2xx bodies are handled separately by ensure_simulation_finished.
    """
    if resp.status_code == 401:
        logger.error("401 Unauthorized %s %s", context, f"[{entity_id}]" if entity_id else "")
        resp.raise_for_status()
    if resp.status_code >= 400:
        logger.error(
            "HTTP %s on %s %s: %s",
            resp.status_code,
            context,
            f"[{entity_id}]" if entity_id else "",
            resp.text[:300],
        )
        resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Polling helper (single source of truth for every wait loop)
# ---------------------------------------------------------------------------

def poll(
    url: str,
    session: requests.Session,
    max_wait_s: float = 3600,
    default_interval_s: float = 5,
    max_http_retries: int = 0,
) -> requests.Response:
    """Reusable poller for endpoints that answer with `Retry-After`.

    Rules enforced:
      * Missing `Retry-After` header AND `Retry-After: 0` both mean the
        resource is finished -> return the final response. They are never
        distinguished from each other.
      * While a positive `Retry-After` is present, sleep exactly that long
        (server-driven backoff); fall back to default_interval_s only when a
        retry is indicated but the header value cannot be parsed.
      * Hard ceiling: raises PollingTimeoutError after max_wait_s of waiting.
      * Transient HTTP failures may be tolerated up to max_http_retries times;
        beyond that the underlying HTTPError propagates.
    """
    elapsed = 0.0
    http_failures = 0
    while True:
        try:
            resp = session.get(url)
            resp.raise_for_status()
            http_failures = 0
        except requests.RequestException:
            if http_failures < max_http_retries:
                http_failures += 1
                wait = default_interval_s
                logger.warning(
                    "Transient HTTP error polling %s (%s/%s), retrying in %ss",
                    url, http_failures, max_http_retries, wait,
                )
                time.sleep(wait)
                elapsed += wait
                if elapsed > max_wait_s:
                    raise PollingTimeoutError(f"Polling {url} exceeded {max_wait_s}s")
                continue
            raise

        retry_after_raw = resp.headers.get("Retry-After")
        if retry_after_raw is None:
            # Finished: a missing header means the final state is in this body.
            return resp

        try:
            wait = float(retry_after_raw)
        except (TypeError, ValueError):
            # A retry is indicated but the header value is unparseable:
            # fall back to the default interval instead of guessing "done".
            wait = default_interval_s
        else:
            if wait == 0:
                # Finished: explicit 0 is treated exactly like an absent header.
                return resp
            if wait < 0:
                wait = default_interval_s

        time.sleep(wait)
        elapsed += wait
        if elapsed > max_wait_s:
            raise PollingTimeoutError(f"Polling {url} exceeded {max_wait_s}s")


# ---------------------------------------------------------------------------
# /authentication
# ---------------------------------------------------------------------------

def check_authenticated(session: requests.Session) -> Optional[Dict[str, Any]]:
    """GET /authentication.

    Returns the user payload (user / token.expiry / permissions) when a 200 is
    received, or None when 204 signals "not authenticated". A 204 is a normal
    must-re-authenticate state, not an error.
    """
    resp = session.get(brain_api_url + "/authentication")
    _handle_response(resp, context="GET /authentication")
    if resp.status_code == 204:
        return None
    return resp.json()


def authenticate(session: requests.Session, email: str, password: str) -> requests.Response:
    """POST /authentication with basic credentials.

    On 201 the `t` cookie is persisted by the session automatically.
    Raises RecaptchaRequiredError / WebAuthnRequiredError / AuthenticationError
    for the documented 401 variants instead of a generic auth failure.
    """
    resp = session.post(brain_api_url + "/authentication", auth=(email, password))
    if resp.status_code == 201:
        return resp
    if resp.status_code == 401:
        body: Any = None
        try:
            body = resp.json()
        except ValueError:
            body = None

        # recaptcha variant: {"recaptcha": ["This field is required."]}
        if isinstance(body, dict) and isinstance(body.get("recaptcha"), list) and any(
            "required" in str(item) for item in body["recaptcha"]
        ):
            raise RecaptchaRequiredError()

        # WebAuthn variant: body describes credential creation options.
        if isinstance(body, dict) and (
            "publicKey" in body
            or "credentialOptions" in body
            or (isinstance(body.get("credentials"), dict) and "createOptions" in body["credentials"])
            or body.get("error") == "webauthn"
        ):
            raise WebAuthnRequiredError(options=body)

        detail = ""
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("code") or str(body)
        raise AuthenticationError(detail=str(detail))

    _handle_response(resp, context="POST /authentication")
    return resp


def logout(session: requests.Session) -> bool:
    """DELETE /authentication.

    Any 2xx is success (the doc's "204 OK"). A 401 INVALID_CREDENTIALS means
    the session was already invalid -- treat as a no-op returning False, not
    a fatal error.
    """
    resp = session.delete(brain_api_url + "/authentication")
    if 200 <= resp.status_code < 300:
        return True
    if resp.status_code == 401:
        logger.info("Logout skipped: session already invalid (INVALID_CREDENTIALS)")
        return False
    _handle_response(resp, context="DELETE /authentication")
    return False


# ---------------------------------------------------------------------------
# POST /simulations
# ---------------------------------------------------------------------------

def validate_simulation_payload(simulation_data: Union[dict, list]) -> None:
    """Client-side validation before sending to POST /simulations."""
    if isinstance(simulation_data, dict):
        return
    if isinstance(simulation_data, list):
        if not (MIN_MULTI_SIMULATIONS <= len(simulation_data) <= MAX_MULTI_SIMULATIONS):
            raise ValueError(
                f"A multi-simulation requires between {MIN_MULTI_SIMULATIONS} and "
                f"{MAX_MULTI_SIMULATIONS} simulation objects, got {len(simulation_data)}"
            )
        for i, item in enumerate(simulation_data):
            if not isinstance(item, dict):
                raise ValueError(f"Multi-simulation item {i} is not a dict")
        return
    raise ValueError("simulation_data must be a dict or a list of dicts")


def flatten_validation_errors(body: Any, prefix: str = "") -> List[str]:
    """Flatten a nested 400 validation-error body into human-readable strings.

    Example output: "settings.region: This field is required."
    Nested errors under `settings` (and any other nested mapping/list) are
    included -- nothing is silently dropped.
    """
    errors: List[str] = []
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
                errors.append(str(entry))
    elif isinstance(body, str):
        errors.append(body)
    return errors


def submit_simulation(
    session: requests.Session,
    simulation_data: Union[dict, list],
) -> str:
    """POST /simulations for a single object or an array of 2-10 objects.

    Returns the progress URL read from the `Location` header
    (`.../simulations/<id>`) -- the id is never parsed out of the body.
    Raises SimulationValidationError with flattened field errors on 400.
    """
    validate_simulation_payload(simulation_data)
    resp = session.post(brain_api_url + "/simulations", json=simulation_data)

    if resp.status_code == 400:
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        errors = flatten_validation_errors(body)
        logger.error("Simulation submission rejected (400): %s", errors)
        raise SimulationValidationError(errors)

    _handle_response(resp, context="POST /simulations")

    location = resp.headers.get("Location")
    if not location and 200 <= resp.status_code < 300 and isinstance(simulation_data, list) is False:
        # Some gateways echo a Location-style body; still prefer headers only.
        location = None
    if not location:
        raise BrainAPIError(
            f"Simulation accepted ({resp.status_code}) but no Location header returned"
        )
    return location


def simulation_id_from_url(url: str) -> str:
    """Extract `<id>` from a `/simulations/<id>` Location URL."""
    return url.rstrip("/").rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Domain-level simulation status checks (transport vs domain errors)
# ---------------------------------------------------------------------------

def ensure_simulation_finished(
    payload: Dict[str, Any],
    simulation_id: str = "",
) -> Dict[str, Any]:
    """Re-check `status` in a finished simulation's JSON body.

    Only COMPLETE/WARNING carry an `alpha` id and are returned. Terminal
    failure states raise SimulationError carrying `message` and the `location`
    object when present. Non-terminal states at this point mean the poller
    returned without a Retry-After but work is still pending -- raised as
    SimulationError so callers can never mistake it for success.
    """
    status = payload.get("status", "")
    sim_id = simulation_id or str(payload.get("id", ""))

    if status in SIMULATION_TERMINAL_FAILED:
        raise SimulationError(
            status=status,
            message=payload.get("message", ""),
            location=payload.get("location"),
            simulation_id=sim_id,
            payload=payload,
        )
    if status not in SIMULATION_TERMINAL_OK:
        # WAITING/SIMULATING (or unknown) without a Retry-After header.
        raise SimulationError(
            status=status or "UNKNOWN",
            message=f"Simulation did not reach a terminal state (status={status!r})",
            location=payload.get("location"),
            simulation_id=sim_id,
            payload=payload,
        )
    if status == "WARNING":
        # Assumption: WARNING behaves like COMPLETE and carries an alpha id;
        # recovery semantics beyond that are undocumented, so just log it.
        logger.warning(
            "Simulation %s finished with WARNING status%s",
            sim_id,
            f": {payload.get('message')}" if payload.get("message") else "",
        )
    return payload


def poll_simulation(
    session: requests.Session,
    location_url: str,
    max_wait_s: float = 3600,
    default_interval_s: float = 5,
    max_http_retries: int = 3,
) -> Dict[str, Any]:
    """Poll a simulation to completion and return its validated payload.

    Combines poll() with the mandatory post-poll status check, so a failed
    simulation surfaces as SimulationError instead of a success-shaped value.
    """
    sim_id = simulation_id_from_url(location_url)
    resp = poll(
        location_url,
        session,
        max_wait_s=max_wait_s,
        default_interval_s=default_interval_s,
        max_http_retries=max_http_retries,
    )
    payload = resp.json()
    return ensure_simulation_finished(payload, simulation_id=sim_id)


def poll_simulation_children(
    session: requests.Session,
    parent_payload: Dict[str, Any],
    max_wait_s: float = 3600,
) -> List[Dict[str, Any]]:
    """After a multi-simulation parent completes, poll each child individually.

    Child status never mirrors the parent, so every child goes through the
    same poller and status check.
    """
    children = parent_payload.get("children") or []
    results: List[Dict[str, Any]] = []
    for child_id in children:
        child_url = child_id if str(child_id).startswith("http") else f"{brain_api_url}/simulations/{child_id}"
        results.append(
            poll_simulation(session, child_url, max_wait_s=max_wait_s)
        )
    return results


# ---------------------------------------------------------------------------
# /alphas/<id> and recordsets
# ---------------------------------------------------------------------------

def get_alpha(session: requests.Session, alpha_id: str) -> Dict[str, Any]:
    """Straightforward fetch of an alpha -- no polling needed once completed."""
    resp = session.get(f"{brain_api_url}/alphas/{alpha_id}")
    _handle_response(resp, context="GET /alphas/<id>", entity_id=alpha_id)
    return resp.json()


def fetch_recordset(
    session: requests.Session,
    alpha_id: str,
    name: str,
    max_wait_s: float = 600,
    default_interval_s: float = 5,
) -> Dict[str, Any]:
    """Fetch /alphas/<id>/recordsets/<name>, reusing poll().

    Recordsets also answer with `Retry-While`/`Retry-After` while being
    computed, so they go through the same single wait loop as simulations.
    """
    url = f"{brain_api_url}/alphas/{alpha_id}/recordsets/{name}"
    resp = poll(
        url,
        session,
        max_wait_s=max_wait_s,
        default_interval_s=default_interval_s,
        max_http_retries=3,
    )
    _handle_response(resp, context=f"GET recordsets/{name}", entity_id=alpha_id)
    return resp.json()


def parse_recordset(recordset_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the generic schema.properties[] + records[][] shape.

    Zips property names onto each record row rather than hardcoding column
    order, since different record sets have different schemas.
    """
    properties = (recordset_payload.get("schema") or {}).get("properties") or []
    names = [p.get("name") for p in properties]
    records = recordset_payload.get("records") or []
    rows = [dict(zip(names, record)) for record in records]
    return {
        "names": names,
        "properties": properties,
        "records": records,
        "rows": rows,
        "min": recordset_payload.get("min"),
        "max": recordset_payload.get("max"),
    }


def format_value(value: Any, prop_type: str) -> str:
    """Format a recordset cell for UI/report layers based on its schema type.

    amount -> K/M/B suffixes, percent -> '%', permyriad -> '‱'.
    """
    prop_type = (prop_type or "").lower()
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if prop_type == "percent":
        return f"{numeric:.4f}%"
    if prop_type == "permyriad":
        return f"{numeric:.4f}\u2031"
    if prop_type == "amount":
        abs_val = abs(numeric)
        sign = "-" if numeric < 0 else ""
        if abs_val >= 1_000_000_000:
            return f"{sign}{abs_val / 1_000_000_000:.2f}B"
        if abs_val >= 1_000_000:
            return f"{sign}{abs_val / 1_000_000:.2f}M"
        if abs_val >= 1_000:
            return f"{sign}{abs_val / 1_000:.2f}K"
        return f"{sign}{abs_val:.2f}"
    return str(value)


def format_recordset_rows(recordset_payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convenience: parsed rows with each cell formatted per its schema type."""
    parsed = parse_recordset(recordset_payload)
    type_by_name = {p.get("name"): p.get("type") for p in parsed["properties"]}
    formatted: List[Dict[str, str]] = []
    for row in parsed["rows"]:
        formatted.append(
            {key: format_value(value, type_by_name.get(key)) for key, value in row.items()}
        )
    return formatted


# ---------------------------------------------------------------------------
# /users/<id>/activities/diversity
# ---------------------------------------------------------------------------

def get_user_diversity(session: requests.Session, user_id: str) -> Dict[str, Any]:
    """GET /users/<id>/activities/diversity through the shared response handler."""
    resp = session.get(f"{brain_api_url}/users/{user_id}/activities/diversity")
    _handle_response(resp, context="GET diversity", entity_id=user_id)
    return resp.json()
