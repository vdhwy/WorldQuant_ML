"""
Regression tests for the BRAIN API response-handling contract.

Focus points:
  * poll(): absent Retry-After header and Retry-After: 0 must both be treated
    as "finished"; every wait has a hard ceiling (no unbounded loops).
  * Domain errors: a finished simulation whose `status` is ERROR/FAIL/TIMEOUT/
    CANCELLED raises a typed SimulationError carrying message + location.
  * POST /simulations: multi-simulation array length validated client-side;
    id read from the Location header, never the body.
  * Template-batch regression (the historical "hangs at template index 9"
    bug): generating >=10 templates where one simulation is forced to end in
    a non-COMPLETE status must terminate and report the failure instead of
    hanging forever.
"""

import sys
import time
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.brain_api as brain_api  # noqa: E402
from core.brain_api import (  # noqa: E402
    PollingTimeoutError,
    RecaptchaRequiredError,
    SimulationError,
    SimulationValidationError,
    WebAuthnRequiredError,
    authenticate,
    ensure_simulation_finished,
    flatten_validation_errors,
    format_recordset_rows,
    format_value,
    parse_recordset,
    poll,
    poll_simulation,
    poll_simulation_children,
    submit_simulation,
    validate_simulation_payload,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, json_data=None, status_code=200, headers=None, text=""):
        self._json = json_data if json_data is not None else {}
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class ScriptedSession:
    """Fake requests.Session answering each URL from a script.

    `script` maps url -> list of responses returned in order; the last entry
    is repeated once exhausted. `sleeps` records every sleep duration so tests
    can assert server-driven backoff without actually waiting.
    """

    def __init__(self, script):
        self.script = {url: list(resps) for url, resps in script.items()}
        self.get_calls = []
        self.post_calls = []
        self.sleeps = []

    def get(self, url, **kwargs):
        self.get_calls.append(url)
        responses = self.script[url]
        if len(responses) > 1:
            return responses.pop(0)
        return responses[0]

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        responses = self.script[url]
        return responses.pop(0)

    def delete(self, url, **kwargs):  # pragma: no cover - used by logout tests
        return self.script[url].pop(0)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Never actually sleep during tests; record durations instead."""
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    yield sleeps


def waiting_response(retry_after="5", sim_id="sim-1"):
    return FakeResponse(
        {"id": sim_id, "status": "WAITING"},
        headers={"Retry-After": retry_after},
    )


# ---------------------------------------------------------------------------
# poll() semantics
# ---------------------------------------------------------------------------

def test_poll_missing_retry_after_header_is_final():
    session = ScriptedSession({
        "http://x/simulations/1": [
            waiting_response("5"),
            FakeResponse({"id": "sim-1", "status": "COMPLETE", "alpha": "abc"}),
        ]
    })
    resp = poll("http://x/simulations/1", session)
    assert resp.json()["status"] == "COMPLETE"


def test_poll_retry_after_zero_is_final__regression_string_zero():
    # Regression: the old code compared the STRING header ("0") against int 0,
    # which is never equal, so a finished simulation reporting Retry-After: 0
    # was polled forever (tight sleep(0) loop). Both spellings must terminate.
    for zero in ["0", 0]:
        session = ScriptedSession({
            "http://x/simulations/1": [
                FakeResponse({"id": "sim-1", "status": "COMPLETE", "alpha": "abc"},
                             headers={"Retry-After": zero}),
            ]
        })
        resp = poll("http://x/simulations/1", session)
        assert resp.json()["status"] == "COMPLETE"


def test_poll_absent_and_zero_headers_treated_identically():
    with_header = ScriptedSession({
        "u": [FakeResponse({"status": "COMPLETE"}, headers={"Retry-After": "0"})]
    })
    without_header = ScriptedSession({
        "u": [FakeResponse({"status": "COMPLETE"})]
    })
    assert poll("u", with_header).json() == poll("u", without_header).json()


def test_poll_is_server_driven_not_fixed_interval(no_real_sleep):
    session = ScriptedSession({
        "http://x/simulations/1": [
            waiting_response("7"),
            waiting_response("13"),
            FakeResponse({"status": "COMPLETE"}),
        ]
    })
    poll("http://x/simulations/1", session)
    assert no_real_sleep == [7.0, 13.0]


def test_poll_has_hard_ceiling(no_real_sleep):
    session = ScriptedSession({
        "http://x/simulations/stuck": [waiting_response("5")]
    })
    with pytest.raises(PollingTimeoutError):
        poll("http://x/simulations/stuck", session, max_wait_s=10)
    # It kept polling until the ceiling, then stopped.
    assert len(session.get_calls) == 3


def test_poll_unparseable_header_falls_back_to_default_interval(no_real_sleep):
    session = ScriptedSession({
        "u": [
            FakeResponse({"status": "WAITING"}, headers={"Retry-After": "soon"}),
            FakeResponse({"status": "COMPLETE"}),
        ]
    })
    poll("u", session, default_interval_s=3)
    assert no_real_sleep == [3.0]


# ---------------------------------------------------------------------------
# Domain-level status checks (transport vs domain errors)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["ERROR", "FAIL", "TIMEOUT", "CANCELLED"])
def test_terminal_failure_statuses_raise_typed_error(status):
    payload = {
        "id": "sim-9",
        "status": status,
        "message": "Bad expression",
        "location": {"property": "regular", "line": 2, "start": 4, "end": 9},
    }
    with pytest.raises(SimulationError) as excinfo:
        ensure_simulation_finished(payload, simulation_id="sim-9")
    err = excinfo.value
    assert err.status == status
    assert err.message == "Bad expression"
    assert err.location["line"] == 2
    assert "sim-9" in str(err)
    assert "line" in str(err)


def test_nonterminal_status_without_retry_after_never_looks_successful():
    with pytest.raises(SimulationError):
        ensure_simulation_finished({"id": "s", "status": "SIMULATING"})


@pytest.mark.parametrize("status", ["COMPLETE", "WARNING"])
def test_complete_and_warning_are_success(status):
    payload = {"id": "s", "status": status, "alpha": "abc"}
    assert ensure_simulation_finished(payload)["alpha"] == "abc"


def test_poll_simulation_combines_poll_and_status_check(no_real_sleep):
    session = ScriptedSession({
        "http://x/simulations/sim-err": [
            waiting_response(),
            FakeResponse({"id": "sim-err", "status": "ERROR", "message": "boom"}),
        ]
    })
    with pytest.raises(SimulationError):
        poll_simulation(session, "http://x/simulations/sim-err")


# ---------------------------------------------------------------------------
# POST /simulations
# ---------------------------------------------------------------------------

def test_multi_simulation_length_validated_client_side():
    validate_simulation_payload({"type": "REGULAR"})
    validate_simulation_payload([{"t": i} for i in range(2)])
    validate_simulation_payload([{"t": i} for i in range(10)])
    for bad_len in (0, 1, 11):
        with pytest.raises(ValueError):
            validate_simulation_payload([{"t": i} for i in range(bad_len)])
    with pytest.raises(ValueError):
        validate_simulation_payload("not-a-dict")


def test_submit_simulation_reads_location_header_not_body():
    session = ScriptedSession({})
    session.post = lambda url, **kw: FakeResponse(
        {"id": "ignored-body-id"}, status_code=201,
        headers={"Location": "/simulations/sim-from-header"},
    )
    url = submit_simulation(session, {"type": "REGULAR"})
    assert url == "/simulations/sim-from-header"


def test_submit_simulation_flattens_nested_400_errors():
    nested = {
        "settings": {
            "region": ["This field is required."],
            "decay": ["A valid integer is required."],
        },
        "regular": ["This field may not be blank."],
    }
    session = ScriptedSession({})
    session.post = lambda url, **kw: FakeResponse(nested, status_code=400)

    with pytest.raises(SimulationValidationError) as excinfo:
        submit_simulation(session, {"type": "REGULAR"})
    flattened = excinfo.value.errors
    assert "settings.region: This field is required." in flattened
    assert "settings.decay: A valid integer is required." in flattened
    assert "regular: This field may not be blank." in flattened


def test_flatten_validation_errors_drops_nothing_nested():
    body = {"a": {"b": {"c": ["deep"]}}, "d": [{"e": ["list-dict"]}], "f": "plain"}
    flat = flatten_validation_errors(body)
    assert sorted(flat) == [
        "a.b.c: deep",
        "d.e: list-dict",
        "f: plain",
    ]


# ---------------------------------------------------------------------------
# Authentication variants
# ---------------------------------------------------------------------------

def test_authenticate_surfaces_recaptcha_distinctly():
    session = ScriptedSession({})
    session.post = lambda url, **kw: FakeResponse(
        {"recaptcha": ["This field is required."]}, status_code=401)
    with pytest.raises(RecaptchaRequiredError):
        authenticate(session, "e", "p")


def test_authenticate_surfaces_webauthn_distinctly():
    session = ScriptedSession({})
    session.post = lambda url, **kw: FakeResponse(
        {"credentials": {"createOptions": {"challenge": "xyz"}}}, status_code=401)
    with pytest.raises(WebAuthnRequiredError):
        authenticate(session, "e", "p")


def test_authenticate_generic_401_uses_detail_code():
    session = ScriptedSession({})
    session.post = lambda url, **kw: FakeResponse({"detail": "INVALID_CREDENTIALS"}, status_code=401)
    from core.brain_api import AuthenticationError
    with pytest.raises(AuthenticationError) as excinfo:
        authenticate(session, "e", "p")
    assert "INVALID_CREDENTIALS" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Recordsets
# ---------------------------------------------------------------------------

def test_parse_recordset_zips_properties_to_records_generically():
    payload = {
        "schema": {"properties": [{"name": "date", "type": "day"},
                                  {"name": "pnl", "type": "amount"}]},
        "records": [["2024-01-02", 1234567], ["2024-01-03", -2500]],
    }
    parsed = parse_recordset(payload)
    assert parsed["rows"][0] == {"date": "2024-01-02", "pnl": 1234567}
    assert parsed["rows"][1]["pnl"] == -2500


def test_format_value_by_schema_type():
    assert format_value(1500000, "amount") == "1.50M"
    assert format_value(980, "amount") == "980.00"
    assert format_value(-2300000000, "amount") == "-2.30B"
    assert format_value(0.0321, "percent").endswith("%")
    assert format_value(12.5, "permyriad").endswith("\u2031")


def test_format_recordset_rows_end_to_end():
    payload = {
        "schema": {"properties": [{"name": "turnover", "type": "percent"}]},
        "records": [[0.41]],
    }
    rows = format_recordset_rows(payload)
    assert rows == [{"turnover": "0.4100%"}]


# ---------------------------------------------------------------------------
# Root-cause documentation for the historical "hangs at template index 9" bug
# ---------------------------------------------------------------------------

def test_old_buggy_condition_documents_root_cause():
    """Encodes the exact defect that caused the template-9 hang.

    `headers.get("Retry-After", 0)` returns the STRING "0" when the header is
    present. The old break condition compared it against the int 0:

        resp.headers.get("Retry-After", 0) == 0

    In Python, "0" == 0 is False, so a finished simulation that reported
    `Retry-After: 0` never satisfied the terminal condition, and with no time
    ceiling the loop spun forever (time.sleep(0) tight loop). Multi-simulation
    batches contain 10 templates (indices 0..9), so generation appeared to
    hang exactly when the first full batch of ten reached completion --
    i.e., at template index 9.
    """
    headers = {"Retry-After": "0"}
    assert not (headers.get("Retry-After", 0) == 0)  # old condition never fired
    assert float(headers.get("Retry-After", 0)) == 0  # correct comparison


# ---------------------------------------------------------------------------
# Template-batch regression: >=10 templates, one forced non-COMPLETE
# ---------------------------------------------------------------------------

def _batch_script(num_templates=10, failing_index=None):
    """Build parent+children payloads for a multi-simulation of templates.

    Mirrors simulate_alpha_list_multi's default batching: 10 templates per
    batch. Each child maps to template index via 'child-<i>'.
    """
    children_ids = [f"child-{i}" for i in range(num_templates)]
    script = {}
    for i, child in enumerate(children_ids):
        child_url = f"{brain_api.brain_api_url}/simulations/{child}"
        if failing_index is not None and i == failing_index:
            script[child_url] = [FakeResponse({
                "id": child,
                "status": "ERROR",
                "message": "Invalid syntax at token",
                "location": {"property": "regular", "line": 3, "start": 0, "end": 8},
            }, headers={"Retry-After": "0"})]
        else:
            script[child_url] = [FakeResponse({
                "id": child,
                "status": "COMPLETE",
                "alpha": f"alpha-{i}",
            })]
    return children_ids, script


def test_ten_template_batch_terminates_when_template_9_errors():
    """THE regression: batch of 10 templates, index 9 forced to ERROR.

    The old code hung here: Retry-After "0" != 0 (str vs int) meant the
    completion condition could never fire, and there was no time ceiling --
    generation stalled indefinitely at template index 9. The loop must now
    terminate and raise/report the failure.
    """
    children_ids, script = _batch_script(10, failing_index=9)
    session = ScriptedSession(script)

    started = time.monotonic()
    with pytest.raises(SimulationError) as excinfo:
        poll_simulation_children(
            session, {"children": children_ids}, max_wait_s=60
        )
    duration = time.monotonic() - started

    assert duration < 30, "batch processing must not hang"
    assert "child-9" in str(excinfo.value.simulation_id)
    assert excinfo.value.location["line"] == 3


def test_ten_template_batch_all_succeed_and_children_polled_individually():
    children_ids, script = _batch_script(10)
    session = ScriptedSession(script)

    results = poll_simulation_children(session, {"children": children_ids})
    assert len(results) == 10
    assert all(r["status"] == "COMPLETE" for r in results)
    # Every child was fetched individually -- child status does NOT mirror
    # the parent's, so a single GET per child is not enough.
    fetched = {url.rsplit("/", 1)[-1] for url in session.get_calls}
    assert fetched == set(children_ids)


# ---------------------------------------------------------------------------
# lib.py integration-level regression (uses lib.simulation_progress flow)
# ---------------------------------------------------------------------------

def test_lib_simulation_progress_returns_on_retry_after_zero_final():
    """End-to-end hang repro through lib.simulation_progress.

    The final poll response carries `Retry-After: 0` (string) with COMPLETE
    status. Old code spun forever on this exact response shape.
    """
    import lib

    session = ScriptedSession({
        f"{brain_api.brain_api_url}/simulations/sim-1": [
            waiting_response("5"),
            waiting_response("5"),
            FakeResponse({"id": "sim-1", "status": "COMPLETE", "alpha": "alpha-1"},
                         headers={"Retry-After": "0"}),
        ],
        f"{brain_api.brain_api_url}/alphas/alpha-1": [
            FakeResponse({"id": "alpha-1", "status": "COMPLETE", "is": {"sharpe": 1.5}}),
        ],
    })

    start_resp = FakeResponse(
        {}, status_code=201,
        headers={"Location": f"{brain_api.brain_api_url}/simulations/sim-1"},
    )
    outcome = lib.simulation_progress(session, start_resp)

    assert outcome["completed"] is True
    assert outcome["result"]["id"] == "alpha-1"


def test_lib_simulation_progress_reports_error_without_hanging():
    import lib

    session = ScriptedSession({
        f"{brain_api.brain_api_url}/simulations/sim-err": [
            waiting_response("5"),
            FakeResponse({
                "id": "sim-err", "status": "ERROR", "message": "compile failed",
                "location": {"property": "regular", "line": 1},
            }),
        ]
    })

    start_resp = FakeResponse(
        {}, status_code=201,
        headers={"Location": f"{brain_api.brain_api_url}/simulations/sim-err"},
    )
    outcome = lib.simulation_progress(session, start_resp)

    assert outcome == {"completed": False, "result": {}}


def test_lib_start_simulation_rejects_bad_multi_sim_length_before_http():
    import lib

    session = ScriptedSession({})

    with pytest.raises(ValueError):
        lib.start_simulation(session, [dict()] )  # 1-element array is invalid
    with pytest.raises(ValueError):
        lib.start_simulation(session, [dict() for _ in range(11)])
    # Nothing was sent over the wire.
    assert session.post_calls == []


def test_lib_multisimulation_progress_polls_children():
    import lib

    children_ids, script = _batch_script(10)
    # Parent completes with all 10 children attached.
    script[f"{brain_api.brain_api_url}/simulations/parent"] = [FakeResponse({
        "id": "parent",
        "status": "COMPLETE",
        "children": children_ids,
    })]
    for i in range(10):
        alpha_id = f"alpha-{i}"
        script[f"{brain_api.brain_api_url}/alphas/{alpha_id}"] = [
            FakeResponse({"id": alpha_id, "is": {"sharpe": 1.0}})
        ]
    session = ScriptedSession(script)

    start_resp = FakeResponse(
        {}, status_code=201,
        headers={"Location": f"{brain_api.brain_api_url}/simulations/parent"},
    )
    outcome = lib.multisimulation_progress(session, start_resp)

    assert outcome["completed"] is True
    assert len(outcome["result"]) == 10
