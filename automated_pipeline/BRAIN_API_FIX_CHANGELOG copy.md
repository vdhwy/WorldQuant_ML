# Changelog — BRAIN API Response Handling & Template Generator Fix

## (a) Response-handling gaps vs. the documented API contract

| # | Gap found in the codebase | Contract requirement | Fix |
|---|---------------------------|----------------------|-----|
| 1 | Every wait loop (`simulation_progress`, `multisimulation_progress`, `get_simulation_result_json`, `get_prod_corr`, `get_self_corr`, `get_check_submission`, `_get_alpha_pnl`, `get_alpha_yearly_stats`, `performance_comparison` in `lib.py`) was a hand-rolled `while True` with **no time ceiling**. | Polling must always have a hard `max_wait_s` ceiling. | New shared poller `poll()` in `core/brain_api.py`; all call sites refactored onto it (`DEFAULT_MAX_WAIT_S = 3600`). |
| 2 | Completion test compared a **string** header to an int: `resp.headers.get("Retry-After", 0) == 0` → `"0" == 0` is `False`. A finished simulation reporting `Retry-After: 0` never satisfied the terminal condition; only an *absent* header broke the loop. | Absent header and `Retry-After: 0` must be treated **identically** as "finished". | `poll()` parses defensively: absent → finished, `"0"` → finished, unparseable/negative → fall back to `default_interval_s` and keep polling (bounded by the ceiling). |
| 3 | Sleep intervals were taken from `Retry-After` where present but some loops used fixed client-side sleeps (e.g. `time.sleep(30)` after HTTP errors, fixed 5 s in `SimulatorTester.monitor_simulation`) with no ceiling interplay. | Backoff must be server-driven; default interval only when a retry is indicated but no usable header exists. | `poll()` sleeps exactly the server-provided value; `monitor_simulation` now honors `Retry-After` with a 5 s fallback. Transient HTTP failures tolerated via `max_http_retries` (preserves old retry-tolerance) inside the same ceiling. |
| 4 | After polling stopped, body `status` was checked only for `ERROR`. `FAIL`, `TIMEOUT`, `CANCELLED` fell through into the success path (they just happened to lack an `alpha` id); `WARNING` semantics were unhandled. | Always re-check `status`: only `COMPLETE`/`WARNING` carry an alpha id; failure states raise a typed error carrying `message` and the `location` object (`property`, `line`, `start`, `end`). | `ensure_simulation_finished()` raises `SimulationError(status, message, location, simulation_id)` for `ERROR/FAIL/TIMEOUT/CANCELLED` *and* for non-terminal states reached without `Retry-After`. `SimulatorTester.monitor_simulation` handles all documented statuses; WARNING is treated as success-with-alpha (assumption noted in code). |
| 5 | Multi-simulation children were fetched with a single non-polled `GET` each; child status was assumed to mirror the parent's completion. Also a missing `continue` meant a non-2xx parent poll fell through and parsed headers/JSON of the error response (and could crash on `children == 0` int). | Each child in `children[]` must go through the same poller + status check. | `poll_simulation_children()` polls every child individually; `multisimulation_progress` refactored accordingly. |
| 6 | No client-side validation of multi-simulation array length; 400 validation bodies were logged as opaque text. | Validate array length 2–10 before sending; flatten nested 400 errors (`settings.region: This field is required.`), dropping nothing. | `validate_simulation_payload()` (raises `ValueError` before any HTTP call) and `flatten_validation_errors()`; `start_simulation`/`submit_simulation` raise `SimulationValidationError.errors`. Simulation id always read from the `Location` header, never the body. |
| 7 | `/authentication` handling treated all 401s as "wrong password"; no distinct recaptcha/WebAuthn handling; `204` on GET conflated with errors by callers; DELETE success/failure semantics ad hoc. | Distinct `RecaptchaRequiredError` / `WebAuthnRequiredError`; GET 204 = "must re-authenticate", not an error; DELETE: any 2xx = success, `401 INVALID_CREDENTIALS` = no-op. | `check_authenticated()`, `authenticate()`, `logout()` in `core/brain_api.py`. |
| 8 | Recordset parsing hardcoded column order per call site; no type-based formatting. | Zip `schema.properties[]` names to `records[][]` rows generically; format `amount` (K/M/B), `percent` (%), `permyriad` (‱). | `fetch_recordset()` (reuses `poll()`), `parse_recordset()`, `format_value()`, `format_recordset_rows()`. |
| 9 | Status-code handling duplicated at ~10 call sites; domain failures returned as ordinary dicts (`{"completed": False}`) indistinguishable from transport errors at the call site; errors did not consistently log alpha/simulation ids. | Centralize HTTP handling; distinguish transport vs domain errors; log entity ids on every error. | `_handle_response()` + typed exception hierarchy (`BrainAPIError`, `AuthenticationError`, `RecaptchaRequiredError`, `WebAuthnRequiredError`, `SimulationValidationError`, `SimulationError`, `PollingTimeoutError`); ids included in messages/logs. |

## (b) Confirmed root cause of the template-9 hang

**File:** `lib.py` → `simulation_progress()` / `multisimulation_progress()` (the wait loop
used by the generate→simulate pipeline that runs batches of templates).

```python
if simulation_progress_response.headers.get("Retry-After", 0) == 0:
    ... break ...
```

`headers.get("Retry-After", 0)` returns the **string** `"0"` when the server sends the
header. In Python `"0" == 0` is `False`, so the intended terminal condition could never
fire for a finished simulation reporting `Retry-After: 0` — the loop executed
`time.sleep(float("0"))` and re-polled forever in a tight loop. There was also **no
maximum-wait ceiling**, so even a stuck `WAITING` simulation would hang indefinitely.

**Why index 9 specifically:** multi-simulations are submitted in batches of up to
10 templates (`simulate_alpha_list_multi`, `limit_of_multi_simulations=10`; batch =
indices 0–9). The batch-level parent poll completes only once the whole batch finishes,
so the first full batch of ten reaching its final `Retry-After: 0` response stalls the
pipeline exactly at template index 9. It is a batch-size boundary, not a coincidence.

Secondary contributors removed at the same time:
- `multisimulation_progress` slept 30 s on a non-2xx poll but **did not `continue`**, then parsed the error response's headers/JSON anyway.
- Child simulations received a single unpollled `GET` (status assumed to mirror parent).
- Bare `except:` handlers around prompt-building in `core/template_generator.py` could mask generation errors (replaced with specific catches + logging).

## Changes made

- **New:** `core/brain_api.py` — centralized response-handling module: typed exceptions, `_handle_response`, `poll()`, auth helpers, simulation submit/poll/children, recordset parse/format, diversity endpoint.
- **Refactored:** `lib.py` — all nine unbounded wait loops now use `poll()`; status re-checks via `ensure_simulation_finished()`; multi-sim children polled individually; client-side payload validation; flattened 400 errors.
- **Fixed:** `core/simulator_tester.py` — server-driven backoff, all documented statuses handled (`FAIL/TIMEOUT/CANCELLED/WARNING`), location info surfaced.
- **Fixed:** `core/template_generator.py` bare `except:` blocks; GUI canvas guards in `step4_generation.py` / `step5_simulation.py` catch `tk.TclError` specifically.
- **Resilient:** `core/__init__.py` no longer explodes under flat-layout imports (guarded re-exports), enabling `from core.brain_api import ...`.
- **Tests:** `tests/test_brain_api_polling.py` (31 tests) — includes the required regression: a ≥10-template batch with one forced non-`COMPLETE` termination terminates and reports the failure instead of hanging, plus end-to-end hang repros through `lib.simulation_progress`.

## (c) Remaining known limitations / assumptions

1. **WARNING semantics:** the docs don't specify recovery behavior for `status: WARNING`. Assumption (stated in code): WARNING behaves like COMPLETE and carries an alpha id; it is surfaced as success with a log entry, and check-level warnings still mark results red in `SimulatorTester` (existing v2 behavior preserved).
2. **Non-terminal state without `Retry-After`:** treated as an error (`SimulationError`) rather than retried, since the server gives no wait hint; callers see a clear failure instead of guessing.
3. **Biometric/persona login** (`lib.start_session`) still waits on human approval without a hard timeout by design (user-interactive flow, not an API-polling path).
4. **`max_http_retries` tolerance:** transient HTTP errors during polling are retried within the same `max_wait_s` budget; persistent failures raise the underlying `requests.HTTPError` (transport errors stay distinct from domain errors).
5. The legacy `RequestHandler`/`RetryHandler` in `core/utils/` remains untouched (used by other tooling); the new module does not depend on it.
6. Tests avoid real network/sleeps entirely; they encode the documented contract, so any future deviation from actual API behavior (e.g. header spelling) should be reflected there.
