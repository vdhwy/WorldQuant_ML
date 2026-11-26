# Agent Instructions: BRAIN API Response Handling & Template Generator Fix

## Purpose

You are an AI coding agent working on a Python client for the WorldQuant BRAIN
simulation API (`https://api.worldquantbrain.com`). You have been given the
official API documentation (authentication, `/simulations`, `/alphas`,
`/alphas/<id>/recordsets`, `/users/<id>/activities/diversity`). Your job is
to:

1. Bring the system's API response-handling logic fully in line with the
   documented contract.
2. Fix the template generator, which currently hangs indefinitely when it
   reaches template index 9.

Read this whole document before touching any code. Locate the existing
codebase first — do not assume it doesn't exist. If no matching code is
found in the repo/workspace, build the response-handling module from
scratch using the spec in Part 1, and flag that no pre-existing template
generator was found before inventing one.

---

## Part 1 — API Response Logic to Implement

### 1.1 Authentication (`/authentication`)

- `GET` → `204` means not authenticated; `200` returns `user`, `token.expiry`,
  `permissions`. Treat `204` as "must re-authenticate," not as an error.
- `POST` → `201` on success, sets the `t` cookie (session library should
  persist cookies automatically). On `401`:
  - If body contains `"recaptcha": ["This field is required."]`, surface a
    distinct `RecaptchaRequiredError` rather than a generic auth failure.
  - Otherwise raise `AuthenticationError` with the `detail` code.
  - A `401` with a credential-create/GET-options body means WebAuthn is
    required — handle as a separate `WebAuthnRequiredError`, do not treat as
    plain invalid credentials.
- `DELETE` → `204` on success (note: doc text says "204 OK" for the delete
  response, treat any 2xx as success), `401` with `INVALID_CREDENTIALS` if
  session already invalid — treat that as a no-op, not a fatal error.

### 1.2 Simulation submission (`POST /simulations`)

- Support both a single simulation object and an array of 2–10 objects
  (multi-simulation). Validate array length client-side before sending.
- On `201 Created`, read the simulation id from the `Location` header
  (`/simulations/<id>`) — never parse it out of the response body.
- On `400 Bad Request`, the body is a nested validation-error object keyed
  by field name (and nested under `settings`). Flatten these into a single
  list of human-readable errors, e.g. `settings.region: This field is
  required.` Do not silently drop nested errors.

### 1.3 Polling (`GET /simulations/<id>`)

This is the most error-prone part of the system — implement it as its own
reusable poller, not inline per call site.

```python
def poll(url, session, max_wait_s=3600, default_interval_s=5):
    elapsed = 0
    while True:
        resp = session.get(url)
        resp.raise_for_status()
        retry_after = resp.headers.get("Retry-After")
        if not retry_after or float(retry_after) == 0:
            return resp  # finished — final state is in resp.json()
        wait = float(retry_after) or default_interval_s
        time.sleep(wait)
        elapsed += wait
        if elapsed > max_wait_s:
            raise TimeoutError(f"Polling {url} exceeded {max_wait_s}s")
```

Rules the response logic must enforce:

- **Never** treat "`Retry-After` absent" and "`Retry-After: 0`" differently
  from "simulation finished" — both mean stop polling.
- **Always** re-check `status` in the final JSON body, even after polling
  stops. `status` can be one of `WAITING, SIMULATING, CANCELLED, COMPLETE,
  WARNING, ERROR, TIMEOUT, FAIL`. Only `COMPLETE` / `WARNING` carry an
  `alpha` id — `ERROR`/`FAIL`/`TIMEOUT`/`CANCELLED` must raise a typed
  exception carrying `message` and, if present, the `location` object
  (`property`, `line`, `start`, `end`) so calling code can point at the
  exact offending line of alpha code.
- **Always** include a hard timeout / max-retry ceiling (see `max_wait_s`
  above). A poller with no ceiling is exactly the kind of bug that causes
  hangs — see Part 2.
- For multi-simulations, after the parent completes, fetch each id in
  `children[]` individually via the same poller; don't assume child status
  mirrors the parent.

### 1.4 Alpha & recordset retrieval

- `GET /alphas/<id>` — straightforward fetch, no polling needed once you
  have the id from a completed simulation.
- `GET /alphas/<id>/recordsets/<name>` **also returns `Retry-After`** while
  the recordset is being computed — reuse the same `poll()` helper, don't
  duplicate the wait loop (the sample code in the doc hand-rolls this twice;
  consolidate it to one function).
- Parse the `schema.properties[]` + `records[][]` shape generically: zip
  property names to each record row rather than hardcoding column order,
  since different record sets have different schemas. Use `type` per
  property to format `amount` (K/M/B suffix), `percent` (`%`), and
  `permyriad` (`‱`) values on the way out to any UI/report layer.

### 1.5 General response-handling conventions to apply system-wide

- Centralize status-code handling in one place (e.g. a `_handle_response`
  wrapper) instead of repeating `if resp.status_code == ...` at each call
  site — this is likely why "API response logic" drifted out of sync with
  the docs.
- Distinguish **transport/HTTP errors** (4xx/5xx via `raise_for_status`)
  from **domain errors** (a `200`/`201` whose body encodes failure, e.g.
  simulation `status: ERROR`). Both must be raised as exceptions, never
  returned as ordinary values, so callers can't accidentally treat a failed
  simulation as a successful one.
- Log the `alpha id` / `simulation id` on every error so failures are
  traceable back to a specific alpha.

---

## Part 2 — Fixing the Template Generator Hang at Template 9

You do not have the generator's source in front of you yet — locate it
(likely named something like `template_generator.py`, `generate_templates`,
or similar) before making changes. Once found, diagnose using this checklist
in order; template index 9 hanging (rather than 0 or a random index) is a
strong signal, not a coincidence:

1. **Off-by-one / boundary condition on the loop counter.** If templates are
   generated from a list/range and the stop condition compares against
   `len(templates)` computed once but the list is mutated in-loop (items
   appended/removed), the terminal condition can become permanently false
   around whatever index the mutation started at. Check whether index 9 is
   special (e.g. the 10th template, a page boundary, a batch-size boundary,
   or the first template requiring a network round-trip like a simulation
   poll).
2. **A blocking call with no timeout inside the loop body.** If template
   generation at step 9 triggers a simulation submission + poll (per Part 1),
   confirm that call site uses the poller with `max_wait_s` set — an
   unbounded `while True` around a `GET` that never returns a
   `Retry-After: 0` (e.g. because the simulation errored and the code only
   checks for the header, not `status`) will hang forever. This is the most
   likely root cause given the API doc you were given: the doc's own sample
   code (`while True: ... if ... == 0: break`) has no timeout and no status
   check — if the generator copies that pattern it will spin forever on any
   simulation that ends in `ERROR`/`FAIL` without ever un-setting
   `Retry-After`.
3. **Exception swallowed inside the loop.** A bare `except: continue` or
   `except: pass` around template 9's generation can mask an error that
   should have advanced or aborted the loop, leaving the loop variable
   unchanged and re-entering the same iteration.
4. **Recursive or retry logic without a base case.** If template failures
   trigger a retry/regenerate call that re-invokes the same function without
   decrementing a retry counter or without the counter being checked before
   the recursive call, template 9 (or whichever template is first to fail
   validation) recurses indefinitely.
5. **Shared mutable state across templates.** If a generator/index variable
   is captured by closure or is a class attribute reused across templates,
   confirm index 9 doesn't collide with a reset happening elsewhere (e.g. a
   cache keyed `0-8` with template 9 falling outside the precomputed range).

### Required fix pattern

Whichever root cause applies, the fix must include ALL of the following,
not just a patch to unblock index 9 specifically:

- A hard iteration/time ceiling on any loop that waits on an external
  response (reuse the `poll()` helper from Part 1.3).
- Explicit status/error checks before deciding to continue looping — never
  loop solely on "did I get a response," always also check "was the
  response a success."
- Remove any bare `except` in the generation loop; replace with catching
  specific exceptions, logging them with the template index, and either
  `continue`-ing to the next template or aborting with a clear error —
  never silently retrying the same index unbounded.
- A regression test (or, if no test suite exists, a manual repro script)
  that specifically generates ≥10 templates, including at least one that is
  forced to return a non-`COMPLETE` simulation status, to confirm the loop
  terminates and reports the failure instead of hanging.

---

## Deliverables

1. A response-handling module (or refactor of the existing one) matching
   Part 1, with the centralized `poll()`/`_handle_response` helpers used by
   every endpoint call site (`/authentication`, `/simulations`, `/alphas`,
   `/alphas/<id>/recordsets/<name>`, `/users/<id>/activities/diversity`).
2. A fix to the template generator per Part 2, with the specific root cause
   identified and stated in the commit message / PR description — not just
   "added timeout."
3. A short changelog summarizing: (a) which response-handling gaps existed
   vs. the documented API contract, (b) the confirmed root cause of the
   template-9 hang, (c) any remaining known limitations.

## Constraints

- Do not fabricate API behavior not present in the documentation above; if
  something is ambiguous (e.g. exact recovery behavior for `WARNING`
  status), state the assumption explicitly in code comments and the
  changelog rather than guessing silently.
- Keep retry/backoff behavior server-driven (`Retry-After`) rather than a
  fixed client-side interval, falling back to `default_interval_s` only
  when the header is missing but a retry is otherwise indicated.
