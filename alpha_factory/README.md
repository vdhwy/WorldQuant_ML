# Alpha Factory

Automated WorldQuant BRAIN alpha generation, multi-simulation execution, qualification, local PnL deduplication, and submission staging pipeline.

## Architecture Overview

```text
alpha_factory/
├── config/
│   ├── settings.yaml        # Runtime configuration (merged over DEFAULT_SETTINGS)
│   └── templates.yaml       # FastExpr templates for expansion
├── brain/
│   ├── auth.py              # Session auth, ~/secrets cache, biometric 2FA polling
│   ├── client.py            # BRAIN REST client with retries & Retry-After polling
│   └── catalog.py           # Operator, dataset, and datafield sync + offline cache bootstrap
├── storage/
│   ├── models.py            # Dataclasses, load_settings(), load_dotenv_file()
│   ├── db.py                # SQLite queue, simulations, quota, catalog, and vault tables
│   └── pnl_store.py         # Daily PnL series storage for local deduplication
├── generation/
│   ├── producer.py          # Queue refill loop (template / algorithmic_llm / evolution)
│   ├── template_expander.py # Template expansion across markets and fields
│   ├── algorithmic_llm.py   # Ollama / OpenAI-compatible FastExpr generation
│   ├── ast_mutator.py       # Safe AST mutation over elite expressions
│   ├── bandit.py            # Multi-armed bandit market/strategy selection
│   ├── validator.py         # FastExpr syntax & field/operator validation
│   └── pre_dedup.py         # Pre-simulation AST & parameter deduplication
├── execution/
│   ├── dispatcher.py        # Multi-sim slot dispatcher, error learning, circuit breakers
│   ├── slot_pool.py         # Concurrent simulation slot tracking (e.g. GLB=2, others=1)
│   └── quota.py             # NY-date daily simulation cap & hourly pacing
├── qualification/
│   ├── checks.py            # Gate 1 performance/sanity & flatline checks
│   ├── correlations.py      # Gate 2 self, production, and power-pool correlation checks
│   ├── dedup_vault.py       # Local PnL correlation dedup with replacement hysteresis
│   └── submitter.py         # Live re-check & rate-limited BRAIN submission
├── ops/
│   └── alerts.py            # Thread-safe Telegram AlertManager with per-key cooldown
└── deploy/
    ├── com.worldquant.alphafactory.plist
    ├── alpha-factory.service
    └── Dockerfile
```

## Setup & Credentials

Install dependencies inside `.venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r alpha_factory/requirements.txt
```

Create a `.env` file in the repository root (loaded automatically via `load_dotenv_file()` without overriding existing environment variables):

```dotenv
# WorldQuant BRAIN credentials (alternative: ~/secrets/platform-brain.json)
BRAIN_CREDENTIAL_EMAIL=you@example.com
BRAIN_CREDENTIAL_PASSWORD=your-password

# Optional: OpenAI-compatible LLM API key (only needed if generation.llm.backend != "ollama")
# LLM_API_KEY=sk-...

# Optional: Telegram alerts (enable ops.telegram_enabled: true in settings.yaml)
# TELEGRAM_BOT_TOKEN=123456:ABC...
# TELEGRAM_CHAT_ID=123456789
```

Alternatively, store BRAIN credentials as `["email", "password"]` or `{"email": "...", "password": "..."}` in `~/secrets/platform-brain.json` (`chmod 600`). Session cookies are cached at `~/secrets/alpha-factory-session.json`.

## Offline Catalog Bootstrap & Dry Run

Before hitting the live BRAIN API, populate the local SQLite catalog from existing JSON caches (`constants/`, `worldquant-miner/generation_two/constants/`, `alpha_mining/constants/`) and run a dry-run cycle:

```bash
# 1. Bootstrap local operators, datasets, and datafields into SQLite (no network needed)
python -m alpha_factory sync-catalog --offline --config alpha_factory/config/settings.yaml

# 2. Verify generation, validation, and pre-dedup without submitting simulations
python -m alpha_factory run --dry-run --config alpha_factory/config/settings.yaml
```

To refresh operators and market datafields directly from BRAIN:

```bash
python -m alpha_factory sync-catalog --config alpha_factory/config/settings.yaml
```

## CLI Commands

Run all commands from the repository root (`/Users/ad/Desktop/WorldQuant`):

| Command | Purpose |
| --- | --- |
| `python -m alpha_factory run --config alpha_factory/config/settings.yaml` | Start the continuous producer + dispatcher + qualification loop (`--dry-run` skips live BRAIN calls). |
| `python -m alpha_factory sync-catalog --config alpha_factory/config/settings.yaml` | Sync operators, datasets, and datafields from BRAIN (or `--offline` from local caches). |
| `python -m alpha_factory status --config alpha_factory/config/settings.yaml` | Print queue depth, NY-date simulation quota usage, active cooldowns, and vault summary. |
| `python -m alpha_factory vault-list --config alpha_factory/config/settings.yaml` | List qualified active alphas in the deduplicated vault ordered by composite score. |
| `python -m alpha_factory export-submittable --config alpha_factory/config/settings.yaml` | Export top active vault alphas ready for review or manual submission. |
| `python -m alpha_factory reconcile --config alpha_factory/config/settings.yaml` | Reconcile local DB state against BRAIN alpha statuses and tags. |
| `python -m alpha_factory submit --config alpha_factory/config/settings.yaml` | Submit top-ranked kept vault alphas (respecting `submission.max_submissions_per_day` and live re-checks). |

## Safe Quota Ramp-Up (200 -> 1,000 -> 5,000 sims/day)

Keep `submission.auto_submit: false` in `alpha_factory/config/settings.yaml` during ramp-up so qualified alphas accumulate in the local vault for inspection first.

1. **Stage 1 — 200 sims/day (Smoke & Auth Validation)**
   - Set `execution.daily_sim_limit: 200` and `execution.warning_threshold: 160`.
   - Keep `execution.hourly_pacing: true` and `execution.max_slots: 3`.
   - Confirm biometric/2FA alerts, catalog coverage, and Gate 1/Gate 2 pass rates via `status` and `vault-list`.
2. **Stage 2 — 1,000 sims/day (Diversity & Dedup Tuning)**
   - Raise `execution.daily_sim_limit: 1000` and `execution.warning_threshold: 800`.
   - Monitor regional circuit breakers, LLM error rates, and local PnL vault deduplication (`dedup.local_corr_threshold: 0.65`).
   - Run `export-submittable` and manually verify top candidates on BRAIN.
3. **Stage 3 — 5,000 sims/day (Full Production)**
   - Restore `execution.daily_sim_limit: 5000` and `execution.warning_threshold: 4000` (default in `settings.yaml`).
   - Enable `ops.telegram_enabled: true` for real-time kept-alpha, quota, biometric, and circuit-breaker notifications.
   - Optionally enable `submission.auto_submit: true` (bounded by `submission.max_submissions_per_day: 2`) only after verifying consistent Gate 2 pass behavior.

## Deployment

### macOS (`launchd`)

Ensure `alpha_factory/data/logs/` exists, copy the plist to `~/Library/LaunchAgents/`, and load it:

```bash
mkdir -p alpha_factory/data/logs
cp alpha_factory/deploy/com.worldquant.alphafactory.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.worldquant.alphafactory.plist
```

Logs are written to `alpha_factory/data/logs/launchd.stdout.log` and `alpha_factory/data/logs/launchd.stderr.log`. To unload:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.worldquant.alphafactory.plist
```

### Linux (`systemd`)

Edit paths/user in `alpha_factory/deploy/alpha-factory.service` (defaults to `/opt/worldquant`) and install:

```bash
sudo cp alpha_factory/deploy/alpha-factory.service /etc/systemd/system/alpha-factory.service
sudo systemctl daemon-reload
sudo systemctl enable --now alpha-factory
sudo journalctl -u alpha-factory -f
```

### Docker

Build from the repository root so `alpha_factory/` is in the build context, and mount persistent data and secrets:

```bash
docker build -f alpha_factory/deploy/Dockerfile -t alpha-factory .
docker run -d --name alpha-factory --restart unless-stopped \
  --env-file .env \
  -v "$(pwd)/alpha_factory/data:/app/alpha_factory/data" \
  -v "$HOME/secrets:/root/secrets" \
  alpha-factory
```
