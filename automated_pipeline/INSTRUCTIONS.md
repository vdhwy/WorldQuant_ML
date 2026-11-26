# Automated Pipeline — Running Instructions
# ══════════════════════════════════════════════════════════════════════════════

This document covers all the modes, flags, and options for running the
WorldQuant BRAIN Automated Simulation Pipeline.


## Quick Start

```bash
cd automated_pipeline

# Full run with default config
python pipeline.py

# Full run with custom templates from file
python pipeline.py --template-file templates.txt

# Dry run (generate expressions only, no simulation)
python pipeline.py --template-file templates.txt --dry-run
```


## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1     | Authenticate with WorldQuant BRAIN |
| 2     | Discover datasets matching config filters |
| 3     | Fetch datafields for each dataset |
| 4     | Load operators (Vector & Time-Series) |
| 5     | Generate expressions from templates → deduplicate → similarity filter |
| 6     | Convert expressions to alpha dicts |
| 7     | Simulate in batches with safe polling |
| 8     | Collect results & export to CSV |


## Running Modes

### 1. Auto Mode (default)
Uses built-in templates + config-file `custom_templates` only.

```bash
python pipeline.py
```

### 2. Template File Mode
Loads templates from a text file (one per line, `#` for comments).

```bash
python pipeline.py --template-file templates.txt
```

### 3. CLI Template Mode
Pass templates directly as arguments (repeatable).

```bash
python pipeline.py --template "ts_delta({field}, {lookback})"
python pipeline.py --template "rank({ts_op}({vector_field}, {lookback}))" \
                    --template "{ts_op}({vec_op}({field}), {lookback})"
```

### 4. Interactive Mode
Prompts you to enter templates one-by-one with live preview.

```bash
python pipeline.py --interactive
```

### 5. Custom-Only Mode
Disables all built-in templates (sentiment, vector, matrix).
Only runs your custom/interactive/file templates.

```bash
python pipeline.py --only-custom --template-file templates.txt
python pipeline.py --only-custom --interactive
```

### 6. Dry Run
Generates expressions and shows samples without simulating.
Useful for verifying templates before spending simulation credits.

```bash
python pipeline.py --template-file templates.txt --dry-run
```


## Market & Filter Overrides

Override config values directly from CLI:

```bash
# Change region and universe
python pipeline.py --region CHN --universe TOP2000

# Change dataset category
python pipeline.py --category SENTIMENT

# Change lookback windows
python pipeline.py --lookback 5 10 20 60

# Combine overrides
python pipeline.py --region EUR --universe TOP2500 --lookback 10 30 \
                    --template-file templates.txt
```


## Similarity Filtering Options

The pipeline applies two layers of similarity filtering to avoid
simulating redundant expressions:

### Layer 1: Historical Similarity
Compares new expressions against previously simulated alphas stored in
the `alpha_mining` database. Uses weighted multi-metric scoring:
- String similarity (40%) + Operator overlap (30%) + Field overlap (20%)
  + Structural similarity (10%)
- Threshold: 0.85

### Layer 2: Intra-batch Structural Fingerprint
Deduplicates expressions within the current batch that share the same
operator structure but differ only in field names. Uses O(n) structural
fingerprinting — replaces field names with canonical placeholders and
groups by the resulting fingerprint.

### CLI Flags

```bash
# Disable ALL similarity filtering (simulate everything)
python pipeline.py --template-file templates.txt --no-similarity

# Disable only historical DB check (keep intra-batch dedup)
python pipeline.py --template-file templates.txt --no-historical-similarity

# Disable only intra-batch dedup (keep all field variants)
python pipeline.py --template-file templates.txt --no-intra-similarity

# Keep 5 field variants per unique operator structure (default: 1)
python pipeline.py --template-file templates.txt --max-per-structure 5

# Combine: skip DB check, keep 3 variants per structure
python pipeline.py --template-file templates.txt \
                    --no-historical-similarity \
                    --max-per-structure 3
```

### What `--max-per-structure N` does

When a template like `rank({ts_op}({matrix_field}, {lookback}))` expands
across 1000+ matrix fields × 20 ts_ops × 3 lookbacks, many expressions
share the exact same operator skeleton:

```
rank(ts_rank(F0, 10))     ← structure fingerprint
rank(ts_rank(field_A, 10))  → kept (1st)
rank(ts_rank(field_B, 10))  → kept if N≥2, else skipped
rank(ts_rank(field_C, 10))  → kept if N≥3, else skipped
...
```

- `--max-per-structure 1` (default): 1 expression per skeleton → minimal simulation cost
- `--max-per-structure 5`: 5 field variants per skeleton → more coverage
- `--no-intra-similarity`: keep all → full Cartesian expansion


## Template Placeholders

| Placeholder      | Expands to                                          |
|------------------|-----------------------------------------------------|
| `{ts_op}`        | Each Time-Series operator (ts_rank, ts_delta, etc.) |
| `{vec_op}`       | Each Vector operator (rank, zscore, etc.)            |
| `{field}`        | Every datafield (any type)                           |
| `{vector_field}` | VECTOR-type datafields only                          |
| `{matrix_field}` | MATRIX-type datafields only                          |
| `{risk_matrix}`  | Risk-related MATRIX fields                           |
| `{pos}`          | Fields containing "pos" (positive sentiment)         |
| `{neg}`          | Fields containing "neg" (negative sentiment)         |
| `{lookback}`     | Each lookback window from config                     |

### Template Examples

```bash
# Simple: single operator + field
ts_delta({field}, {lookback})

# Nested: vector op inside time-series op
rank({ts_op}({vec_op}({vector_field}), {lookback}))

# Sentiment: compare positive vs negative fields
{ts_op}({vec_op}({neg}) - {vec_op}({pos}), {lookback})

# Group neutralization
group_neutralize({ts_op}({field}, {lookback}), sector)

# Risk-adjusted
vector_neut(ts_zscore({matrix_field}, {lookback}), {risk_matrix})

# Complex multi-level (two independent field slots via different placeholder types)
multiply(ts_backfill({vector_field}, {lookback}), group_rank(multiply(rank(ts_backfill({vector_field}, {lookback})), group_rank(ts_backfill({matrix_field}, {lookback}), subindustry)), industry))
```

> **Tip:** Using two different placeholder types (e.g. `{vector_field}` and
> `{matrix_field}`) gives you **two independent field slots** that expand
> via Cartesian product. The same placeholder used multiple times always
> gets the same value.


## Configuration File

The pipeline reads from `pipeline_config.yaml`. Key sections:

```yaml
market:
  region: "USA"          # USA, EUR, GBR, CHN, ASI, GLB, etc.
  universe: "TOP3000"    # TOP500, TOP700, TOP2000, TOP3000, etc.
  delay: 1

dataset_filters:
  category: "model"      # "" for all, or: SENTIMENT, FUNDAMENTAL, etc.
  coverage_min: 0.0
  value_score_min: 0.0

expressions:
  lookback_days: [10, 30, 60]
  templates:
    sentiment_comparison: false
    vector_ops: false
    matrix_ops: false
  custom_templates:
    - "ts_rank({matrix_field}, {lookback}) - ts_mean({matrix_field}, {lookback})"

simulation:
  batch_size: 200
  multi_sim_limit: 10
  concurrent_limit: 3

output:
  results_dir: "pipeline_results"
  cache_file: "simulation_cache.parquet"
```


## Common Recipes

```bash
# Explore a new region with dry run first
python pipeline.py --region JPN --universe TOP2000 --category model \
                    --template-file templates.txt --dry-run

# Full simulation, no dedup (test all field variants)
python pipeline.py --template-file templates.txt --no-similarity

# Production run with moderate diversity
python pipeline.py --template-file templates.txt --max-per-structure 3

# Quick test of a single expression pattern
python pipeline.py --only-custom \
                    --template "ts_rank({matrix_field}, {lookback})" \
                    --max-per-structure 1

# Combine CLI templates + file templates
python pipeline.py --template "ts_delta({field}, {lookback})" \
                    --template-file templates.txt

# Override everything from CLI
python pipeline.py --region EUR --universe TOP2500 \
                    --category FUNDAMENTAL --lookback 5 20 60 \
                    --only-custom --template-file templates.txt \
                    --max-per-structure 5
```


## Output

Results are saved to `pipeline_results/results_{REGION}_{TIMESTAMP}.csv`.

Key columns in the output CSV:
- `expression` — the alpha expression that was simulated
- `alpha_id` — WorldQuant BRAIN alpha ID
- `sharpe` — Sharpe ratio
- `fitness` — fitness score
- `turnover`, `drawdown`, `returns`, `margin`
- Submission checks: `low_sharpe`, `low_fitness`, `high_turnover`, etc.

The simulation cache (`simulation_cache.parquet`) tracks previously simulated
alphas so re-runs automatically skip them.
