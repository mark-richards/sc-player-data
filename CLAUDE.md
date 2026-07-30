# Claude Instructions

## Environment
- Run Python with `py` (not `python3` or `python`)
- Shell: bash on Windows 11 — use Unix paths and syntax

## Workflow
- Read files before proposing changes
- Don't add features, comments, or error handling beyond what's asked
- Prefer editing existing files over creating new ones

## Directory Guide

| Directory | Purpose |
|---|---|
| `data/raw/fanfooty/` | Scraped FanFooty .txt match files (by year/round) |
| `data/raw/supercoach/` | SC API JSON files — completeStatspack per player + round archives (by year) |
| `data/raw/herald_sun/` | Herald Sun article data |
| `data/processed/` | Cleaned CSVs: per-round fanfooty, master_player_data.csv, league_master.csv |
| `data/live/` | Auto-refreshed from SC API each pipeline run — ladder, fixtures, rosters, player stats |
| `data/live/json/` | Raw SC API round JSON responses |
| `data/predictions/` | LightGBM model output CSVs |
| `draft_prep/SC {year}/` | SC player lists, draft results — pre-season snapshots |
| `models/` | Trained ML model artifacts (.joblib) |
| `reports/` | Generated newsletter markdown files |
| `archive/notebooks/` | .ipynb files — DISCOVERY ONLY, never copy code from these |
| `archive/scripts/` | One-off analysis scripts — reference only |
| `archive/legacy/` | Old R data, legacy inputs — not used by active pipeline |

**Never use `inputs/` or `outputs/` — removed. Use `data/live/` instead.**

**Never use `fantasy_banter_data/` — removed. Use `data/raw/supercoach/` instead.**

**.ipynb files in `archive/notebooks/` are historical reference only — never use as code sources.**

## Multi-Device Git Workflow

This repo is edited from multiple devices (this PC, phone, other laptops), but the weekly
newsletter only runs unattended on one PC via Task Scheduler, which syncs with
`origin/master` immediately before each run and aborts if the checkout is dirty or has
diverged from origin. GitHub is the single source of truth — follow this on every device:

- **At the start of a session** doing code changes: `git fetch origin` then
  `git merge --ff-only origin/master` so edits start from current code.
- **At the end of a session** where source files changed: stage the specific changed files
  (not `-A`), commit with a clear message, and push to `origin master` — do this
  automatically, without waiting to be asked.
- If push is rejected (non-fast-forward), pull/rebase and retry. **Never force-push.**
- If a real merge conflict occurs, stop and surface it to the user — do not auto-resolve.
- Never commit `credentials.json`, `client_secret_old.json`, `secret/`, or anything else
  gitignored (`data/`, `models/`, `reports/`, `draft_prep/`, `docs/`).

## Active Pipeline Entry Points

| Script | Purpose |
|---|---|
| `run_waiver_newsletter.py` | Weekly newsletter — auto-refreshes all data, then generates newsletter |
| `run_predictions.py` | LightGBM score predictions — auto-refreshes data before predicting |
| `run_pipeline.py` | Full weekly pipeline — ingest → process → train → predict → newsletter |
| `download_2026_statspacks.py` | Download SC completeStatspack JSONs → `data/raw/supercoach/{year}/` |
| `build_master_dataset.py` | Merge SC statspacks + fanfooty → `data/processed/master_player_data.csv` |
| `build_features.py` | Parse raw fanfooty .txt → `data/processed/{year}_round_N_fanfooty_data.csv` |
| `ingest_fanfooty.py` | Scrape fanfooty.com.au → `data/raw/fanfooty/{year}/round_N/*.txt` |
| `website/app.py` | Flask web dashboard |
