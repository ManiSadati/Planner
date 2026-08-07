# PTOAS Repo Checker

This folder contains the first scaffold for a daily repo watcher. It is designed to:

- scan the local PTOAS and AscendNPU-IR forks;
- track new branch heads and commits since the last scan;
- track new or updated GitHub issues and PRs for `hw-native-sys/PTOAS`;
- call OpenAI once per daily run to summarize only new changes;
- update `reports/README.md` with the current PTOAS state;
- write `reports/daily/YYYY-MM-DD.md` only when the daily change is large enough.

Nothing is scheduled or executed yet.

## Layout

- `config.json`: repo paths, model, report paths, and change thresholds.
- `src/repo_checker/`: collector, state manager, OpenAI summarizer, and report writer.
- `systemd/`: 7am user timer template.
- `scripts/install_systemd_timer.sh`: installs the timer when you are ready.
- `.repo-checker-state/state.json`: created on first run and ignored by git.
- `reports/README.md`: generated status report target.
- `reports/daily/`: generated daily big-change reports.

## Intended Setup

Your `.venv` is currently treated as an env file. Keep secrets there or in `.env`:

```bash
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp-... # optional, but helps GitHub rate limits
```

When you are ready to run manually:

```bash
cd /home/m84446336/Planner/explorer
set -a
[ -f .venv ] && . ./.venv
[ -f .env ] && . ./.env
set +a
PYTHONPATH=src python3 -m repo_checker --config config.json
```

When you are ready to schedule it for 7am UTC:

```bash
bash scripts/install_systemd_timer.sh
```

The first run bootstraps state by default and does not summarize old history because `initial_backfill` is `false`.
