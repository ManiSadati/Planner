# PTOAS Repo Checker

This folder contains the first scaffold for a daily repo watcher. It is designed to:

- scan the local PTOAS and AscendNPU-IR forks;
- track new branch heads and commits since the last scan;
- cap fork discovery at direct forks and forks-of-forks;
- summarize branches from metadata first: file counts, line counts, locations,
  and bounded Markdown excerpts;
- track new or updated GitHub issues and PRs for `hw-native-sys/PTOAS`;
- call OpenAI once per daily run to summarize only new changes;
- update `reports/README.md` with the current PTOAS state;
- write `reports/daily/YYYY-MM-DD.md` only when the daily change is large enough.

Nothing is scheduled or executed yet.

## Layout

- `AGENT.md`: operating contract for the scheduled API-based explorer bot.
- `config.json`: repo paths, model, report paths, and change thresholds.
- `docs/branch-triage-policy.md`: branch/file scoring policy for avoiding missed design branches without chasing AI-generated noise.
- `src/repo_checker/`: collector, state manager, OpenAI summarizer, and report writer.
- `systemd/`: 7am user timer template.
- `scripts/install_systemd_timer.sh`: installs the timer when you are ready.
- `.repo-checker-state/state.json`: created on first run and ignored by git.
- `reports/README.md`: generated status report target.
- `reports/daily/`: generated daily big-change reports.
- `reports/backfill/`: Codex-led one-time exploration and re-exploration reports.

## Intended Setup

Your `.venv` is currently treated as an env file. Keep secrets there or in `.env`:

```bash
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp-... # optional, but helps GitHub rate limits
```

When you are ready to run manually:

```bash
cd $HOME/Planner/explorer
set -a
[ -f .venv ] && . ./.venv
[ -f .env ] && . ./.env
set +a
PYTHONPATH=src python3 -m repo_checker --config config.json
```

For a one-off lookback report without changing the normal daily state cursor:

```bash
PYTHONPATH=src python3 -m repo_checker --config config.json --since-days 4 --preserve-state
```

When you are ready to schedule it for 7am Eastern, Monday through Saturday:

```bash
bash scripts/install_systemd_timer.sh
```

The first run bootstraps state by default and does not summarize old history because `initial_backfill` is `false`.
After that baseline, newly seen branches are inspected once with the same
metadata-first limits.

## Current Limits

Automatic GitHub/GitCode fork discovery is not implemented yet. The current
runner watches configured local repos/remotes plus configured GitHub issues and
PRs. `max_fork_depth=2` is the required limit for the fork-discovery collector
when that piece is added.
