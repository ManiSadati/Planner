# PTOAS Repo Checker

This folder contains the operational scaffold for the daily repo watcher. It is designed to:

- scan the local PTOAS and AscendNPU-IR forks;
- track new branch heads and commits since the last scan;
- cap fork discovery at direct forks and forks-of-forks;
- discover GitHub fork branches for configured GitHub repos and store their
  heads in `.repo-checker-state/state.json`;
- summarize branches from metadata first: file counts, line counts, locations,
  and bounded Markdown excerpts;
- score new branches, issues, and PRs before OpenAI sees them;
- track new or updated GitHub issues and PRs for `hw-native-sys/PTOAS`;
- call OpenAI once per daily run to summarize only new changes;
- update `reports/README.md` with the current PTOAS state;
- write `reports/daily/YYYY-MM-DD.md` only when the daily change is large enough.

The user systemd timer is installed and active on this server. It runs once per day at 7:00am Eastern time.

## Layout

- `AGENT.md`: operating contract for the scheduled API-based explorer bot.
- `config.json`: repo paths, model, report paths, and change thresholds.
- `docs/branch-triage-policy.md`: branch/file scoring policy for avoiding missed design branches without chasing AI-generated noise.
- `src/repo_checker/`: collector, state manager, OpenAI summarizer, and report writer.
- `src/repo_checker/github_fork_collect.py`: GitHub direct-fork and fork-of-fork branch collector.
- `src/repo_checker/triage.py`: deterministic scoring for branch/issue/PR priority.
- `systemd/`: 7am user timer template.
- `scripts/install_systemd_timer.sh`: installs or reloads the user systemd timer.
- `.repo-checker-state/state.json`: created on first run and ignored by git.
- `reports/README.md`: generated status report target.
- `reports/daily/`: generated daily big-change reports.
- `reports/backfill/`: Codex-led one-time exploration and re-exploration reports.

## Intended Setup

Your `.venv` is currently treated as an env file. Keep secrets there or in `.env`. `GITHUB_TOKEN` is expected for reliable GitHub PR/issue collection:

```bash
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=github_pat_... # read-only token for GitHub API rate limits
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

The timer is already installed on this server. To reinstall or reload the user systemd units after changing `systemd/` files:

```bash
bash scripts/install_systemd_timer.sh
```

Useful checks:

```bash
systemctl --user list-timers ptoas-repo-checker.timer --all
systemctl --user status ptoas-repo-checker.timer
journalctl --user -u ptoas-repo-checker.service --since today
```

The first run bootstraps state by default and does not summarize old history because `initial_backfill` is `false`.
After that baseline, newly seen branches are inspected once with the same
metadata-first limits.

## Current Limits

GitHub fork discovery is implemented for configured GitHub repos and capped at
`max_fork_depth=2`. It scans direct forks and forks-of-forks, records branch
heads in state, and uses metadata-first compare summaries for changed branches.
Configured `priority_forks` are scanned before the rest of the fork network.
Their named priority branches and default branches are queried explicitly, so
important branches are not lost behind `max_branches_per_fork`; current named
watches include `WenboCodes/PTOAS:new-vf-fusion-design` and
`TaoTao-real/PTOAS:feature-vmi`.

Manually maintained status below `## Persistent Watch Context` in the generated
report is retained when the daily run refreshes the rest of that README.

GitCode fork/issue/PR discovery is not implemented yet. AscendNPU-IR still uses
configured local branch tracking only.

The service sets `TMPDIR` to `./.tmp` so the daily job does not depend on the
server root `/tmp` filesystem.
