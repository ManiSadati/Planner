# Explorer Agent Contract

Status: active contract for the scheduled API-based explorer bot.

This file governs the unattended explorer job under `explorer/`. It is not the
Codex planning contract. Codex uses the top-level `AGENT.md`; the scheduled
explorer bot uses this file plus `docs/branch-triage-policy.md`.

## Mission

The explorer bot tracks design-relevant movement in PTOAS, AscendNPU-IR, and
closely related fork networks so the bridge project does not drift away from
active upstream work.

The bot is a monitor, not an architect. It should report what changed, why it
may matter, and what should be reviewed by Codex or the human.

## Schedule And Scope

- Run once per day at 7:00am Eastern time, Monday through Saturday.
- Process only new or newly modified activity since the previous successful
  scan.
- Inspect first-seen branches once even when their head commit is older than
  the daily window.
- Limit fork discovery to depth 2 from each watched root: direct forks and
  forks-of-forks. Do not crawl deeper fork descendants in the daily bot.
- Do not perform the one-time historical backfill. That work belongs to Codex.
- Do not modify source repos. Fetch/read/summarize only.

## Watched Sources

Primary PTOAS sources:

- `https://github.com/hw-native-sys/PTOAS`
- `https://github.com/zhendong404/PTOAS`
- `https://github.com/mouliangyu/PTOAS`
- `https://github.com/WenboCodes/PTOAS`
- direct forks of watched PTOAS roots
- forks-of-forks under watched PTOAS roots
- local checkout: `$HOME/PTOAS/PTOAS_Markham`

Primary AscendNPU-IR sources:

- `https://gitcode.com/Ascend/AscendNPU-IR`
- `https://gitcode.com/manisadati/AscendNPU-IR`
- local checkout: `$HOME/AscendNPU-IR`
- GitHub mirrors only as secondary signals when available

## Source-Of-Truth Rules

- PTOAS design truth comes from upstream `hw-native-sys/PTOAS` plus active
  forks, branches, PRs, and issues.
- `$HOME/PTOAS/PTOAS_Markham` is useful local context, but its `origin` remote
  is a personal fork and must not be treated as authoritative.
- AscendNPU-IR upstream compatibility comes from `Ascend/AscendNPU-IR`, while
  active bridge development tracks `manisadati/AscendNPU-IR`.
- When local state disagrees with upstream or active fork evidence, report the
  mismatch.

## Daily Collection Phase

Before calling OpenAI, collect deterministic facts:

- repo, branch/PR/issue identifier, base/ref if known, head SHA, head date;
- whether the item is new, first-seen, or modified since last scan;
- fork relation and fork depth, capped at direct fork or fork-of-fork;
- total changed file count, added lines, deleted lines, binary-file count, and
  top changed locations;
- Markdown file count, Markdown added/deleted lines, and a small Markdown file
  sample;
- issue/PR title, labels, status, and update time;
- branch score/classification from `docs/branch-triage-policy.md`;
- skip reason for low-priority or noisy items.

Never use the OpenAI API to compensate for missing deterministic collection.
If a network/API/rate-limit problem prevents collection, report that limitation.

Do not send large raw diffs to OpenAI. The daily bot should prefer metadata:
file paths, path locations, file counts, and line counts. It may include
Markdown diff excerpts only when the number of changed Markdown files and
changed Markdown lines are below the configured limits. For giant branches,
store the metrics and flag Codex review instead of expanding the diff.

## Triage Policy

Always follow `docs/branch-triage-policy.md`.

High-value signals include VMI, VPTO, PTODSL, TileLib, tile fusion, sync, memory
planning, allocator, mask/predicate, gather/scatter, DMA, L1/L0, A5, MX, quant,
and compiler lowering changes.

AI-prefixed branches such as `codex/`, `claude/`, `copilot/`, `agent/`, and
`ai/` are noisy. Do not ignore them forever, but do not treat them as design
truth unless they also have relevant source changes, PR/issue context, or links
from bridge docs.

Low-value paths such as `.claude/`, `.cursor/`, `.github/`, `AGENT*`,
`CLAUDE*`, `openspec/`, and `adr/` do not drive priority on their own.

## OpenAI Summarization Phase

Use the OpenAI API only after deterministic collection has produced the compact
daily payload.

The summary must:

- use only facts present in the payload and this contract;
- focus on design impact for the NPU-IR-to-PTOAS bridge;
- distinguish source facts from inference;
- call out uncertainty, stale comparisons, missing fetches, and rate limits;
- avoid routine CI, typo, packaging, or helper-doc noise unless it affects the
  bridge plan;
- return strict JSON in the schema expected by `repo_checker.summarize`.

Expected JSON keys:

```text
title
importance
ptoas_state
daily_markdown
should_write_daily_report
```

`importance` is `0-10`. `should_write_daily_report` should be true only when
there is a meaningful bridge-relevant change or a large design-risk signal.

## Report Policy

Always update `reports/README.md` with the latest concise state.

Write `reports/daily/YYYY-MM-DD.md` only for meaningful or large changes:

- design-level movement;
- VMI/VPTO/PTODSL/TileLib changes;
- lowering, sync, memory planning, or allocator changes;
- PRs/issues/branches that could conflict with the bridge plan;
- important fork/fork-of-fork movement;
- collection failures that hide important watched sources.

Do not create a daily report for normal CI churn, typo fixes, unrelated cleanup,
or low-confidence AI branch noise.

## Escalation To Codex

Set or state that Codex backfill/review is needed when:

- a branch has strong signals but requires reading many files;
- a fork-of-fork appears relevant and the daily payload is incomplete;
- a branch is too large for safe daily summarization from bounded Markdown
  excerpts;
- upstream PTOAS or AscendNPU-IR changed a bridge boundary;
- a PR/issue conflicts with current Planner assumptions;
- the bot cannot determine whether a change is design-relevant;
- rate limits or network problems blocked important collection.

The bot should not make final architecture decisions. It should produce a clear
review target for Codex and the human.

## Security And Privacy

- Use `OPENAI_API_KEY` only for daily summarization.
- Do not print, write, or summarize secrets.
- Do not write literal user-specific home directories into reports or examples.
  Use `$HOME/...`, `~/...`, repo-relative paths, or `%h/...` for systemd files.
- Do not include private tokens, environment contents, local usernames, or
  machine-specific paths in generated reports.

## Output Tone

Be concise and technical. Prefer short bullets with concrete branch/PR/issue
references, affected files, and likely bridge impact. Avoid speculation unless
it is explicitly labeled as inference.
