# Explorer Branch Triage Policy

Last updated: 2026-08-10

Purpose: prevent explorer from missing important branch-local design work while also avoiding noise from AI-generated branches and helper docs.

This policy exists because the first PTOAS Stage 2 pass missed `WenboCodes/PTOAS:new-vf-fusion-design`, which contained important VMI-level VF fusion design docs. The fix is not "read every branch deeply." The fix is a deterministic triage pass that surfaces likely-important branches before the AI summary step.

## Core Principle

Explorer should not rely on recency alone.

Old branches can contain durable design intent. Recent branches can be AI churn. A branch is important when multiple signals line up:

- watched repo or trusted fork;
- human-looking branch or PR context;
- design-relevant branch/file names;
- meaningful changed files;
- large or structured design/code changes;
- links from issues, PRs, or prior bridge docs.

No single signal is enough by itself.

## Fork Depth Limit

Daily explorer should crawl at most two fork levels from each watched root:

1. direct forks;
2. forks of those direct forks.

Do not crawl third-level fork descendants in the unattended daily job. If a
second-level fork points to another promising fork network, record the source
and flag it for Codex-led review instead of expanding the daily scope.

## Daily Vs Backfill

Daily explorer:

- focuses on new or newly modified activity since the last successful scan;
- still inspects first-seen branches once, even if their head commit is old;
- records why a branch was skipped so it is not silently lost;
- should summarize only high-confidence relevant changes.

Codex-led backfill:

- can spend more work on older branches, forks-of-forks, and branch-local docs;
- should inspect any suspicious branch that daily explorer deferred;
- should update Planner docs when a branch changes bridge assumptions.

## Strong Branch Signals

These should usually trigger changed-file inspection.

Branch name contains project-relevant design terms:

```text
vmi
vpto
vf
fusion
lowering
backend
tilelib
ptodsl
pto-isa
sync
memplan
memory-plan
allocator
mask
predicate
gather
scatter
dma
l1
l0
a5
mx
quant
```

Branch name is human-looking and design-specific:

```text
new-vf-fusion-design
feature-vmi
feature-vpto-backend
tmp_tile_memory_plan
mte_l1_l0
feature-vmi-vcvt-f4
```

Changed files include design-specific docs or source:

```text
docs/*design*
docs/designs/*
docs/*vmi*
docs/*vpto*
docs/*fusion*
include/PTO/IR/*
include/PTO/Transforms/*
lib/PTO/IR/*
lib/PTO/Transforms/*
ptodsl/docs/*
ptodsl/ptodsl/tilelib/*
lib/TileOps/*
tools/ptoas/*
```

Large changes should increase priority when they touch relevant areas:

- more than about 300 changed lines in design docs;
- more than about 100 changed lines in IR, pass, lowering, TileLib, sync, or memory planning code;
- a new docs folder with a design-specific name;
- multiple coordinated files across docs plus implementation.

Line count is not decisive by itself. It is a multiplier when the file path and branch/source look relevant.

## Weak Or Noisy Signals

These should not be ignored forever, but they should not automatically create high-priority reports.

AI branch prefixes are noisy:

```text
codex/
claude/
copilot/
agent/
ai/
```

Branches with these prefixes should be deprioritized unless at least one stronger signal is also present:

- open PR against upstream with relevant title;
- touches core IR/pass/lowering files;
- mentioned by an issue we already care about;
- modifies a known watch area like VMI, VPTO, PTODSL, sync, memory planning, or gather/scatter;
- large nontrivial source change, not just generated docs or test scaffolding.

Low-value paths by default:

```text
.claude/
.cursor/
.github/
AGENT*
CLAUDE*
openspec/
adr/
```

These paths should not drive priority on their own. They can be useful only as supporting evidence when paired with stronger signals, such as a human branch name, relevant source changes, or an upstream PR/issue discussion.

Important nuance:

- `RFC-*` or `design` docs in a human-looking branch can be strong.
- `ADR` or `openspec` files in an AI-prefixed branch are weak unless linked to code/PR evidence.
- `AGENT*` and `.claude/` files mostly describe agent workflow, not PTOAS design truth.

## First-Seen Branch Rule

When explorer sees a branch for the first time in a watched repo, it should:

1. Record branch name, repo, head sha, head date, and parent/fork relation.
2. Compare the branch to the relevant base when possible.
3. Collect changed file paths and rough line counts.
4. Apply the scoring rules below.
5. Store the result in explorer state even if skipped.

This prevents old but important branches from being missed just because they were created before the daily window.

## Payload And Diff Limits

Daily explorer should be metadata-first. For each changed or first-seen branch,
collect:

- total changed file count;
- added/deleted line counts from `git diff --numstat` or API equivalent;
- Markdown file count and Markdown added/deleted line counts;
- top changed path locations, usually one or two directory levels;
- a bounded sample of changed file paths, prioritizing `.md` and `.markdown`
  files.

Do not ask the model to read large raw diffs. A branch with thousands of files
or millions of changed lines should produce a compact metrics record and, if it
has strong signals, a Codex review target. Markdown excerpts are allowed only
when both the number of changed Markdown files and the number of changed
Markdown lines are below configured limits.

For source-code-heavy branches, daily explorer should usually send file paths,
locations, counts, and commit/PR context. Codex can do the deeper source review
when the metadata indicates bridge relevance.

## Suggested Scoring

Use scoring for deterministic triage before any OpenAI call.

Positive signals:

| Signal | Score |
| --- | ---: |
| Watched primary repo or named fork | +3 |
| Fork-of-fork under watched active fork | +2 |
| Human-looking branch name | +2 |
| Branch name contains strong design term | +3 |
| Changed file path contains strong design term | +3 |
| Touches core IR/pass/lowering/TileLib/sync/memplan code | +4 |
| New design-specific docs folder | +4 |
| Open upstream PR with relevant title | +5 |
| Referenced by relevant issue/PR/bridge doc | +5 |
| Large relevant docs change | +2 |
| Large relevant source change | +3 |

Negative signals:

| Signal | Score |
| --- | ---: |
| AI branch prefix (`codex/`, `claude/`, etc.) | -3 |
| Only touches `.claude/`, `AGENT*`, `.github/`, `openspec/`, or `adr/` | -4 |
| Only CI/test-cache/package metadata changes | -3 |
| Default branch is very far behind upstream and no relevant changed files found | -2 |
| Branch has no PR/issue/context and no design/source paths | -3 |

Suggested classification:

| Score | Classification | Action |
| ---: | --- | --- |
| 8+ | Investigate | Include changed-file summary and top docs/source in daily candidate list. |
| 4-7 | Watch | Record and mention only if connected to current bridge work. |
| 1-3 | Low priority | Store state; do not summarize unless requested. |
| 0 or less | Likely noise | Store skip reason; do not summarize. |

## False Positive Control

The first focused re-exploration showed that old divergent branches can look
artificially important if their compare includes inherited AI docs or stale
process files. Explorer should avoid deciding from a short file sample.

Implementation rules:

- score from the complete changed-file list available from the compare API;
- keep relevant source/docs counts separate from AI/process-doc counts;
- record ahead/behind status and mark heavily diverged compares as lower
  confidence until a better base or merge-base is available;
- downweight branches where most changed files are `.claude/`, `.cursor/`,
  `.github/`, `AGENT*`, `CLAUDE*`, `openspec/`, `adr/`, or generated workflow
  files unless relevant source, PR, issue, or human design docs also exist;
- do not let `adr/` files alone create priority, but allow ADR files to support
  priority when the same branch also has an RFC, design overview, or relevant
  source changes;
- distinguish "legacy design context" from "current source of truth" when a
  branch is far behind upstream but contains useful human-written docs.

## Required Output Per Candidate

Explorer should store this for each branch or PR candidate:

```text
repo
branch_or_pr
base_ref
head_sha
head_date
fork_depth
first_seen_at
last_seen_at
classification
score
positive_signals
negative_signals
changed_file_count
added_lines
deleted_lines
markdown_file_count
markdown_added_lines
markdown_deleted_lines
top_locations
changed_files_sample
important_files
diff_excerpt_policy
skip_reason
needs_codex_backfill
```

If `needs_codex_backfill=true`, daily explorer should not try to deeply analyze it. It should flag it for Codex-led review.

## Special Watch Items

Always keep these in the watch inventory:

- `WenboCodes/PTOAS:new-vf-fusion-design`
- `mouliangyu/PTOAS` and forks of `mouliangyu/PTOAS`
- upstream PRs/issues mentioning VMI, VPTO, PTODSL, sync, memory planning, allocator, mask, predicate, gather/scatter, L1/L0, MX, quant, LLVM branch changes

## What Explorer Should Not Do

- Do not summarize every AI-created branch.
- Do not treat `codex/` or `claude/` branch names as trustworthy design intent.
- Do not treat `.claude/`, `AGENT*`, `openspec/`, or `adr/` changes as high priority by default.
- Do not decide design truth from a fork branch without comparing upstream/fork relation.
- Do not hide skipped branches. Record the skip reason.
- Do not use the OpenAI API key to compensate for missing deterministic collection.
- Do not crawl deeper than direct forks and forks-of-forks in the daily bot.
- Do not send giant branch diffs to OpenAI; send metrics and review targets.

## Implementation Notes

The daily agent should run in two phases:

1. Deterministic collection and scoring:
   - list watched repos/forks/branches;
   - cap fork discovery at depth 2;
   - compare first-seen or changed branches;
   - gather changed-file counts, top locations, Markdown counts, and line
     counts before any text diff;
   - score and classify candidates.
   - use an optional GitHub token when available so PR file-list collection does
     not fail silently under unauthenticated API rate limits.

2. AI summarization:
   - receive only `Investigate` and selected `Watch` candidates;
   - summarize what changed and whether it affects the bridge;
   - write a daily report only for meaningful changes.

This keeps the API-key agent small and focused while reducing the chance of missing branch-local design work.
