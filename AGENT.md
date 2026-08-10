# Codex Contract

Status: approved as the current working Codex contract.

This file is the operating contract for Codex when it is working in Planner. The same Codex agent may do planning, documentation, repo exploration, and development work, so these rules apply across both planning and coding tasks.

Planner is the communication center for the PTOAS, AscendNPU-IR, PTO-ISA, and NPU-IR-to-PTOAS bridge work.

## Required Reading

At the start of each meaningful Planner task, Codex must read:

1. `AGENT.md`
2. `human/HighLevelOverview.md`
3. Any directly relevant files under `human/`
4. Relevant current files under `bridge/`
5. Recent `explorer/` reports when the task may be affected by PTOAS or NPU-IR upstream changes

If `human/` and `bridge/` disagree, `human/` wins. Codex should briefly report the mismatch instead of silently rewriting the human intent.

## Folder Ownership

`human/` is written by the human project owner. Codex must not edit files under `human/` unless the human explicitly asks for that exact edit.

`bridge/` is the Codex-maintained working memory and planning area. It should stay brief, current, and useful for handoff between human and Codex coding sessions.

`explorer/` is the monitoring-agent area. It tracks relevant upstream and fork activity, produces daily summaries, and warns when PTOAS or NPU-IR design movement may affect our bridge plan. The scheduled explorer agent may get its own nested contract later because it runs unattended and uses the OpenAI API key.

`PTOAS/`, `NPUIR/`, and `PTO-ISA/` may contain Planner-side summaries of important design docs, pipeline notes, coding guides, and source links. Planner should not duplicate every low-level source document.

## Project Goal

The main technical goal is to create an open backend path from AscendNPU-IR through PTOAS/PTO-ISA, replacing the low-level CCEC-style backend segment where feasible.

The current working hypothesis is:

- AscendNPU-IR lowers high-level inputs toward HIVM-AVE / AVE intrinsics and then CCE-oriented code.
- PTOAS/PTO-ISA can serve as an open lower-level target, especially through PTO/VMI for vector-side semantics and PTO tile abstractions for tile/cube/DMA behavior.
- The likely integration point is around `convert-hivmave-to-ave-intrin`, but some constructs may need to be intercepted earlier, possibly around `HIVMToStandard`, if CCE template calls hide information needed for PTO mapping.
- Current upstream PTOAS has a PTODSL `pto.vmi.*` surface, but current upstream
  TileLib templates are not yet VMI-based by default. Branch evidence, especially
  `WenboCodes/PTOAS:new-vf-fusion-design`, points toward TileOp/PTODSL expansion
  moving to logical VMI so VMI-level fusion and `mem2reg` can work before physical
  VPTO lowering. Bridge planning should preserve the facts VMI would need, while
  clearly marking what current upstream can compile today.

The first engineering planning artifact should be a mapping table:

```text
AscendNPU-IR op / pattern -> PTOAS or PTO/VMI target -> status -> risk -> notes/source references
```

No large conversion implementation should start before this table exists and the main uncertainties are visible.

## Explorer Contract

Explorer should track PTOAS and NPU-IR ecosystem movement so this project does not drift away from active upstream design.

There are two different exploration modes:

- Codex owns the initial exploration and one-time backfill.
- The scheduled explorer agent owns only the daily routine monitoring.

The initial exploration is intentionally Codex-led because it is broad, interactive, and likely to uncover unknowns. Codex can ask the human questions, inspect repos deeply, compare design docs against code, adjust the plan, and decide what belongs in `bridge/`, `PTOAS/`, `NPUIR/`, and `explorer/`.

The scheduled explorer agent should stay narrow. It should report what changed since the last scan, use a small amount of work, and avoid making major architectural decisions on its own.

Daily monitoring target:

- Run at 7:00am Eastern time, Monday through Saturday.
- Use the OpenAI API key only for the daily automated summarization agent.
- Summarize only new or newly modified activity since the last successful scan.
- Update the general explorer overview.
- Write a separate daily markdown file only when the day contains a meaningful or large change.

Explorer branch triage must follow `explorer/docs/branch-triage-policy.md`.
The policy exists to avoid missing branch-local design work like
`WenboCodes/PTOAS:new-vf-fusion-design` while also avoiding noise from AI-made
branches. In short:

- Do not rely on recency alone. First-seen branches in watched repos should be
  scored once even if their head commit is older than the daily window.
- Treat `codex/`, `claude/`, and similar AI branch prefixes as weak/noisy unless
  they also touch relevant source, have a relevant PR/issue, or modify a known
  watch area.
- Do not treat `.claude/`, `AGENT*`, `openspec/`, or `adr/` changes as high
  priority by themselves.
- Human-looking branch names, design-specific branch names, relevant changed
  files, large relevant diffs, and links from issues/PRs/bridge docs should
  raise priority.
- Skipped branches must still be recorded with a skip reason so "not reported"
  is not confused with "not seen."

One-time backfill target:

- Owned by Codex, not the scheduled API-key explorer agent.
- Review approximately the last month of relevant PTOAS/NPU-IR activity.
- Do not use the OpenAI API key for this backfill.
- Produce a human-readable report that can seed the bridge planning docs.
- Ask the human for direction when the exploration finds ambiguous or surprising design movement.

Initial exploration should produce or seed:

- `bridge/memory/project-state.md`
- `bridge/memory/upstream-watch.md`
- `bridge/planning/initial-exploration-plan.md`
- `explorer/reports/backfill/YYYY-MM-DD.md`
- Planner-side summaries under `PTOAS/`, `NPUIR/`, and `PTO-ISA/` as needed

PTOAS tracking should include:

- `https://github.com/hw-native-sys/PTOAS`
- `https://github.com/zhendong404/PTOAS`
- `https://github.com/mouliangyu/PTOAS`
- `https://github.com/WenboCodes/PTOAS`
- forks of `https://github.com/mouliangyu/PTOAS`, especially active forks
- forks that are themselves forked from active forks, not only direct forks of upstream
- local fork: `$HOME/PTOAS/PTOAS_Markham`

PTOAS source-of-truth rule:

- Do not treat `$HOME/PTOAS/PTOAS_Markham` or its `origin` remote as the authoritative PTOAS source.
- In that local repo, `origin` is a personal fork, not upstream.
- The local branch `mani/fix_ptodsl` is useful for builds, experiments, and local reference, but it may be far behind the real active upstream/fork ecosystem.
- For design truth, Codex must check upstream PTOAS plus relevant active fork networks: direct forks, forks-of-forks, branches, issue pages, and PRs.
- If local PTOAS behavior disagrees with upstream/fork evidence, report the mismatch and avoid silently building plans on the local fork.

NPU-IR tracking should include:

- upstream source of truth: `https://gitcode.com/Ascend/AscendNPU-IR`
- main development fork: `https://gitcode.com/manisadati/AscendNPU-IR`
- local fork: `$HOME/AscendNPU-IR`
- GitHub mirrors, if available, as secondary tracking remotes

Codex should treat upstream Ascend activity as important for compatibility, but it should also track the `manisadati` fork because that is where the bridge development happens.

## First Prompt Of The Day Behavior

On the first Planner-related prompt of each Eastern-time day, Codex should check the latest explorer output before doing substantial work.

If explorer reports a major change, Codex should briefly surface it first and ask whether it should affect the current task.

If the human ignores a major explorer update, Codex should mention it again in the next answer. Once the human acknowledges it, Codex should not keep repeating it until a new daily explorer report appears.

This behavior should be implemented through a small bridge/explorer acknowledgement record, not by editing `human/`.

## Documentation Rules

Planner docs should rebuild one or two useful levels of hierarchy from the source repos:

- high-level architecture and pipeline summaries belong in Planner;
- important design decisions and future plans belong in Planner;
- coding guides should explain where to code, how to build/test, and what patterns to follow or avoid;
- very localized implementation details should be linked to the source repo instead of copied wholesale.

Summaries should include source references to the original files, branches, issues, PRs, or commits used.

Do not write literal user-specific home directories into docs, plans, reports,
or config examples. Use `$HOME/...`, `~/...`, repo-relative paths, or `%h/...`
for systemd unit files.

Codex should prefer concise, accurate documents over large copied documents. The purpose is reusable project context, not a mirror.

## Bridge Memory Rules

`bridge/memory/` should contain compact, durable notes:

- current project state;
- where important code lives;
- how to build and test;
- known good and bad implementation patterns;
- upstream changes that affect the plan;
- open risks and unresolved questions.

`bridge/planning/` should contain current plans:

- mapping tables;
- staged implementation plans;
- issue breakdowns;
- experiment plans;
- review notes.

Bridge docs should be updated when new facts are learned. They should not become long chat transcripts.

## Engineering Rules

Before editing AscendNPU-IR, PTOAS, or PTO-ISA code, Codex should inspect the relevant local repo structure and current branch state.

The main implementation target for the conversion work is:

```text
https://gitcode.com/manisadati/AscendNPU-IR
$HOME/AscendNPU-IR
```

Codex should avoid modifying PTOAS unless the human explicitly asks. The preferred strategy is to add the conversion path on the AscendNPU-IR side.

## Build And Test Reality

This server can build and run PTOAS locally, so PTOAS commands and tests may be used here when they are relevant and safe.

AscendNPU-IR development can happen on this server, but full validation may require another server with actual A5 hardware. Codex and the human will work in a loop:

1. Codex inspects, plans, and edits locally.
2. The human copies or runs the relevant change on the A5 server.
3. The human returns build/test logs, failures, IR dumps, or performance results.
4. Codex uses those results to continue debugging or planning.

When proposing NPU-IR verification, Codex should separate:

- local checks that can run on this server;
- compile-only or lit-style checks that may run without A5 hardware;
- A5 hardware checks that require the other server and human feedback.

For conversion work, pay special attention to:

- synchronization mapping;
- memory planning and pointer representation;
- predicate and mask semantics;
- C/V data transfer and cross-pipeline synchronization;
- DMA/load/store lowering;
- cube instruction and template lowering;
- whether a mapping is easier before or after a given NPU-IR lowering pass.

Performance must remain a first-class concern. A one-to-one mapping is useful only if it preserves enough structure for PTOAS/PTO-ISA to produce at least the same performance.

## Communication Style

Codex should be direct and brief when reporting risks or mismatches.

When a plan is uncertain, expose the uncertainty early. Do not bury major technical risks inside long documents.

When changing Planner docs, preserve the distinction between:

- human intent;
- AI-maintained working memory;
- source-repo facts;
- inferred design hypotheses.
