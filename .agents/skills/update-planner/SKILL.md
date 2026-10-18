---
name: update-planner
description: Record PTOAS and AscendNPU-IR bridge decisions, implementation status, test evidence, and next actions in the coworker-shared Planner repository without overstating coverage.
---

# Update Planner and Hand Off

Read [Planner's contract](../../../AGENT.md), its required reading, and the
[planning index](../../../bridge/planning/README.md). Resolve Planner from this
skill's location; the shell may be inside a different checkout.

## Reconcile

1. Inspect the current diff and recent relevant documents before editing.
   Separate the human's decisions, implemented code, verification evidence,
   proposed work, and deferred experiments. Report contradictions with human
   intent; do not resolve them by rewriting `human/`.
2. Update the narrowest maintained document: `bridge/` for cross-repo work,
   `NPUIR/` or `PTOAS/` for single-project guidance. Update the planning index
   only when priorities or navigation change; link to details instead of copying
   the same instructions into several summaries.
3. For template work, maintain the relevant inventory/triage rows and their CSV
   counterpart when changing coverage status. Preserve unrelated edits and do
   not bulk-label instances supported based on one successful kernel.
4. Record source revisions, branch/merge state, installed-build caveats, exact
   selected symbols, configurations tested, results, evidence paths, limitations,
   and the next bounded action. Native fallback success is not PTODSL coverage;
   compile, simulator, and physical hardware validation are distinct claims.

## Handoff Rules

- Use repository-relative paths and `$HOME` in examples, never personal account
  paths. Keep large generated logs/build outputs out of git unless explicitly
  requested as a small curated evidence set.
- Preserve the current per-template adapter decision. Record the uniform-ABI
  branch as deferred, not the production plan, unless the human changes course.
- Do not edit `human/`, scheduled Explorer behavior, global configuration,
  compiler source, or submodule pins as a documentation side effect.
- Do not commit, push, or mark an MR merged without the requested authorization
  and supporting repository evidence.
- End with changed document paths, decisions captured, outstanding questions,
  and a compact next-agent brief: objective, owned files, dependencies, checks,
  and work that must remain untouched.
