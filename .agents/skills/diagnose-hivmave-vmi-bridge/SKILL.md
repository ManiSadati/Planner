---
name: diagnose-hivmave-vmi-bridge
description: Use when logs or MLIR expose unsupported AVE/HIVM-to-PTOAS VMI conversions, failed legalization, residual CCE helper calls, or downstream failures while adding AscendNPU-IR bridge support.
---

# Diagnose and Implement AVE-to-VMI Support

Adapted from `AscendNPU-IR/skills/diagnose-hivmave-vmi-bridge/SKILL.md` for
Planner's role and toolchain conventions. This is a maintained local workflow,
not a symlink or a requirement to load a second copy. Read
[Planner's contract](../../../AGENT.md), its required reading, the
[current plan](../../../bridge/planning/README.md), and the current assignment.
Resolve Planner from this file's location, not the shell's working directory.

## 1. Establish scope and provenance

Resolve `ASCEND_NPU_IR_ROOT` from the assignment/environment and verify its
checkout and worktree. If unset, inspect sibling `AscendNPU-IR` and the documented
checkout; do not assume a personal absolute path or the latest branch name.
Record status, branch, revision, relevant submodule revisions, build/install
paths, compiler executables, target, options, bridge mode, and cache directory.
Use matching pinned PTOAS/LLVM tools; keep the two LLVM libraries separate.
Unknown installed revisions remain unknown, even if neighboring source has a fix.

Clarify analysis versus implementation and compilation versus execution from
the request. Before implementation, explain the problem and ranked solutions,
owned files and checks; satisfy applicable approval requirements. Honor existing
approval without asking again for the same scope. Preserve unstaged/untracked
work; do not reset, stash, stage, commit, or switch another worktree's branch.

## 2. Inventory logs and recover the actual operation

Inventory logs, testcase node IDs/parameters, commands, status codes, dump paths,
and first actionable diagnostics. Use `rg --files` and search diagnostic text:

```sh
rg -n 'failed to legalize|op is unsupported|undefined symbol|IR Dump|error:' "$TASK_LOGS"
```

For `.log.gz`, use `gzip.open(path, 'rt', errors='replace')` in Python; ordinary
`rg` does not decompress it. Group by operation plus rejected contract, retaining
each unique case ID. Repeated stderr and reduced/raw copies are not new cases.
Distinguish a pass rejection from wrapper exceptions and later linker errors.

Build a case table: case/parameters, first failing stage, operation or symbol,
exact rejection, artifact and line, provenance, and confirmed versus unknown.
For direct MLIR inputs, parse with the matching dialect/tool versions first.

Recover operand/result types, ranks, shapes, layouts, memory spaces, dynamic
operands, attributes, and relevant producers/consumers. For truncated `...`
dumps, follow raw-log/cache paths, same-case artifacts, compressed companions,
then older full dumps; reproduce with the original command when permitted.
Older or derived IR is evidence only for its own revision/specialization.

Extract real dump markers, not markers embedded in Python escaped stderr strings.
Strip pytest's literal `E` prefix while preserving indentation, include affine
and type aliases, stop at the complete module boundary, and verify parsing.
Never construct guessed types or attributes from an operation name alone.

For fresh capture, inspect supported printing flags; with `bishengir-compile`
use pass-specific before/after dumps and `--mlir-print-ir-after-failure`, with
`--mlir-disable-threading` when needed. Other tools may not support those flags.
Preserve stdout/stderr and the original flags/target. Failure dumps may already
contain partial conversions and are not automatically valid pass inputs.

## 3. Classify before choosing the implementation boundary

| Evidence | Investigation and likely owner |
|---|---|
| `ave.hir.*` or memref legalization | Pattern, type converter, legality, consumers in `bishengir/lib/Conversion/HIVMAVEToPTOASVMI/`; compare native `HIVMAVEToAVEIntrin` semantics. |
| Structured HIVM/template or DMA rejection | Current selection/lowering in `bishengir/lib/Conversion/HIVMTemplatesToPTO/`, native template source, and pipeline order. |
| `_mlir_ciface_*` undefined symbol | Trace surviving declaration and calls, selected native identity, PTODSL availability, ABI, and actual link inputs. |
| Valid VMI rejected by PTOAS | Inspect current pinned VMI verifier and physical lowering; distinguish representation limits from bridge misuse. |
| Frontend failure or UB overflow | Diagnose the earlier stage; a VMI pattern cannot repair an operation never reaching it. |
| Source supports the exact form, invoked binary rejects it | Check build/install/resource provenance and caches before changing source. |

Search the exact diagnostic, then read the complete matcher and helpers. Inspect
`regbase/PassPipeline.cpp` and the loaded Triton backend to confirm which path
actually runs. Do not assume a legacy direct rewrite, an early adapter, or a
shared-ABI resolver is active from an old report. Current code and current human
direction must establish the design; report consequential conflicts first.

For templates, explicitly distinguish four facts:

1. Native CCE implementation and registered datatype instance exist.
2. Corresponding PTODSL authoring source exists.
3. Generated MLIR exports the exact selected symbol and is registered/installed.
4. This compilation imports that definition with the correct operand ABI.

None implies the next. Follow the
[current template design guide](../../../bridge/designs/ptodsl-template-development-and-fallback.md)
and [native investigation skill](../investigate-native-template/SKILL.md).
For example, a native FP32 1D UB-copy template can exist while its PTODSL
replacement is absent, or an existing bridge matcher accepts only static
lengths. A dynamic copy may then leave a CCE call even though LLVM emission
succeeds. Inspect offsets, sizes, strides, and the selected resource before
choosing template implementation, matcher/adapter repair, or direct lowering.
Linking a descriptor-based native helper to a bare-pointer declaration is not
a valid ABI repair.

## 4. Establish the semantic mapping and approved fix

Create a small mapping table: rejected form, native semantics, target VMI/PTO
sequence, guards, risk, and supporting source/test evidence. Rank solutions:
extend an existing valid lowering; generalize an observed family; or normalize
upstream when that preserves semantics more clearly. Rebuild stale tools when
support already exists. Explain a downstream fix versus a guarded workaround
when VMI cannot represent the exact contract; PTOAS edits require direction.

Check masks and inactive lanes, logical versus physical widths/layouts, two-result
ordering, numeric versus bit conversions, signedness, rounding/saturation,
byte versus element/bit offsets, dynamic extents and independent strides,
memory spaces, padding/tails, descriptor fields, and synchronization as relevant.
A matching instruction name is not a semantic proof. In particular, physical
sparse B16 carriers may require a checked composition instead of logical
interleave; an unmasked conversion is valid only with a proven consumer graph.

After approval, create the requested branch from the agreed base, using an
isolated worktree when necessary. Keep one atomic contract change. A newly
exposed blocker gets its own diagnosis and scope decision. Preserve deferred
workarounds and disabled-finalizer policy: do not enable
`finalize-ptoas-vmi-bridge` or `finalize-ptoas-vpto-predicate-memory` without
explicit authorization. Do not transplant their rewrites to hide an unapproved
pipeline change.

## 5. Regress, implement, and build

Add a reduced lit/FileCheck regression exposing the operation and the consumers
needed to prove its contract. Verify it fails before the fix for the intended
reason, not syntax, parsing, or an unrelated pass. Add nearby unsupported-form
rejections where loosening guards could miscompile. Use `split-file` when one
failure would obscure others. Assert addresses, masks, result use, ABI, and
observable emitted operations rather than C++ text.

Extend the actual selected pattern/type conversion/resource adapter, with precise
guards. Preserve metadata until its last use. Do not mark an unsupported op legal,
drop attributes, or leave unresolved casts to obtain a successful exit. For a
template compute body, follow existing PTODSL authoring/import mechanisms rather
than adding a second implementation in a C++ conversion.

Reuse the checkout's configured build and generated lit configuration. Build the
changed compiler tools and required installed resources; tool paths in lit may
still resolve to a different worktree. Check `lit.site.cfg.py` and effective
executables. Avoid rebuilding LLVM unnecessarily or replacing shared tools while
another task uses them. Keep run artifacts and caches private.

## 6. Verify each requested stage separately

Run the new tests and relevant existing suites against the rebuilt tools. Keep
pre-existing failures separate. Replay affected original specializations where
available, with the recorded target/options/mode and fresh caches. Derived probes
and historical IR do not establish exact-case success. Honor compilation-only
scope; obtain separate authorization before launches/simulator or hardware tests.

| Stage | Required evidence and limits |
|---|---|
| Conversion | Matching tool exit, checked VMI/PTO, no unexpected residual AVE/memref/casts. |
| VPTO/LLVM emission | Actual pinned PTOAS command, exit and emitted artifacts; downstream scope/layout/mask failures remain blockers. |
| Object/link | Actual object and final link exits. Inspect symbol definitions and unresolved dependencies with matching `llvm-nm`/`llvm-dis` or emitted LLVM. |
| Runtime/correctness | Authorized launch and numerical testcase results; profiler/process exit alone is insufficient. |

Inspect emitted VMI/LLVM for residual `_mlir_ciface_*`, native template calls,
and declarations without imported bodies. External LLVM/runtime intrinsics and
explicitly approved native fallback are not automatically errors; identify
which dependencies must be resolved by this flow and verify actual link inputs.
Do not report link success from an empty symbol scan or LLVM emission. A passing
fallback run is not proof that the PTODSL implementation was selected.

## 7. Document and hand off

For every conversion, state **one-to-one or composition**, show the target
sequence, and explain how the relevant semantics above are preserved. Separate
one logical VMI operation from later physical instruction expansion. Record
supported and rejected contracts, regressions, tool revisions, commands, stage
results, and remaining uncertainties.

For a genuine PTOAS VMI representation/lowering gap, update the existing entry
in the [compatibility tracker](../../../bridge/designs/ave-ptoas-vmi-compatibility-tracker.md)
or add a precise new one: exact contract, diagnostic, revision, affected cases,
workaround and limits, implementation/deferred status, and long-term solution.
A missing bridge pattern or PTODSL resource is not itself a missing VMI
instruction. Use [update-planner](../update-planner/SKILL.md) for the narrow
appropriate document, without editing `human/` or rewriting unrelated history.

Finish with branch/worktree, changed files, case/mapping table, evidence paths,
commands/results by stage, unverified linking/runtime, next blocker and its
scope, plus a suggested `fix(bridge): support <contract>` message. Review
`git diff --check` and untracked files; leave work unstaged and uncommitted.
