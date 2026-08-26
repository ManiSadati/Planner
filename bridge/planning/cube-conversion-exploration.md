# Cube Conversion Exploration Plan

Last updated: 2026-08-26

Status: planning only. No Cube source trace, conversion implementation, or
fixture modification is part of this stage.

## Goal

Determine how AscendNPU-IR Cube compute and its required DMA/staging sequence
should enter PTOAS. The first fixture is:

```text
bridge/triton-example/cube_dotproduct.py
```

The immediate deliverable is a reviewed mapping table, not code.

## Starting State

- Row softmax and RMSNorm are supported by the current AVE-to-VMI bridge for
  the accepted fixtures, with performance accepted as on par with NPU-IR.
- The vector path therefore moves to maintenance mode. New vector instructions
  can be added when a concrete Cube or future kernel exposes a missing row.
- Simple GM->UB and UB->GM DMA conversion already provides a useful bridge
  pattern, but Cube introduces different memory roles, layouts, pipelines, and
  accumulator behavior.
- No one-to-one Cube mapping is assumed yet.

## Core Decision

For every NPU-IR Cube or related DMA template used by the fixture, compare the
actual CCE template implementation against PTOAS/PTO semantics.

Classify each row as one of:

| Classification | Meaning | Intended action |
| --- | --- | --- |
| `direct` | One PTO dialect operation preserves the complete source contract. | Emit that PTO operation from a guarded NPU-IR conversion. |
| `PTO composition` | No single operation matches, but a small PTO sequence preserves the contract. | Emit and test the explicit sequence. |
| `template rewrite` | The CCE template hides behavior that must be rebuilt from structured NPU-IR information. | Intercept before the CCE call and rewrite that template lowering in PTO dialect. |
| `unsupported/unknown` | Equivalence cannot yet be shown. | Keep the CCE path and collect more evidence. |

Operation names are not enough to claim `direct`. The comparison must include
dtype, tile role, layout, valid `m/n/k`, transpose, accumulator initialization,
precision modes, padding, address ownership, pipeline assignment, and sync.

## Investigation Stages

1. **Freeze the fixture and revisions**
   Record the exact `cube_dotproduct.py`, active Wilson NPU-IR revision, PTOAS
   upstream/fork revision, target SOC, and compiler options used later.
2. **Trace the real NPU-IR pipeline**
   Starting from the fixture's early IR, record major forms of Cube compute,
   operand DMA/staging, accumulator/result movement, and synchronization before
   they become CCE calls.
3. **Locate selected CCE templates**
   Identify the template declaration, selection logic, call site, and actual
   implementation for every row reached by the fixture.
4. **Inventory PTO targets**
   Check PTOAS/PTO Cube, tile movement, MTE, fixpipe/store, and sync operations
   against the exact source contracts.
5. **Complete the mapping table**
   Assign one of the four classifications, a preferred interception point,
   risks, and source references to each row.
6. **Choose the first implementation slice**
   Select the smallest complete path only after human review. Preserve the CCE
   path behind the default-off PTO conversion switch.
7. **Validate later**
   Use local IR/PTOAS checks and simulator comparison first; use the human-run
   A5 server for authoritative runtime and performance validation.

## Initial Mapping Table

All rows are deliberately `unknown` until the fixture and implementations are
traced.

| Stage | NPU-IR op/template | Actual CCE behavior | Candidate PTO target | Decision | Evidence needed |
| --- | --- | --- | --- | --- | --- |
| Input GM->L1 | unknown, possibly normal load or `nd2nz` | unknown | `pto.tload`, PTO MTE, or composition | unknown | real pre-template IR and selected template |
| L1->L0A | unknown Cube operand staging | unknown | PTO tile movement / MTE to Left role | unknown | layout, dtype, valid shape, pipeline |
| L1->L0B | unknown Cube operand staging | unknown | PTO tile movement / MTE to Right role | unknown | layout, dtype, valid shape, pipeline |
| Cube compute | unknown structured Cube op, likely `mmadL1`/`mma*` family | unknown | PTO Cube/matmul operation or composition | unknown | exact op, template body, `m/n/k`, init/accumulate modes |
| L0C result | unknown accumulator/fixpipe path | unknown | PTO accumulator move/store or fixpipe composition | unknown | destination, conversion, quantization, layout |
| Synchronization | unknown flags/events around MTE and Cube pipes | unknown | explicit PTO sync or preserved NPU-IR sync | unknown | producer/consumer pipes, event lifetime, ownership |

## Review Gate Before Coding

Do not implement the Cube conversion until the table answers:

- which CCE templates the starter fixture actually uses;
- whether each target is a true one-to-one PTO mapping;
- which rows require PTO compositions or template rewrites;
- where the last structured, semantics-complete interception point is;
- who owns memory planning, addresses, and synchronization;
- what baseline and bridge outputs will prove numerical equivalence.

## Related Documents

- `bridge/memory/cube-conversion-status.md`
- `bridge/planning/npuir-to-ptoas-mapping.md`
- `bridge/planning/dma-template-rewrite-plan.md`
- `bridge/memory/dma-template-mapping.md`
- `bridge/designs/ave-to-ptoas-vmi-conversion-design.md`
