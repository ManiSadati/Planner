# AVE to PTOAS VMI Compatibility Tracker

Last updated: 2026-08-26

This file tracks AVE instruction compatibility gaps encountered so far while
lowering AscendNPU-IR AVE to PTOAS VMI. Keep entries short; detailed analysis
belongs in separate design notes.

This is also a standing input to Explorer. A PTOAS branch, issue, PR, or
upstream commit matching one of the watch triggers below is bridge-relevant
even when it does not mention AscendNPU-IR. The current comparison baseline is
`hw-native-sys/PTOAS:main` at `fc8db5e`; bridge workarounds remain active until
the corresponding bridge regression is rerun successfully.

| AVE instruction or contract | Compatibility gap summary | PTOAS change to watch | Current bridge handling | Fix later |
| --- | --- | --- | --- | --- |
| Signed integer `ave.hir.vsmins` / `ave.hir.vsmaxs` | Direct PTOAS VMI vector-scalar min/max is not safe for signless AVE `i32`: PTOAS normalizes the vector carrier to unsigned while leaving the scalar signless, and unsigned ordering would change signed min/max behavior. See `signed-vector-scalar-min-max-compatibility.md`. | Changes to `VMINormalizeSignlessIntToUnsigned`, VMI integer vector-scalar min/max typing, signed scalar normalization, or dynamic/partially masked scalar forms. | Emit explicitly signed vector/scalar VMI for the constant-scalar, all-active-mask form; reject dynamic scalars and partial masks. | Generalize dynamic scalars and partial masks only after a PTOAS-compatible signed scalar contract is verified. |
| `ave.hir.reduction <ADD>` / `<MAX>` | AVE reductions can keep the original vector width, while PTOAS VMI floating reductions return a one-lane vector. | Changes to unified `vcadd`/`vcmax`, legacy `reduce_*`, reduction result types, layout assignment, or reduction-combine passes that can preserve or reconstruct a wider result. | Emit the PTOAS reduction and broadcast back to the AVE result width when needed. | Revisit if PTOAS gains direct support for AVE-style reduction result shapes. |
| `ave.hir.vload <BRC_B32>` | Broadcast loads must stay on the vector pipeline for multi-lane results, while PTOAS layout assignment rejects the one-lane unified VMI broadcast-load form. | Changes to `pto.vmi.vload {dist_mode = "brc"}`, unified-to-legacy broadcast-load lowering, or layout validation for one-lane/singleton results. | Use VMI broadcast load for multi-lane results; use `pto.load_scalar` followed by `pto.vmi.broadcast` for one-lane results. | Replace the one-lane workaround if PTOAS accepts one-lane VMI broadcast loads. |
| `ave.hir.masked_store <ONEPT_B32>` | The scalar-staging path stores one vector lane into a rank-zero UB destination, which does not map as a regular vector store shape. | Changes to VMI store result-shape validation, scalar/one-point stores, singleton vectors, or rank-zero UB destinations. | Emit a zero-offset `pto.vmi.store` for the accepted rank-zero UB `f32` destination. | Revisit if PTOAS exposes a direct one-point/rank-zero VMI store contract. |

## Tracking Rules

- Explorer should distinguish proposed support, merged support, and
  bridge-validated support. A merged PTOAS change does not automatically retire
  the NPU-IR workaround.
- Mention the affected tracker row in the daily report and link the exact PTOAS
  issue, PR, branch, or commit.
- Promote an exact semantic match to `Investigate`. Use `Watch` when only a
  related pass or contract changed.
- Remove or simplify bridge handling only after the corresponding AscendNPU-IR
  conversion test and PTOAS lowering test pass on the updated baseline.

The PTOAS-side overview is `PTOAS/design/lowering-pipeline.md`; detailed bridge
decisions remain in `bridge/designs/ave-to-ptoas-vmi-conversion-design.md`.
