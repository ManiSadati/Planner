# AVE to PTOAS VMI Compatibility Tracker

Last updated: 2026-08-24

This file tracks AVE instruction compatibility gaps encountered so far while
lowering AscendNPU-IR AVE to PTOAS VMI. Keep entries short; detailed analysis
belongs in separate design notes.

| AVE instruction or contract | Compatibility gap summary | Current bridge handling | Fix later |
| --- | --- | --- | --- |
| Signed integer `ave.hir.vsmins` / `ave.hir.vsmaxs` | Direct PTOAS VMI vector-scalar min/max is not safe for signless AVE `i32`: PTOAS normalizes the vector carrier to unsigned while leaving the scalar signless, and unsigned ordering would change signed min/max behavior. See `signed-vector-scalar-min-max-compatibility.md`. | Emit explicitly signed vector/scalar VMI for the constant-scalar, all-active-mask form; reject dynamic scalars and partial masks. | Generalize dynamic scalars and partial masks only after a PTOAS-compatible signed scalar contract is verified. |
| `ave.hir.reduction <ADD>` / `<MAX>` | AVE reductions can keep the original vector width, while PTOAS VMI floating reductions return a one-lane vector. | Emit the PTOAS reduction and broadcast back to the AVE result width when needed. | Revisit if PTOAS gains direct support for AVE-style reduction result shapes. |
| `ave.hir.vload <BRC_B32>` | Broadcast loads must stay on the vector pipeline for multi-lane results, while PTOAS layout assignment rejects the one-lane unified VMI broadcast-load form. | Use VMI broadcast load for multi-lane results; use `pto.load_scalar` followed by `pto.vmi.broadcast` for one-lane results. | Replace the one-lane workaround if PTOAS accepts one-lane VMI broadcast loads. |
| `ave.hir.masked_store <ONEPT_B32>` | The scalar-staging path stores one vector lane into a rank-zero UB destination, which does not map as a regular vector store shape. | Emit a zero-offset `pto.vmi.store` for the accepted rank-zero UB `f32` destination. | Revisit if PTOAS exposes a direct one-point/rank-zero VMI store contract. |
