# AVE to PTOAS VMI Compatibility Tracker

Last updated: 2026-09-03

This file tracks AVE instruction compatibility gaps encountered so far while
lowering AscendNPU-IR AVE to PTOAS VMI. Keep entries short; detailed analysis
belongs in separate design notes.

This is also a standing input to Explorer. A PTOAS branch, issue, PR, or
upstream commit matching one of the watch triggers below is bridge-relevant
even when it does not mention AscendNPU-IR. The last broad vector-kernel
comparison baseline is `hw-native-sys/PTOAS:main` at `fc8db5e`. This review also
compared the AscendNPU-IR PTO snapshot and bridge at `57175b6c`, including the
2026-09-02 `sync_block_set`/`sync_block_wait` conversion under review, with
PTOAS at `bdcb319d6`. That comparison is evidence of interface drift, not a
replacement for end-to-end regression. Bridge workarounds remain active until
the matching conversion and PTOAS regression both pass.

| AVE instruction or contract | Compatibility gap summary | PTOAS change to watch | Current bridge handling | Fix later |
| --- | --- | --- | --- | --- |
| Signed integer `ave.hir.vsmins` / `ave.hir.vsmaxs` | Direct PTOAS VMI vector-scalar min/max is not safe for signless AVE `i32`: PTOAS normalizes the vector carrier to unsigned while leaving the scalar signless, and unsigned ordering would change signed min/max behavior. See `signed-vector-scalar-min-max-compatibility.md`. | Changes to `VMINormalizeSignlessIntToUnsigned`, VMI integer vector-scalar min/max typing, signed scalar normalization, or dynamic/partially masked scalar forms. | Emit explicitly signed vector/scalar VMI for the constant-scalar, all-active-mask form; reject dynamic scalars and partial masks. | Generalize dynamic scalars and partial masks only after a PTOAS-compatible signed scalar contract is verified. |
| `ave.hir.reduction <ADD>` / `<MAX>` | AVE reductions can keep the original vector width, while PTOAS VMI floating reductions return a one-lane vector. | Changes to unified `vcadd`/`vcmax`, legacy `reduce_*`, reduction result types, layout assignment, or reduction-combine passes that can preserve or reconstruct a wider result. | Emit the PTOAS reduction and broadcast back to the AVE result width when needed. | Revisit if PTOAS gains direct support for AVE-style reduction result shapes. |
| `ave.hir.vload <BRC_B32>` | Broadcast loads must stay on the vector pipeline for multi-lane results, while PTOAS layout assignment rejects the one-lane unified VMI broadcast-load form. | Changes to `pto.vmi.vload {dist_mode = "brc"}`, unified-to-legacy broadcast-load lowering, or layout validation for one-lane/singleton results. | Use VMI broadcast load for multi-lane results; use `pto.load_scalar` followed by `pto.vmi.broadcast` for one-lane results. | Replace the one-lane workaround if PTOAS accepts one-lane VMI broadcast loads. |
| `ave.hir.masked_store <ONEPT_B32>` | The scalar-staging path stores one vector lane into a rank-zero UB destination, which does not map as a regular vector store shape. | Changes to VMI store result-shape validation, scalar/one-point stores, singleton vectors, or rank-zero UB destinations. | Emit a zero-offset `pto.vmi.store` for the accepted rank-zero UB `f32` destination. | Revisit if PTOAS exposes a direct one-point/rank-zero VMI store contract. |
| Partial `pge <VL*>` masks on AVE arithmetic | PTOAS accepts masks on unified VMI arithmetic, but its unified-to-legacy expansion currently discards masks for binary operations and most unary operations. The bridge does not enforce the narrow condition under which this is harmless. | Changes to unified VMI arithmetic expansion, inactive-lane/passthrough semantics, `pmode`, or native masked VPTO arithmetic. | Preserve the AVE mask in emitted VMI. This is only known safe when inactive lanes are never observed, such as a straight-line tail whose final store uses the same mask. | Either make PTOAS preserve inactive-lane semantics or add a proven bridge expansion. Until then, reject partial-mask arithmetic unless use analysis proves inactive lanes cannot escape. |
| Stored scalar `f32 -inf` | The validated softmax simulator path produced invalid results when the exact negative-infinity initializer was stored through `pto.store_scalar`. Replacing it changes IEEE behavior for `-inf` and NaN inputs. | Changes to scalar constant/store lowering, VPTO/LLVM constant materialization, and simulator handling of floating infinities. | Rewrite any `f32 -inf` that directly feeds `pto.store_scalar` to `-FLT_MAX`. The implementation is broader than the intended reduction-initializer use. | Remove the rewrite after PTOAS carries `-inf` correctly. Before then, constrain it to a proven max-reduction initializer and add exceptional-value tests. |
| `hivm.hir.sync_block_set` / `sync_block_wait` | The source carries core kind, producer and consumer pipes, sync mode, and an optional FFTS base operand. The vendored PTO `sync.wait` cannot carry `ffts_mode`, and neither target op carries the FFTS base. Current PTOAS also restricts A5 static FFTS IDs to `[0, 15]`, while the vendored verifier and bridge allow `[0, 31]`. | Changes to `SyncSetOp`/`SyncWaitOp`, named `set_cross_block`/`wait_cross_block` and `set_intra_block`/`wait_intra_block` ops, FFTS event ranges, and A5 sync lowering. | Emit `pto.sync.set` on the source `tpipe` and `pto.sync.wait` on the source consumer `pipe`; carry the numeric mode only on set; ignore core kind, the opposite pipe, and `ffts_base_addr`; accept static IDs through 31. IDs used by the current smoke test reach PTOAS VPTO, but the general mapping is not proven. | Refresh the PTO snapshot, preserve matching set/wait mode, enforce the target-specific ID range, and map to the named operation when appropriate. Reject any source FFTS base or core/pipe form whose semantics cannot be represented and verified. |
| Vendored PTO dialect snapshot | AscendNPU-IR compiles against a copied PTO dialect while the emitted text is consumed by a separately evolving PTOAS. The reviewed trees already differ in sync operations/verifiers and VMI memory operands, so local verification can accept IR that current PTOAS rejects or interprets differently. | Any PTOAS IR, parser, verifier, builder, normalization, or pipeline change; currently especially sync mode/event changes and removal of VMI memory `repeat_stride`. | Keep using the in-tree snapshot so AscendNPU-IR and its MLIR version remain build-independent. Compatibility is checked later by invoking PTOAS on emitted text. | Record an exact PTOAS source revision for each snapshot, refresh headers and implementations as one unit, and add a CI test that sends representative bridge output through that exact PTOAS revision. |
| Memref descriptor to `!pto.ptr` collapse | Dynamic memref offsets, sizes, and strides are not representable in the pointer-only VMI ABI. The bridge treats converted arguments as normalized logical base pointers and may synthesize zero for unavailable metadata. This is correct only if the preceding NPU-IR pipeline has already incorporated the descriptor offset and unused metadata stays unused. | PTO pointer/view ABI, descriptor-bearing PTO types, dynamic-shape materialization, or VMI memory addressing changes. | Accept a narrow rank-one/unit-stride path, reconstruct explicit `pto.addptr` offsets where available, and reject several used dynamic-size/noncontiguous forms. | State and verify the normalized-base invariant at the pass boundary, reject any used metadata that cannot be reconstructed, or adopt a descriptor-preserving PTO ABI. Add nonzero-offset and dynamic-layout end-to-end tests. |
| External CCE-call memref ABI | External Cube templates require ranked memrefs while the PTO kernel boundary is pointer-based. The resulting pointer-to-memref `unrealized_conversion_cast` and `pto.preserve_external_call_memrefs` marker are bridge-private conventions, not a general PTOAS VMI contract. | PTOAS external-call ABI, CCE template integration, memref materialization, and unknown/residual cast handling. | In external-call mode, preserve PTO-address-space memrefs internally, change kernel arguments back to PTO pointers, and leave controlled pointer-to-memref casts at entry. | Replace bridge-private residual casts with an explicit supported ABI/materialization operation, or retire this compatibility route once the PTO-native Cube path covers the required kernels. |
| PTO module target/kernel metadata | PTOAS `pto.kernel_kind` has only `cube` and `vector`; mixed VPTO is represented structurally and normalized into one child module per kind. The bridge still hard-codes `pto.target_arch = "a5"`. | PTO target/device-spec schema, function core-kind rules, section splitting, and mixed Cube/Vector module packaging. | Preserve `hivm.func_core_type` through conversion. Pure modules receive one module-level kind. Mixed modules omit the outer kind and wrap every defined AIC/AIV function in `pto.section.cube`/`pto.section.vector`; PTOAS then creates the canonical child modules. Reject unsplit `MIX`, ambiguous, or unclassified definitions. | Derive the target from the confirmed source device specification. Revisit the single-block/void-function restriction if a mixed kernel reaches the bridge with a more general CFG or result ABI. |

## Review Priority

1. Resolve the sync snapshot/range/mode mismatch and silent source-field loss
   before treating FlashAttention block synchronization as complete.
2. Prevent partial arithmetic masks from reaching a PTOAS path that discards
   them unless inactive lanes are proven unobservable.
3. Narrow and ultimately remove the `-inf` to `-FLT_MAX` rewrite before claiming
   IEEE exceptional-value coverage.
4. Automate PTO dialect snapshot provenance and bridge-output compatibility
   testing; local AscendNPU-IR verification is not sufficient.
5. Generalize pointer ABI and target metadata only when kernels outside the
   currently proven A5 normalized-base contract require it.

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
