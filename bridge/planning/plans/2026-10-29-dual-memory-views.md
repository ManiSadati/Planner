# Dual memory views in AVE-to-VMI

User-approved design: keep explicit PTO pointers for VMI users and preserve
memref descriptors where native calls require their ABI. Work in the existing
clean feature checkout; no commits, PTOAS edits, or finalizer pass.

The problem is that preserved memrefs obscure even literal UB addresses from
downstream alignment analysis. Ranked solutions: recover both views in the
bridge (selected); change downstream proof analysis (excluded).

Recover aligned base pointers and separate descriptor offsets through known
pointer-backed casts, reinterpret views, subviews, metadata bases and selects.
A reinterpret offset replaces the descriptor offset; a subview adds offsets
scaled by source strides. Keep native call descriptors unchanged. Unknown
provenance must retain extraction; do not invent alignment facts.

For directly called private outlined vector helpers, use pointer parameters for static
identity-layout memref arguments only when all callers provide a proven
32-byte-aligned base. Reconstruct the static descriptor inside the helper for
native calls. Do not change kernel or external signatures, address-taken
functions, or unproven arguments. Arbitrary dynamic descriptor ABI
specialization is outside this bounded change.

- [x] Add failing regressions for shared buffers, view offsets, outlined
      helpers with descriptor consumers, and misaligned/unproven boundaries.
- [x] Implement pointer-view recovery and guarded helper specialization in the
      existing pass; remove dead descriptor scaffolding.
- [x] Rebuild the private converter, run focused/existing conversion tests,
      and replay reduced MmadL1 behavior into pinned PTOAS VPTO.
- [x] Document compositions, conservative fallbacks and evidence.

Compilation only. Private reduced-driver converter build plus retained pinned
PTOAS CLI; no full production rebuild, runtime or numerical certification.


Review rulings:
- Only private helper signatures may change. A generated-looking public name
  is insufficient ABI evidence; public helper captures can remain blocked.
- Record aligned arguments only after all direct callers pass proof, allowing
  private helper chains without assuming arbitrary pointer arguments aligned.
- Preserve argument descriptors when pointer element normalization would make
  descriptor reconstruction type-incompatible (for example i8 versus ui8).

The private-only restriction above was subsequently superseded by explicit
human direction to convert compiler-owned outlined VF definitions and callers,
even if source-public. See the [follow-up plan](2026-10-29-outlined-memory-views.md).
