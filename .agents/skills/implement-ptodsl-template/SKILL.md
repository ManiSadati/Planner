---
name: implement-ptodsl-template
description: Implement or generalize an AscendNPU-IR PTODSL template instance, including offline MLIR generation, its existing argument adapter, registration, and focused compiler checks.
---

# Implement a PTODSL Instance

Read [Planner's contract](../../../AGENT.md), its required reading, the
[current plan](../../../bridge/planning/README.md), and the
[development guide](../../../bridge/designs/ptodsl-template-development-and-fallback.md).
Resolve the code checkout from `ASCEND_NPU_IR_ROOT` or verify the documented
checkout. Confirm the assigned instance and files before editing shared code.

## Implementation

1. Establish the exact native instance name and CCE contract from current source
   and native-flow IR. Export `__pto_` followed by that native name; reuse native
   selection instead of inventing another shape-based instance search.
2. Author computation/transfers in the appropriate PTODSL Python generator.
   Preserve native layout, tails, real versus padded dimensions, offsets and
   stride units, init/accumulate behavior, precision, and event semantics.
   Do not duplicate the compute body in a C++ rewrite.
3. Use the current per-template adapter design. Preserve the metadata actually
   needed by this implementation; do not collapse memrefs to bare pointers and
   lose offsets or dimensions. Native-ABI resolver entries need compatible
   wrappers, not an assumed automatic adapter.
4. Generate the instance MLIR offline, inspect its exported definition, and add
   it to the appropriate CMake resource/install list. Normal compilation parses
   installed MLIR and imports helpers; it must not execute the Python generator.
5. Add focused selection/import, ABI, and failure tests. Rebuild/install matching
   compiler resources before checking generated IR. Report numerical testing
   separately; use the existing runtime test suite when requested.

## Boundaries

- Missing exact-name resources may retain native calls for explicit fallback;
  a present but invalid implementation must fail, not silently fall back.
- Keep uniform descriptor-ABI migration deferred unless explicitly requested.
  Do not change PTOAS, native templates, or submodule pins incidentally.
- Agree on exclusive ownership before editing shared generators, CMake lists,
  selection passes, or install prefixes alongside another agent.
- Return changed paths, the native/PTODSL symbol pair, supported contract,
  generation command, verification results, and remaining limitations.
