---
name: investigate-native-template
description: Trace an AscendNPU-IR native template's selection, CCE implementation, and IR contract before planning a PTODSL replacement or diagnosing a mismatch.
---

# Investigate a Native Template

Start with [Planner's contract](../../../AGENT.md), its required reading, and
the [current plan](../../../bridge/planning/README.md). These links are relative
to this skill, not the shell's working directory. Resolve the code checkout
from `ASCEND_NPU_IR_ROOT`, or verify `$HOME/AscendNPU-IR` before using it.

## Trace

1. Record the checkout revision and requested operation or exact native symbol.
   Find its selection/lowering code and implementation under
   `bishengir/lib/Template/lib/RegBase/`; do not infer behavior from names alone.
2. Follow one concrete native IR instance through selection and its call ABI.
   Distinguish the structured HIVM op, native function name, C-interface wrapper,
   and bitcode definition. They are not interchangeable symbol identities.
3. Read the implementation's conditions and helpers. Identify datatype,
   transpose/precision, logical extents, padded layout, offsets/strides,
   accumulation, scratch storage, and synchronization requirements.
4. Compare against available PTO/PTODSL operations and the current adapters.
   Separate what maps directly, what needs template logic, and what is missing.
   Consult the [development guide](../../../bridge/designs/ptodsl-template-development-and-fallback.md)
   before proposing selection or fallback changes.

## Evidence and Scope

- Return exact symbol names, source file/line references, a short native IR
  excerpt, and a contract-to-PTO mapping with gaps and risks.
- Label source inspection, compile evidence, and numerical evidence separately.
  An old dump is evidence only for its recorded revision/configuration.
- Investigation is read-only unless implementation or evidence capture was
  requested. Do not update submodules, patch native CCE templates, run large
  simulations, or change dispatch architecture as part of this skill.
- Independent investigations can run in parallel; give each worker a distinct
  instance/family and return findings rather than editing shared files.
