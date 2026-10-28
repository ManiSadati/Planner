# Deferred Work

Items here are retained for a later explicit decision, not active assignments.
Do not enable, install, merge or resume them simply because implementation or
historical handoff instructions exist. Preserve their branches and evidence.

## HOST Callbacks And Workspace Metadata

**Deferred by the project owner on 2026-10-28. Do not adopt these changes now.**
This is a documentation decision, not a rollback of existing mainline behavior.
Keep the working environments unchanged; do not install the new Triton consumer,
cherry-pick the workspace branch, change dependency pins or repair the compiler
as part of this deferral. Existing native/shared-mainline HOST support is not
being removed or disabled.

### Where The Work Lives

| Component | Preserved implementation |
| --- | --- |
| Older NPU-IR experiment | `manisadati/AscendNPU-IR:mani/set2d-ptodsl-workspace`, workspace commit `595ecac83324a96114787c3e120c573f105157fe` |
| Separate preceding SET2D work | `51fc74fe2bea09f358d56281d32066159b210d0c`; do not confuse it with the HOST export change |
| NPU-IR worktree | `$HOME/worktrees/AscendNPU-IR-set2d-ptodsl-workspace` |
| Shared-mainline inspection baseline | `wilsoncxfeng/AscendNPU-IR:master`, `a8989093f4980299dd145d5321cd9dd1e948a312` |
| Published Triton consumer | `Melikano/triton-ascend:feature/ptoas-host-metadata-mainline`, `3e9b51d0577516ffc8d60d838b61d8ba4b857c6a`, based on `461aa8f0`; not installed |

### Contract And Recorded Status

- The older workspace change separates HOST callbacks before device VMI
  conversion, preserves native symbols/helpers, lowers to CPU LLVM IR, and
  exports through `--ptoas-host-output-file`. Its 12 compiler/source/test paths
  do not change Triton, PTOAS, native templates, selection or adapters.
- Shared mainline already separates HOST code and automatically emits
  `<device-output>.host.mlir`. That artifact is host MLIR, not LLVM IR. Do not
  blindly cherry-pick the older change or introduce competing export contracts.
- The published Triton consumer lowers/links the mainline sidecar, executes
  callbacks and retains metadata results. Its report records **86 passing
  CPU/mock-device checks and seven skipped real-mainline integration cases**.
  It is not an installed or end-to-end validated baseline.
- The older compiler experiment records **70/70 compiler checks**, CPU callback
  returns **24, 14, 0 and 8**, and a BF16 callback returning **8 bytes per block**.
  That BF16 device path then hits the independent `PIPE_S` restriction. These
  results are specific to that branch, not mainline runtime evidence.
- The matching-mainline build is blocked by intrinsic IDs referenced by NPU-IR
  but absent from pinned Ascend LLVM `8ce9217f`, including `hivm_GET_COREID`,
  `hivm_SET_COND`, `hivm_vdintlvv2` and `hivm_vcaddv2_v64f32_x`. Header
  regeneration reproduced the same definitions; this is not merely a stale
  generated header. No pin change has been approved here.
- No NPU numerical matrix or physical-A5 validation establishes this feature.
  Missing or failed callback processing must not silently become workspace zero.

### When Explicitly Resumed

1. Coordinator/compiler owner: recheck revisions and compare the old workspace
   implementation with current shared mainline. Report what is already covered
   and only then identify specific missing functionality, checks or tests worth
   porting. The remaining-gap comparison has not been performed by this note.
2. Compiler/toolchain owner: establish a compatible mainline compiler,
   optimizer and translator. Propose dependency/pin changes before applying
   them; do not invent intrinsic IDs or drop native translation behavior.
3. Installer agent: keep the mainline `.host.mlir` contract unless a different
   contract is explicitly approved. Run the seven pending integration cases and
   two mainline compiler fixtures, then obtain approval and coordinate any
   candidate installation. Recheck hashes and preserve rollback environments.
4. Testing agent: after compiler/frontend integration is verified, validate
   callback-derived workspace allocation and numerical results on supported
   kernels. Track `PIPE_S` separately; physical A5 testing remains a separate gate.

### Handoffs And Evidence

- [Older compiler workspace implementation](../memory/ptoas-host-workspace-2026-10-26.md)
- [Shared-mainline contract comparison and handoff](../memory/ptoas-mainline-workspace-handoff-2026-10-27.md)
- [Published Triton consumer, checks and dependency blocker](../memory/triton-host-metadata-mainline-2026-10-28.md)
- Older compiler evidence: `$HOME/tmp/ptoas-host-workspace-20261026/`
- Publication evidence: `$HOME/tmp/set2d-branch-publication-20261027/`
- Mainline handoff evidence: `$HOME/tmp/ptoas-mainline-workspace-20261027/`
- Installer evidence: `$HOME/tmp/triton-host-metadata-mainline-20261028/`

This entry records the supplied handoffs and the owner's deferral. It does not
claim a fresh compiler review, build, installation or test run.
