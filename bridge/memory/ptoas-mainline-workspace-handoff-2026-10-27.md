# Mainline workspace frontend handoff

Installer update, 2026-10-28: [Triton implementation and dependency blocker](triton-host-metadata-mainline-2026-10-28.md).
The backend changes are published on Triton-Ascend
`feature/ptoas-host-metadata-mainline` at `3e9b51d05` and pass CPU-only checks,
but matching-mainline compiler verification and candidate installation remain
blocked. The linked integration-agent report records the branch, evidence and
dependency gap. The existing candidate and shared tools were preserved; the
source-inspection handoff below is historical.

The user is not adopting the SET2D branch now and wants Triton-Ascend workspace
integration based on the shared fork's mainline instead. No compiler/frontend
source, build, installation or runtime was changed for this handoff.

Checked `wilsoncxfeng/AscendNPU-IR:master` at
`a8989093f4980299dd145d5321cd9dd1e948a312` through the live Git remote and local
matching objects. Commit `10b4b0cb1` already introduces a different host contract:

- `--emit-ptoas-vmi -o <output>` automatically writes `<output>.host.mlir` when
  HOST functions exist. The separate pass runs before device lowering.
- The sidecar retains host MLIR, callback metadata and host helpers. It is not
  LLVM IR and still needs host lowering, translation and shared-library linking.
- The SET2D branch's `--ptoas-host-output-file` option is absent here. Its
  instructions must not be used against this mainline compiler.
- Successful callback-free compilation need not produce a sidecar; the installer
  must verify a matching compiler's contract before assigning that meaning.

Source references on that revision: regbase `PassPipeline.cpp`'s
`addPTOASHostSeparationPass`, `SeparatePTOASHostModulePass` in
`HIVMAVEToPTOASVMI.cpp`, and compiler fixture `triton/emit-ptoas-vmi-host.mlir`.
The conversion fixture `preflight-host-device-separation.mlir` includes host
helper preservation and index-to-LLVM lowering. This is source inspection,
not new test evidence against a built mainline compiler.

The personal fork `manisadati/AscendNPU-IR:master` was separately verified at
`4eb992ee81faa4e15d3756223ba74325f42953fc` and predates the bridge. Name the shared
fork explicitly when requesting mainline integration; these masters differ.

Copyable installer prompt:
`$HOME/tmp/ptoas-mainline-workspace-20261027/installer-agent-prompt.txt`.
It supersedes the SET2D-specific prompt for the user's current installation plan.
The [older SET2D workspace export](ptoas-host-workspace-2026-10-26.md) remains
published and separately validated, but is not required by this handoff.

Verification can avoid SET2D/PIPE_S: build a mixed AIC/AIV compiler fixture with
a native-named HOST workspace callback returning 16, lower/link the real host
artifact and call it using the backend's ctypes ABI. Test other sizes, zero,
callback-free output, failures, metadata caching and size-times-grid launcher
generation without NPU launches. The installer must establish the available host
lowering/translation tools on the actual mainline build; preservation of host
MLIR alone does not establish a usable callback library. Full NPU workspace
correctness remains testing-agent work with a supported kernel/sync path.
