# Current Local Test Environment

Updated: 2026-10-28. Everyday guidance for agents on this server, not an
installation guide for other servers.

Use the existing Triton-Ascend 3.6 candidate for native NPU-IR and PTOAS bridge
development/testing:

```bash
source "$HOME/.venv/npuir-sim-3.6-candidate/bin/activate"
```

| Purpose | Selected CANN |
| --- | --- |
| Compilation: NPU-IR, PTOAS and device binaries | **9.2** |
| Torch-NPU runtime and simulator | **9.1.0-beta.3** |

Activation already selects the matching compiler wrappers and runtime paths.
The wrappers use CANN 9.2 in compiler children; the parent Python/msprof process
stays on CANN 9.1. Do not source CANN 9.2 into that runtime process, bypass the
wrappers with old manual tool-path exports, reinstall Triton, or create another
environment for routine work. Simulator target remains `Ascend950PR_9589`.
Native compilation is the activation default; bridge selection remains explicit.

This profile selects the installed toolchain from NPU-IR `3bf5d9898`, not
automatically whichever source branch an agent is editing. Record actual tool
provenance; source edits require a matching, coordinated build before testing.
Keep shared tools stable while other agents run. Report missing setup or a
revision mismatch rather than silently switching environments.

The installer reports exact native/bridge vadd simulator passes and 46 passing
frontend checks. This is not a complete template matrix or physical-A5 result.
HOST callback consumption remains [deferred](../planning/deferred-work.md#host-callbacks-and-workspace-metadata)
and uninstalled; the toolchain refresh did not enable it.

Use the [refresh handoff](toolchain-refresh-cann92-2026-10-28.md) only when exact
revisions, validation evidence, wrapper internals or rollback details are needed.
Other servers need their own verified setup; this local activation path is not
an instruction to install or migrate a coworker's environment.
