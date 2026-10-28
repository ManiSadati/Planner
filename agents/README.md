# Project Agent Roles

Each role has one maintained `AGENT.md` entry point. Shared workflows remain in
`.agents/skills/`; do not duplicate skills into every role folder.

| Role | Entry point | Codex agent name |
| --- | --- | --- |
| Investigator / Coordinator | [AGENT.md](investigator-coordinator/AGENT.md) | `investigator_coordinator` |
| AVE-to-VMI Developer | [AGENT.md](ave-vmi-developer/AGENT.md) | `ave_vmi_developer` |
| Template Developer | [AGENT.md](template-developer/AGENT.md) | `template_developer` |
| Triton-Ascend / Installer | [AGENT.md](triton-ascend-installer/AGENT.md) | `triton_ascend_installer` |
| Test Author | [AGENT.md](test-author/AGENT.md) | `test_author` |
| Validator | [AGENT.md](validator/AGENT.md) | `bridge_validator` |
| CI Runner | [AGENT.md](ci-runner/AGENT.md) | `ci_runner` |

## Starting a Role

For an ordinary session, give the agent its role file and a bounded assignment:

```text
Read agents/investigator-coordinator/AGENT.md in Planner.
Investigate F16 transpose-A MmadL1 support. Do not modify compiler code or
run simulations. Report the native contract, missing implementation, and plan.
```

For conversion support from logs or MLIR, use the AVE-to-VMI Developer:

```text
Read agents/ave-vmi-developer/AGENT.md in Planner.
Diagnose missing AVE-to-VMI support from <logs-or-MLIR-path> using
ASCEND_NPU_IR_ROOT=<checkout>. Identify the exact cases and rejected contracts,
then present the smallest fix, files, and compilation-only checks for approval.
After approval, implement and validate in the agreed branch. Preserve existing
changes, keep deferred workarounds deferred, and do not stage or commit.
```

The same role can complete an already approved implementation assignment; pass
the agreed scope, base branch, owned files, and verification budget explicitly.

For Triton-Ascend/environment work, use the combined installer role:

```text
Read agents/triton-ascend-installer/AGENT.md in Planner and its primary skill.
Audit the installed Triton-Ascend version and current bridge patch against the
matching 3.6 source. Propose the minimal native/bridge option-sharing changes
and installation plan. Do not install or modify code yet. Ask before removing
any existing patch hunk or behavior.
```

The small [Codex definitions](../.codex/agents/) explicitly tell custom agents
to read these files. Role instructions live here; those TOML files are loaders.
`AGENT.md` is our chosen explicit entry point, not a claim that Codex discovers
arbitrary role files automatically. Codex's conventional automatic guidance
filename is `AGENTS.md`; custom agents use standalone TOML definitions. See
[official agent documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents)
and [instruction discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
These definitions are project-local to Planner; sessions launched in sibling
repositories need the role file supplied explicitly or a separate installation.

## Shared Contract

- Read [Planner's contract](../AGENT.md), its required reading, and the
  [current plan](../bridge/planning/README.md). Current human direction wins over
  role defaults. Roles do not grant permissions beyond the assigned task.
- Read primary skills at task start, and supporting skills only when relevant.
  Follow linked files explicitly if they are not in the session's skill list.
- Before local build/runtime work, read the short
  [current environment note](../bridge/memory/current-local-environment.md).
  Use the existing activated profile; installation guides are not routine setup.
- Assign the objective, repository/revision, owned files, dependencies, permitted
  checks, runtime budget, and expected output. Missing authorization for broad
  design changes or expensive runs is not permission to proceed.
- Parallelize independent work only. Shared generators, selection passes,
  resource lists, toolchain builds/installs, and submodule updates need a single
  agreed writer. Separate worktrees do not isolate a shared install prefix.
- Keep logs/caches private to each run. Do not switch another worker's branch,
  replace its tools, revert its edits, or kill unrelated processes.
- Investigate intended behavior and implementation, environment, and design
  causes. The first failing stage is a symptom location, not automatically the
  place to patch. Escalate consequential ABI or architecture changes with
  evidence and options instead of silently implementing them.
- Report revisions, selected symbols, commands, evidence paths, results and
  limitations. Distinguish native, fallback, PTODSL, simulator, and A5 results.
  No commits/pushes, global configuration changes, or `human/` edits without
  the corresponding explicit request.

## Superpowers

Use selected Superpowers files as optional process references, not a second
project contract. Resolve `SUPERPOWERS_ROOT`, defaulting to `$HOME/superpowers`,
then read `skills/<name>/SKILL.md` there when the role's trigger applies.
The local checkout reviewed for this setup was `bb92a77`; these references are
not vendored, globally installed, or version-pinned by Planner. If unavailable,
say so and follow the project skills; do not silently clone/install anything.

| Reference | Read when | Project-specific boundary |
| --- | --- | --- |
| `systematic-debugging` | Unexpected behavior needs causal investigation. | Consider design from the beginning; do not wait for three failed patches. Investigation does not authorize code edits. |
| `dispatching-parallel-agents` | Independent tasks can safely be delegated. | Respect shared toolchain ownership and the assigned test budget; do not automatically run a full simulator suite afterward. |
| `verification-before-completion` | Preparing a result or completion claim. | Tie evidence to revision, command, environment, and scope. Label historical results; do not rerun costly suites merely to answer a status question. |
| `requesting-code-review` | An independent review of a substantial change is useful or requested. | Supply requirements, exact diff/base, and evidence. Do not assume a reviewer tool exists or invent a review that did not happen. |

Do not follow framework bootstrap, blanket activation, unrelated skill installs,
automatic merges, or destructive reset/revert instructions. Read supporting
Superpowers files only when needed; copying a reference's workflow does not
expand task scope. These roles inherit session permissions, not separate
role-enforced sandboxes. No workers or scheduled jobs start from these files.
