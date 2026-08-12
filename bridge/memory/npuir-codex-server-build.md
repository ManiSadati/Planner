# AscendNPU-IR Build on the Codex Server

Last updated: 2026-08-11

## Purpose

This note records the non-A5 server workflow for building AscendNPU-IR tools
locally enough to replay saved MLIR / NPU-IR dumps.

This is separate from `bridge/memory/npu_ir_installation.md`, which is the A5
machine workflow for installing the runtime environment, running Triton Python
kernels, and producing compiler dumps from real hardware/runtime context.

## Boundary

The Codex-accessible server should be used for:

- source inspection;
- bridge pass development;
- build-only checks that do not require A5 runtime support;
- replaying saved IR files copied from the A5 workflow;
- stopping at or around `convert-hivmave-to-ave-intrin` for current bridge work.

The Codex-accessible server should not be treated as sufficient for:

- running the full Triton Python lowering workflow;
- validating A5 runtime behavior;
- proving final binary correctness or performance.

## Current Local Repo State

As of 2026-08-11:

```text
repo: $HOME/AscendNPU-IR
branch: mani/DMA
tracking: wilsoncxfeng/master
HEAD: 08031590 !2160 merge fix-something into master
```

Configured remotes:

```text
origin: git@gitcode.com:manisadati/AscendNPU-IR.git
wilsoncxfeng: git@gitcode.com:wilsoncxfeng/AscendNPU-IR.git
```

Important submodule snapshots:

```text
third-party/llvm-project: 8ce9217f83f3376970d10428de4e2d1c15039e08
third-party/shmem:        4cb0d6e0e33b555daba03da828f42f7f4e246c0d
third-party/torch-mlir:   5e9c503194b644ef53e3f788ade7afc76ce269e4
```

The LLVM submodule is expected at:

```text
$HOME/AscendNPU-IR/third-party/llvm-project
```

It is not enough to have another LLVM checkout somewhere else. Verify the
submodule snapshot with:

```bash
cd "$HOME/AscendNPU-IR"

git ls-tree HEAD third-party/llvm-project
git -C third-party/llvm-project rev-parse HEAD
git submodule status third-party/llvm-project
```

All three should agree on the pinned submodule commit.

## Submodule Strategy

The full recursive submodule update can be very heavy, especially for LLVM. The
preferred first command is still targeted and deterministic:

```bash
cd "$HOME/AscendNPU-IR"
git submodule update --init third-party/llvm-project
```

If the regular submodule update is too slow or fails due transfer problems, a
manual filtered clone of the exact LLVM branch may be used as a bootstrap, then
the same verification commands above must be run.

The local `$HOME/PTOAS/llvm-project` checkout may be used only as a git object
reference cache to avoid duplicate downloads. The build itself must use the
NPU-IR submodule path under `$HOME/AscendNPU-IR/third-party/llvm-project`.

## Build Command

Use `$HOME/tmp` for temporary and ccache files on this server. The root
filesystem can be close to full, and default ccache temp paths may be
unwritable.

```bash
cd "$HOME/AscendNPU-IR"

mkdir -p \
  "$HOME/tmp/npuir-build-tmp" \
  "$HOME/tmp/ccache" \
  "$HOME/tmp/ccache-tmp"

TMPDIR="$HOME/tmp/npuir-build-tmp" \
CCACHE_DIR="$HOME/tmp/ccache" \
CCACHE_TEMPDIR="$HOME/tmp/ccache-tmp" \
./build-tools/build.sh \
  -o ./build \
  --build-type Release \
  -j 32
```

The A5 installation note uses a Clang 15 toolchain. On this server, configure
accepted the system Clang 14 during the first build attempt. Do not assume this
is guaranteed; if the build later fails due compiler version, use the Clang 15
setup from the A5 installation note but keep temp/cache directories under
`$HOME/tmp`.

## Current Build Status

The local build completed successfully after setting `CCACHE_DIR` and
`CCACHE_TEMPDIR`.

Installed tools:

```text
$HOME/AscendNPU-IR/build/install/bin/bishengir-compile
$HOME/AscendNPU-IR/build/install/bin/bishengir-opt
```

Verified versions:

```text
bishengir-compile 1.2.0, Wilson fork commit 08031590767f, LLVM 19.1.7 8ce9217f83f3
bishengir-opt     1.2.0, Wilson fork commit 08031590767f, LLVM 19.1.7 8ce9217f83f3
```

## After Build

Use the replay wrapper:

```bash
cd "$HOME/Planner"
bash bridge/tools/replay_npuir_from_device_spec.sh
```

The replay plan is tracked in:

```text
bridge/planning/npuir-device-spec-replay.md
```

The current bridge endpoint is `convert-hivmave-to-ave-intrin`. Final compiler
stages after that are not required for the current DMA/vector mapping
investigation.
