# PTO-ISA Repo And Validation Notes

Last updated: 2026-08-10

Scope: local planning note for PTO-ISA bridge work. Do not treat this as a full
build recipe yet.

## Repo State

Local repo:

```text
$HOME/pto-isa
```

Current local branch at scan time:

```text
master
896d8ec69aaf5b623fead5afcae7a657fa784a2b feat: add overloaded TCVT_IMPL functions for enhanced type handling
```

Current remote at scan time:

```text
origin git@gitcode.com:cann/pto-isa.git
```

Important worktree note:

```text
M include/pto/npu/a5/TBinOp.hpp
```

Treat this modified file as existing user/local work. Do not revert or rely on
it as upstream truth without a separate diff review.

## Source Areas

Primary semantic references:

- `docs/mkdocs/src/manual/01-overview.md`
- `docs/mkdocs/src/manual/03-state-and-types.md`
- `docs/mkdocs/src/manual/04-tiles-and-globaltensor.md`
- `docs/mkdocs/src/manual/05-synchronization.md`
- `docs/mkdocs/src/manual/06-instructions.md`
- `docs/mkdocs/src/manual/08-virtual-isa-and-ir.md`
- `docs/PTOISA.md`
- `docs/isa/*.md`

Primary API and type references:

- `include/pto/common/pto_instr.hpp`
- `include/pto/common/type.hpp`
- `include/pto/common/pto_tile.hpp`
- `include/pto/common/event.hpp`
- `include/pto/common/tassign_check.hpp`

A5 implementation references:

- `include/pto/npu/a5/TLoad.hpp`
- `include/pto/npu/a5/TStore.hpp`
- `include/pto/npu/a5/TMov.hpp`
- `include/pto/npu/a5/TMatmul.hpp`
- `include/pto/npu/a5/TSync.hpp`
- `include/pto/npu/a5/SyncAll.hpp`
- `include/pto/npu/a5/MGather.hpp`
- `include/pto/npu/a5/MScatter.hpp`

## Validation Layers

Local checks that may be possible on this server:

- CPU simulator functional tests.
- Documentation consistency scripts under `docs/tools/`.
- CostModel tests if dependencies are present.
- Static source checks and focused compile checks.

Commands documented by the repo:

```bash
python3 tests/run_cpu.py --clean --verbose
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
./build.sh --run_all --a3 --sim
```

A5/NPU validation likely needs the human hardware loop:

- NPU ST tests under `tests/npu/a5/`.
- A5 manual kernels under `kernels/manual/a5/`.
- Runtime validation requiring CANN/BiSheng and Ascend hardware or simulator.
- Sync-heavy cases involving `SYNCALL`, mixed AIC/AIV behavior, or cross-core
  launch metadata.

When proposing validation, separate:

```text
local CPU-sim/static checks
compile-only or simulator checks with CANN/BiSheng
A5 hardware checks requiring human feedback
```

## Bridge Development Rules

- Check both the per-op doc and `include/pto/common/pto_instr.hpp` before using
  an instruction as a mapping target.
- For A5-specific behavior, check the `include/pto/npu/a5/` implementation
  header. CPU simulator behavior can be more permissive or semantically simpler.
- Preserve `TileType`, dtype, rows/cols, valid rows/cols, layout, boxed/fractal
  layout, GlobalTensor shape/stride/layout, and event dependency information.
- Do not map NPU-IR `set_flag`/`wait_flag` to PTO events unless the
  source/destination pipe pair and token lifetime are explicit.
- Do not map NPU-IR `sync_block*` to `SYNCALL` unless participant-set semantics
  and cross-core visibility assumptions match.
- For cube rows, start with `TMATMUL*` plus surrounding `TLOAD`/`TMOV`/`TSTORE`
  movement, not VMI.
- For ND-to-NZ rows, start with `TLOAD` Mat layout paths and `TMOV` ND-to-NZ
  paths before considering vector-only lowering.
