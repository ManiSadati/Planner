# Q2 native missing-template reproduction

This bundle replays one saved Q2 kernel compilation that emitted native
template calls and then failed to resolve their implementations at device
linking. It includes the actual compiler input, so the recipient does not need
the original machine, benchmark checkout, virtual environment, Triton, PyTorch,
pytest, or benchmark harness. It executes no kernel or simulator.

The recipient needs Linux, Python 3.8+, an installed `bishengir-compile` with
its native template libraries, and a compatible CANN A5 device compiler.
Those toolchain dependencies are not bundled. A newer compiler may reject the
old IR/options or fix the missing symbols; both outcomes are useful evidence.

## Run on another machine

Copy this **whole directory**, or obtain it through a Planner checkout. From
this directory:

```bash
python3 run.py \
  --compiler /path/to/npuir/install/bin/bishengir-compile \
  --cann-root /path/to/cann \
  --output /path/to/new/repro-output
```

Supply local paths. `--cann-root` must contain `set_env.sh`; the runner sources
it in a child shell and makes the usual CANN compiler directories available.
If the local compiler/CANN environment is already configured, omit
`--cann-root`. Any additional loader configuration needed by that installation
must already be present in the invoking environment.

`--output` must name a new directory. Omit it to create a fresh system temporary
directory. The runner prints its location. `--dry-run` prints the command
without executing any tools or creating outputs. `--timeout 180` is the
default compiler watchdog, in seconds; the compiler process group is killed
on timeout or interruption.

The recorded target is **`Ascend910_9589`**, passed to the A5 backend in the
historical toolchain. This is the exact historical spelling. If a newer
compiler requires another spelling, pass `--target LOCAL_TARGET`; the result
records that override. Do not silently substitute an A3 target or its bitcode.

The runner clears both known PTOAS bridge-enabling environment variables and
uses the native path. It does not build, install, patch or fetch a toolchain.

## Exact source configuration

| Property | Recorded value |
|---|---|
| Repository revision | Q2TritonKernel `ebf78b205a87ecc5cd935a214bd2ef0fc176dcd9` |
| Test | `tests/fla_perf/test_solve_tril_perf.py::test_solve_tril_perf[B1-T1024-H64-chunk_size64]` |
| Parameters | B=1, T=1024, H=64, chunk_size=64, float32 |
| Selected JIT kernel | `merge_16x16_to_64x64_inverse_kernel` |
| Kernel source | `src/kernels/fla/ops/utils/solve_tril.py` |
| Launch controls | `DOT_PRECISION="ieee"`, `num_stages=2`, `multibuffer=True`, `sync_solver=True`, `USE_TMA=False`, local-buffer multibuffer limit `"no-limit"` |
| Autotuning | None |
| Validation scope | One kernel compilation from a performance test; no numerical claim |

The saved early IR is the exact input to `bishengir-compile` for this kernel,
not a reduced synthetic example. Its SHA256 is
`5b294ffbfec7967161b63ee0f99236505fd78d47716ca8063cbdac9fa4fdcbfc`.
The runner verifies it against [case.json](case.json). Runtime scalar operands
remain in the input, as in the original compilation; no runtime tensors are
needed to compile it. This does not rerun the complete Python test.
The input preserves the compiler-emitted whitespace so its historical hash
remains unchanged.

## What the historical failure established

After `convert-hivm-to-std`, native IR contained calls including:

```mlir
func.call @transpose_3d_without_last_axis_float(...)
func.call @vadd_1d_float(...)
```

Operand lists are abbreviated here. The device linker then reported:

```text
ld.lld: error: undefined symbol: _mlir_ciface_transpose_3d_without_last_axis_float
ld.lld: error: undefined symbol: _mlir_ciface_vadd_1d_float
```

See [historical-errors.txt](historical-errors.txt). The `_mlir_ciface_` names
are C-interface wrappers for the corresponding native calls. This establishes
unresolved native implementations in the selected historical link setup. It
does **not** establish that every upstream release lacks those implementations,
or that every installed bitcode bundle was exhaustively inspected. The local
audit could not read one CANN AIV bundle with the available LLVM version; that
tool error was not treated as proof of missing definitions.

Historical replay provenance (2026-10-16):

| Component | Revision |
|---|---|
| NPU-IR | `3ed2ee73764157ee39f2e0650c24fb740c0ac68b` |
| Ascend LLVM | `8ce9217f83f3376970d10428de4e2d1c15039e08` |
| Pinned PTOAS | `a7ed88b81eaacee966672f0097584e76e06b8b8d` |
| PTO LLVM | `fa2fd1f75d742fec91ecf40d98c9c7961647ec42` |
| CANN | `9.1.0-beta.3` |

Compiler and adjacent c310 bitcode hashes are preserved in `case.json`.
Provenance separates source revisions from installed products: matching a
Git revision alone does not guarantee matching compiler/library contents.
PTOAS was pinned in that checkout but its bridge was disabled for this run.

## Read and return the result

The output directory contains:

- `compile.log`: complete compiler output, including requested before/after
  HIVMToStandard modules and any linker errors. Several internal compiler
  attempts may appear; repeated dumps are not separate test executions.
- `result.json`: compiler exit, outcome, unresolved symbols, native call names
  seen in dumps, and whether the requested dump markers appeared.
- `command.json` / `command.txt`: exact executed command, including local paths
  and any CANN environment setup.
- `toolchain.json`: actual compiler version/hash, supplied CANN root, and
  hashes of adjacent installed c310 bundles where present. Adjacent files are
  inventory evidence, not proof of which bundle the backend selected.
- `case.json` and `input.ttadapter.mlir`: the bundled selection and input.

Return `result.json`, `toolchain.json`, `command.json` and `compile.log`.
The script returns the compiler exit code, including nonzero when the expected
link failure is reproduced. Interpret `outcome` separately:

| Outcome | Meaning |
|---|---|
| `reproduced_missing_template_link_failure` | At least one historical missing-symbol diagnostic appeared during a failed compilation; inspect the exact symbol list. |
| `linked_successfully` | This compilation succeeded with the recipient's toolchain; inspect dumps to see whether the same calls were emitted. |
| `other_failure` | Compilation/setup failed for another reason, including incompatible IR/options or missing local tools. |
| `timeout` / `interrupted` | Compilation did not finish; available logs are retained. |

Missing dumps or failure before lowering are **unknown coverage**, not proof
that a template is absent. Linking success alone is not numerical validation.

## Bundle validation

The input hash, path portability, dry run, CANN setup with paths containing
spaces, outcome classification, and process-group timeout cleanup were checked
using a dummy compiler. The original native failure is historical evidence;
this publication did not rerun real compilation, kernels or campaigns.
