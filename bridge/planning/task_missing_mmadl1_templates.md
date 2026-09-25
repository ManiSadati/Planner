# Missing MmadL1 Templates Implementation Plan

> **For the implementer:** Complete Task 1 and Task 2 as separate changes. Run
> each task's red/green test cycle before starting the next task. Do not combine
> unrelated ND2NZ, Fixpipe, I4, bias, transpose-A, or unit-flag work with this
> assignment.

**Goal:** Add the two smallest missing MmadL1 PTODSL specializations:
`__pto_mmadl1_bf16_f32_tb` and `__pto_mmadl1_f32_f32_ieee_nn`.

**Architecture:** Both helpers must reuse the existing shared
`_mmadl1_body`; this task adds typed wrappers and checked-in MLIR resources,
registers those resources with the importer and CMake, and extends the C++
MmadL1 matcher to select them. The implementation must not duplicate the Cube
load/MAD sequence.

**Tech stack:** Python PTODSL, MLIR, C++, CMake, LLVM lit/FileCheck, PTOAS,
and the Planner bridge test runner.

**Context:**
[PTOAS failure categorization](ptoas_failure_categorization.md), especially
“CCE Template Rewrite Feasibility in PTODSL/PTO.”

## Expected impact

| Template | Affected tests | Expected result |
|---|---|---|
| `__pto_mmadl1_bf16_f32_tb` | 10 `test_chunk_scaled_dot_kkt_fwd_perf` shapes | Removes their shared MmadL1 specialization failure. This is the only failure currently recorded for this logical testcase, so all 10 should be rerun. |
| `__pto_mmadl1_f32_f32_ieee_nn` | 20 solve-tril shapes | Removes the FP32 IEEE MmadL1 specialization failure. Some cases are expected to continue failing at ND2NZ or Fixpipe; advancement to those later failures still validates template selection. |

## Global constraints

- Keep the two implementations separately reviewable and independently
  testable.
- Reuse `_mmadl1_body`; do not copy its event, L1-to-L0, or MAD sequence.
- Preserve the current normalized M/K/N contract of `[1, 64]`.
- Do not change `enable_I4`, transpose-A, bias, result-tensor, or unit-flag
  handling.
- Generated files under `PTODSL/instantiations/` are checked-in build
  resources. Generate them from the Python source; do not hand-edit them.
- BF16-TB must pass `b_transpose=True` to `_mmadl1_body`.
- FP32-IEEE must leave `tf32_mode` unset. It must not use
  `pto.Tf32Mode.ROUND_EVEN` or select the existing HF32 helper.

## Prerequisites

Run commands from `/home/m00967009/Workspace` unless a step says otherwise.
Set these paths once:

```bash
export WORKSPACE_ROOT=/home/m00967009/Workspace
export ASCEND_NPU_IR_ROOT="$WORKSPACE_ROOT/AscendNPU-IR"
export PLANNER_ROOT="$WORKSPACE_ROOT/Planner"
export Q2_ROOT=/home/x84315859/Q2TritonKernel
export PTOAS_BIN="$WORKSPACE_ROOT/PTOAS/build/tools/ptoas/ptoas"
export PYTHONPATH="$ASCEND_NPU_IR_ROOT/third-party/ptoas/ptodsl:$ASCEND_NPU_IR_ROOT/build-ptoas/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ASCEND_NPU_IR_ROOT/build/bin:$PATH"
```

If the actual Q2 checkout is elsewhere, change only `Q2_ROOT`. Confirm the
tools before editing:

```bash
test -x "$ASCEND_NPU_IR_ROOT/build/bin/bishengir-opt"
test -x "$ASCEND_NPU_IR_ROOT/build/bin/llvm-lit"
test -x "$PTOAS_BIN"
python3 -c 'from ptodsl import pto; print("ptodsl import: OK")'
```

## Shared file map

| File | Responsibility |
|---|---|
| `AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py` | PTODSL specialization metadata, typed wrappers, probe generation, and explicit-resource generation |
| `AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/PTODSL/instantiations/*.mlir` | Checked-in generated helper bodies loaded by `bishengir-compile` |
| `AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/PTODSLTemplateImporter.h` | Stable helper symbol constants |
| `AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/PTODSLTemplateImporter.cpp` | Resource filename, symbol, and instantiation registration |
| `AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/MatmulRegionToPTO.cpp` | MmadL1 contract matching and helper selection |
| `AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/CMakeLists.txt` | Copies and installs generated resources |
| `AscendNPU-IR/bishengir/test/Conversion/HIVMTemplatesToPTO/mmadl1-specializations-ptodsl.mlir` | Targeted conversion and resource-import regression tests |

---

## Task 1: BF16 inputs, FP32 accumulation, transposed B

**Deliverable:** `hivm.hir.mmadL1` with BF16 inputs, FP32 accumulation,
`b_transpose`, and no HF32 flag converts to
`@__pto_mmadl1_bf16_f32_tb`.

**Closest references:**

- Use `mmadl1_bf16_f32` for pointer and accumulator types.
- Use `mmadl1_f16_f32_tb` for transpose behavior.
- Do not implement new PTO operations; BF16 MAD and the required L0B layout
  path already exist.

### 1.1 Write the failing conversion test

Create
`AscendNPU-IR/bishengir/test/Conversion/HIVMTemplatesToPTO/mmadl1-specializations-ptodsl.mlir`
with this first test:

```mlir
// RUN: bishengir-opt %s --convert-hivm-templates-to-pto='use-ptodsl-template=true' | FileCheck %s

#cbuf = #hivm.address_space<cbuf>
#cc = #hivm.address_space<cc>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @mmadl1_bf16_f32_tb() {
    // CHECK-LABEL: func.func @mmadl1_bf16_f32_tb
    // CHECK: call @__pto_mmadl1_bf16_f32_tb
    // CHECK-NOT: hivm.hir.mmadL1
    %c0_i64 = arith.constant 0 : i64
    %cneg1_i64 = arith.constant -1 : i64
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %a = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xbf16, #cbuf>
    %b = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xbf16, #cbuf>
    %c = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf32, #cc>
    hivm.hir.mmadL1 {
        already_set_real_mkn,
        b_transpose,
        fixpipe_for_result_already_inserted = true,
        normalized_in_L0C}
        ins(%a, %b, %true, %c64, %c64, %c64 :
            memref<4x4x16x16xbf16, #cbuf>,
            memref<4x4x16x16xbf16, #cbuf>, i1, index, index, index)
        outs(%c : memref<4x4x16x16xf32, #cc>)
        sync_related_args(%cneg1_i64, %cneg1_i64, %cneg1_i64,
                          %cneg1_i64, %cneg1_i64, %cneg1_i64,
                          %cneg1_i64 : i64, i64, i64, i64, i64, i64, i64)
    return
  }

  // CHECK-LABEL: func.func private @__pto_mmadl1_bf16_f32_tb
  // CHECK-SAME: pto.ptodsl.instantiation = "mmadl1_bf16_f32_tb"
  // CHECK: pto.mte_l1_l0b {{.*}} {transpose = false}
  // CHECK: pto.mad {{.*}} : !pto.ptr<bf16, l0a>, !pto.ptr<bf16, l0b>, !pto.ptr<f32, l0c>
  // CHECK: pto.mad_acc {{.*}} : !pto.ptr<bf16, l0a>, !pto.ptr<bf16, l0b>, !pto.ptr<f32, l0c>
}
```

Build the current converter if needed, then prove the test is red:

```bash
cmake --build "$ASCEND_NPU_IR_ROOT/build" --target bishengir-opt -j2
"$ASCEND_NPU_IR_ROOT/build/bin/llvm-lit" -v \
  "$ASCEND_NPU_IR_ROOT/bishengir/test/Conversion/HIVMTemplatesToPTO/mmadl1-specializations-ptodsl.mlir"
```

Expected before implementation: **FAIL** because the BF16-TB operation stays
unconverted or because `@__pto_mmadl1_bf16_f32_tb` is absent. Record the
failure in the review description.

### 1.2 Add the PTODSL specialization

Modify `nd2nz_mmadl1_64_ptodsl.py`:

1. Add this `SPECIALIZATIONS` entry:

   ```python
   "bf16_f32_tb": (pto.bf16, pto.f32, 2, 16, True, None),
   ```

2. Add this `EXPLICIT_INSTANTIATIONS` entry:

   ```python
   "mmadl1_bf16_f32_tb": (
       "mmadl1_bf16_f32_tb",
       "__pto_mmadl1_bf16_f32_tb",
       "bf16_f32_tb",
   ),
   ```

3. Add a typed wrapper with the same arguments as `mmadl1_bf16_f32`, but
   with a distinct Python name and `b_transpose=True`:

   ```python
   @pto.func(returns=None)
   def mmadl1_bf16_f32_tb(
       a_l1: pto.ptr(pto.bf16, "mat"),
       b_l1: pto.ptr(pto.bf16, "mat"),
       result_l0c: pto.ptr(pto.f32, "acc"),
       a_l0a: pto.ptr(pto.bf16, "left"),
       b_l0b: pto.ptr(pto.bf16, "right"),
       init: pto.i1,
       actual_m: pto.i64,
       actual_k: pto.i64,
       actual_n: pto.i64,
       wait_l1a_event: pto.i64,
       wait_l1b_event: pto.i64,
       release_l1a_event: pto.i64,
       release_l1b_event: pto.i64,
       l0_pingpong_event: pto.i64,
   ):
       _mmadl1_body(
           a_l1, b_l1, result_l0c, a_l0a, b_l0b, init,
           actual_m, actual_k, actual_n, wait_l1a_event,
           wait_l1b_event, release_l1a_event, release_l1b_event,
           l0_pingpong_event, b_transpose=True
       )
   ```

4. Add a `make_probe_kernel` branch for `bf16_f32_tb` that uses
   `nd2nz_bf16`, selects `mmadl1_bf16_f32_tb`, produces BF16 output, and uses
   `f32_bf16` pre-quantization. Copy the BF16-NN branch and change only the
   selected wrapper.

### 1.3 Generate and inspect the resource

Run the source-contract test:

```bash
python3 "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py" \
  --self-test
```

Expected: `PTODSL Cube source-contract checks passed`.

Generate the checked-in resource:

```bash
python3 "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py" \
  --emit-instantiation mmadl1_bf16_f32_tb \
  > "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/PTODSL/instantiations/mmadl1_bf16_f32_tb.mlir"
```

Then inspect the generated file:

```bash
rg -n "__pto_mmadl1_bf16_f32_tb|mte_l1_l0b|pto.mad|tf32_mode" \
  "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/PTODSL/instantiations/mmadl1_bf16_f32_tb.mlir"
```

Required observations:

- the symbol is `@__pto_mmadl1_bf16_f32_tb`;
- A/B pointers are BF16 and L0C is FP32;
- L0B staging has `transpose = false`, matching the existing F16-TB helper;
- both `pto.mad` and `pto.mad_acc` exist;
- `tf32_mode` is absent.

### 1.4 Register the resource and select it

Make these C++/CMake changes:

1. In `PTODSLTemplateImporter.h`, add:

   ```cpp
   inline constexpr llvm::StringLiteral kPTODSLMmadL1BF16F32TBSymbol =
       "__pto_mmadl1_bf16_f32_tb";
   ```

2. In `PTODSLTemplateImporter.cpp`:
   - increase `kTemplateResources` from 11 to 12;
   - register `mmadl1_bf16_f32_tb.mlir`, the new symbol constant, and
     instantiation `mmadl1_bf16_f32_tb`.

3. In `HIVMTemplatesToPTO/CMakeLists.txt`, add
   `mmadl1_bf16_f32_tb.mlir` to `PTOAS_PTODSL_CUBE_RESOURCES`.

4. In `MatmulRegionToPTO.cpp`, change the BF16/F32 branch so it accepts both
   values of `bTranspose` and selects TB or NN:

   ```cpp
   } else if (inputElementType.isBF16() &&
              accumulatorElementType.isF32() && !hf32) {
     inputShape = {4, 4, 16, 16};
     helperSymbol = bTranspose ? kPTODSLMmadL1BF16F32TBSymbol
                               : kPTODSLMmadL1BF16F32NNSymbol;
   ```

Do not weaken the earlier broad-contract rejection or change the buffer-shape
checks.

### 1.5 Build and prove the test is green

```bash
cmake --build "$ASCEND_NPU_IR_ROOT/build" --target bishengir-opt -j2
"$ASCEND_NPU_IR_ROOT/build/bin/llvm-lit" -v \
  "$ASCEND_NPU_IR_ROOT/bishengir/test/Conversion/HIVMTemplatesToPTO/mmadl1-specializations-ptodsl.mlir"
```

Expected: **PASS**.

Suggested commit after review:

```text
feat(bridge): add BF16 transposed-B MmadL1 template
```

---

## Task 2: FP32 inputs and IEEE FP32 accumulation, NN

**Deliverable:** FP32/F32 MmadL1 without `enable_HF32` and without logical
transpose converts to `@__pto_mmadl1_f32_f32_ieee_nn`.

**Closest reference:** Use `mmadl1_f32_f32_hf32` for all pointer types and
shapes, but call `_mmadl1_body` without `tf32_mode`. IEEE is represented by
the absence of TF32/HF32 mode, not by a new precision attribute.

### 2.1 Append the failing conversion test

Append this function inside the existing test module, before the private
helper checks:

```mlir
  func.func @mmadl1_f32_f32_ieee_nn() {
    // CHECK-LABEL: func.func @mmadl1_f32_f32_ieee_nn
    // CHECK: call @__pto_mmadl1_f32_f32_ieee_nn
    // CHECK-NOT: hivm.hir.mmadL1
    %c0_i64 = arith.constant 0 : i64
    %cneg1_i64 = arith.constant -1 : i64
    %c64 = arith.constant 64 : index
    %true = arith.constant true
    %a = hivm.hir.pointer_cast(%c0_i64) : memref<8x4x16x8xf32, #cbuf>
    %b = hivm.hir.pointer_cast(%c0_i64) : memref<8x4x16x8xf32, #cbuf>
    %c = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf32, #cc>
    hivm.hir.mmadL1 {
        already_set_real_mkn,
        fixpipe_for_result_already_inserted = true,
        normalized_in_L0C}
        ins(%a, %b, %true, %c64, %c64, %c64 :
            memref<8x4x16x8xf32, #cbuf>,
            memref<8x4x16x8xf32, #cbuf>, i1, index, index, index)
        outs(%c : memref<4x4x16x16xf32, #cc>)
        sync_related_args(%cneg1_i64, %cneg1_i64, %cneg1_i64,
                          %cneg1_i64, %cneg1_i64, %cneg1_i64,
                          %cneg1_i64 : i64, i64, i64, i64, i64, i64, i64)
    return
  }
```

Append these checks near the other private-helper checks:

```mlir
  // CHECK-LABEL: func.func private @__pto_mmadl1_f32_f32_ieee_nn
  // CHECK-SAME: pto.ptodsl.instantiation = "mmadl1_f32_f32_ieee_nn"
  // CHECK: pto.mte_l1_l0b {{.*}} {transpose = true}
  // CHECK: pto.mad {{.*}} : !pto.ptr<f32, l0a>, !pto.ptr<f32, l0b>, !pto.ptr<f32, l0c>
  // CHECK: pto.mad_acc {{.*}} : !pto.ptr<f32, l0a>, !pto.ptr<f32, l0b>, !pto.ptr<f32, l0c>
```

Run the targeted lit test and record the expected **FAIL** because the IEEE
specialization is absent:

```bash
"$ASCEND_NPU_IR_ROOT/build/bin/llvm-lit" -v \
  "$ASCEND_NPU_IR_ROOT/bishengir/test/Conversion/HIVMTemplatesToPTO/mmadl1-specializations-ptodsl.mlir"
```

### 2.2 Add the PTODSL specialization

Modify `nd2nz_mmadl1_64_ptodsl.py`:

1. Add:

   ```python
   "f32_f32_ieee_nn": (pto.f32, pto.f32, 4, 8, False, None),
   ```

2. Add:

   ```python
   "mmadl1_f32_f32_ieee_nn": (
       "mmadl1_f32_f32_ieee",
       "__pto_mmadl1_f32_f32_ieee_nn",
       "f32_f32_ieee_nn",
   ),
   ```

3. Add `mmadl1_f32_f32_ieee` with the same signature as
   `mmadl1_f32_f32_hf32`, but call `_mmadl1_body` as follows:

   ```python
   _mmadl1_body(
       a_l1, b_l1, result_l0c, a_l0a, b_l0b, init,
       actual_m, actual_k, actual_n, wait_l1a_event,
       wait_l1b_event, release_l1a_event, release_l1b_event,
       l0_pingpong_event, b_transpose=False
   )
   ```

   Do not pass `tf32_mode`.

4. Add a `make_probe_kernel` branch using `nd2nz_f32`, the IEEE wrapper,
   FP32 output, and no pre-quantization.

### 2.3 Generate and inspect the resource

```bash
python3 "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py" \
  --self-test
python3 "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py" \
  --emit-instantiation mmadl1_f32_f32_ieee_nn \
  > "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/PTODSL/instantiations/mmadl1_f32_f32_ieee_nn.mlir"
```

Inspect the result:

```bash
rg -n "__pto_mmadl1_f32_f32_ieee_nn|pto.mad|tf32_mode" \
  "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/PTODSL/instantiations/mmadl1_f32_f32_ieee_nn.mlir"
! rg -q "tf32_mode" \
  "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/PTODSL/instantiations/mmadl1_f32_f32_ieee_nn.mlir"
```

The second command must succeed, proving `tf32_mode` is absent. Also confirm
that both `pto.mad` and `pto.mad_acc` use FP32 A, B, and L0C pointers.

### 2.4 Register and select the IEEE resource

1. Add `kPTODSLMmadL1F32F32IEEENNSymbol` to
   `PTODSLTemplateImporter.h` with value
   `__pto_mmadl1_f32_f32_ieee_nn`.
2. Increase `kTemplateResources` from 12 to 13 and add the resource tuple in
   `PTODSLTemplateImporter.cpp`.
3. Add the MLIR resource to `PTOAS_PTODSL_CUBE_RESOURCES` in CMake.
4. Replace the existing F32/F32 HF32-only matcher branch with:

   ```cpp
   } else if (inputElementType.isF32() &&
              accumulatorElementType.isF32() && !bTranspose) {
     inputShape = {8, 4, 16, 8};
     helperSymbol = hf32 ? kPTODSLMmadL1F32F32HF32NNSymbol
                         : kPTODSLMmadL1F32F32IEEENNSymbol;
   ```

This must preserve HF32 selection when `enable_HF32` is present.

### 2.5 Build and prove both tests are green

```bash
cmake --build "$ASCEND_NPU_IR_ROOT/build" --target bishengir-opt -j2
"$ASCEND_NPU_IR_ROOT/build/bin/llvm-lit" -v \
  "$ASCEND_NPU_IR_ROOT/bishengir/test/Conversion/HIVMTemplatesToPTO/mmadl1-specializations-ptodsl.mlir"
```

Expected: **PASS**, with both calls and both imported resource bodies checked.

Suggested commit after review:

```text
feat(bridge): add FP32 IEEE MmadL1 template
```

---

## Final validation

### 1. Run focused and full BishengIR tests

```bash
"$ASCEND_NPU_IR_ROOT/build/bin/llvm-lit" -v \
  "$ASCEND_NPU_IR_ROOT/bishengir/test/Conversion/HIVMTemplatesToPTO"
cmake --build "$ASCEND_NPU_IR_ROOT/build" --target check-bishengir -j2
```

Both commands must complete with zero unexpected failures.

### 2. Confirm VMI/VPTO lowering with the generator probes

Emit one standalone probe for each new specialization:

```bash
python3 "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py" \
  --emit-mlir --specialization bf16_f32_tb > /tmp/mmadl1_bf16_f32_tb.mlir
python3 "$ASCEND_NPU_IR_ROOT/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py" \
  --emit-mlir --specialization f32_f32_ieee_nn > /tmp/mmadl1_f32_f32_ieee_nn.mlir
```

Inspect the PTO contracts:

```bash
rg -n "mte_l1_l0b|pto.mad|pto.mad_acc|tf32_mode" \
  /tmp/mmadl1_bf16_f32_tb.mlir /tmp/mmadl1_f32_f32_ieee_nn.mlir
! rg -q "tf32_mode" /tmp/mmadl1_f32_f32_ieee_nn.mlir
```

Lower both probes from VMI to VPTO:

```bash
"$PTOAS_BIN" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  --emit-vpto /tmp/mmadl1_bf16_f32_tb.mlir \
  -o /tmp/mmadl1_bf16_f32_tb.vpto.mlir
"$PTOAS_BIN" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  --emit-vpto /tmp/mmadl1_f32_f32_ieee_nn.mlir \
  -o /tmp/mmadl1_f32_f32_ieee_nn.vpto.mlir
test -s /tmp/mmadl1_bf16_f32_tb.vpto.mlir
test -s /tmp/mmadl1_f32_f32_ieee_nn.vpto.mlir
```

Both PTOAS commands must exit zero and produce non-empty VPTO files. This is
required because the generator self-test proves the PTODSL source contract,
not the complete PTOAS backend path.

### 3. Run the affected Q2 regressions

Activate the CANN/Python environment documented in
[`../hardware-benchmark-runner.md`](../hardware-benchmark-runner.md), then:

```bash
cmake --build "$ASCEND_NPU_IR_ROOT/build" \
  --target bishengir-compile bishengir-opt -j2
export PATH="$ASCEND_NPU_IR_ROOT/build/bin:$PATH"
cd "$PLANNER_ROOT"
python3 bridge/tools/run_q2_q3_hardware_tests.py \
  --repo q2 --q2-root "$Q2_ROOT" \
  --pattern '*test_chunk_scaled_dot_kkt_fwd_perf*' \
  --jobs 1
```

Acceptance for BF16-TB:

- all 10 shapes are executed;
- no log contains `type, transpose, and precision have no pre-generated specialization`;
- no log refers to missing `__pto_mmadl1_bf16_f32_tb`;
- because this was the only recorded blocker, investigate any remaining
  failure rather than declaring it expected.

Then run the solve-tril families:

```bash
python3 bridge/tools/run_q2_q3_hardware_tests.py \
  --repo q2 --q2-root "$Q2_ROOT" \
  --pattern '*test_solve_tril_mxr_perf.py*' \
  --pattern '*test_solve_tril_perf.py*' \
  --jobs 1
```

Acceptance for FP32-IEEE:

- all 20 solve-tril shapes are executed;
- no log reports the missing FP32 IEEE MmadL1 specialization;
- a remaining ND2NZ or Fixpipe failure is allowed and must be reported as a
  later known blocker, not as failure of IEEE helper selection.

### 4. Review the diff

```bash
git -C "$ASCEND_NPU_IR_ROOT" diff --check
git -C "$ASCEND_NPU_IR_ROOT" status --short
git -C "$ASCEND_NPU_IR_ROOT" diff -- \
  bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py \
  bishengir/lib/Template/lib/RegBase/Cube/PTODSL/instantiations \
  bishengir/lib/Conversion/HIVMTemplatesToPTO \
  bishengir/test/Conversion/HIVMTemplatesToPTO/mmadl1-specializations-ptodsl.mlir
```

The final change must contain exactly two new typed templates and their tests.
It must not contain unrelated generated artifacts, bridge outputs, Q2 source
changes, or formatting-only rewrites.

## Definition of done

- Both new explicit MLIR resources are generated and checked in.
- The importer validates and installs both resources.
- The matcher selects BF16-TB only for BF16/F32 with `b_transpose`.
- The matcher selects FP32-IEEE only for FP32/F32 without `enable_HF32` and
  without B transpose.
- Existing BF16-NN, F16-TB, and FP32-HF32 behavior remains unchanged.
- FP32-IEEE generated IR contains no `tf32_mode`.
- Targeted lit tests and `check-bishengir` pass.
- Q2 logs no longer contain either targeted specialization error.
- Remaining solve-tril ND2NZ/Fixpipe failures are recorded separately.
