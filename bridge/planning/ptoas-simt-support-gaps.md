# AscendNPU-IR SIMT support gaps in PTOAS

Audit date: 2026-10-15. This is a local source audit for `Planner/bridge`, not an end-to-end hardware certification.

## Result

**PTOAS has substantial SIMT support, but it does not cover the full AscendNPU-IR SIMT surface.** The concrete gaps are scheduling yield, some program-info semantics, full shuffle controls, volatile loads, atomic type/result cases, a broad math-library surface, and explicit arithmetic rounding modes.

This report checks all **223 concrete operations in AscendDPXOps.td**, the SIMT section of HIVMRegbaseIntrins, and the HIVM SIMT-facing memory operations. Template-family coverage is outside this report’s scope. The complete DPX matrix below prevents a same-name search from overlooking renamed equivalents. “Missing” means no corresponding dedicated scalar SIMT PTO operation was found; it does **not** prove the computation cannot be expressed using LLVM, standard MLIR, or software.

## Source baseline and evidence

| Checkout | HEAD |
| --- | --- |
| `AscendNPU-IR` | `0a8c614bc531bc4ebe1cbcdfb946f8f255768e79` |
| `AscendNPU-IR/third-party/ptoas` | `3c1c98fcc8ec0c49b8bededc9a0c7599c6a71a02` |

The comparison baseline is **AscendNPU-IR's pinned `third-party/ptoas` submodule**, not a standalone PTOAS checkout. At AscendNPU-IR revision `0a8c614bc531bc4ebe1cbcdfb946f8f255768e79`, the recorded gitlink and checked-out submodule HEAD both equal `3c1c98fcc8ec0c49b8bededc9a0c7599c6a71a02`. Baseline source and test links below point to that submodule.

Primary sources (paths are relative to this report):

- [DPX operations](../../../AscendNPU-IR/bishengir/include/bishengir/Dialect/AscendDPX/IR/AscendDPXOps.td), especially program information, memory, shuffle, atomic, cast, and custom math sections.
- [DPX lowering](../../../AscendNPU-IR/bishengir/lib/Conversion/AscendDPXToHIVMRegbaseIntrins/AscendDPXToHIVMRegbaseIntrins.cpp), including `AscendDPXYieldOpLowering`, `AscendDPXAtomicOpLowering`, shuffle packing, volatile loads, and complex barriers.
- [DPX math lowering](../../../AscendNPU-IR/bishengir/lib/Conversion/AscendDPXToHIVMRegbaseIntrins/AscendDPXMathOpsLowering.cpp): direct conversions and `_mlir_ciface_simt_*` library calls.
- [HIVM intrinsic operations](../../../AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.td), SIMT section starting with `LaunchFuncOp`.
- [PTO SIMT definitions](../../../AscendNPU-IR/third-party/ptoas/include/PTO/IR/VPTOOps.td), [PTO system/section definitions](../../../AscendNPU-IR/third-party/ptoas/include/PTO/IR/PTOOps.td), and [PTO attributes](../../../AscendNPU-IR/third-party/ptoas/include/PTO/IR/PTOAttrs.td).
- [PTO verification](../../../AscendNPU-IR/third-party/ptoas/lib/PTO/IR/VPTO.cpp), particularly `verifyAtomicCommon`, shuffle and conversion verifiers.
- [PTO LLVM emission](../../../AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/VPTOLLVMEmitter.cpp), particularly `buildShuffleControlValue`, `buildAtomicCalleeName`, runtime queries, and scalar load/math conversion patterns.
- [SIMT math helper declarations](../../../AscendNPU-IR/bishengir/lib/Template/include/RegBase/SimtUtils.h), used as evidence for instruction type coverage.

## Missing or incomplete PTOAS support to track

This table groups instruction gaps by category and explains their impact and possible resolutions. The complete AscendDPX matrix later in this report lists every DPX operation individually, including supported operations and candidate decompositions. Several matrix rows can therefore correspond to one summary row; the summary also includes relevant HIVM-level restrictions.

| Gap | NPU-IR evidence | PTOAS finding | Bridge consequence / possible resolution |
| --- | --- | --- | --- |
| SIMT scheduling yield | DPX `yield` lowers to HIVM `yield` intrinsic; used in complex-barrier spin loops | No corresponding SIMT yield op. `pto.yield` is a fusion terminator | Add a separate scheduling op and emitter, or preserve the LLVM intrinsic explicitly. |
| Physical core ID | DPX `core_id` lowers to `tpe_get_COREID` | PTO has `get_veccoreid`, emitting `llvm.hivm.tpe.get.VECCOREID`; no exact COREID equivalent found | Establish semantic equivalence before mapping; otherwise expose COREID. |
| Arbitrary shuffle control | DPX `shfl.*` accepts independent SSA `lane_mask`, `clamp`, `index`; HIVM forms accept packed `ctrl` | PTO `shuffle_*` has one SSA lane/offset and a static `width` restricted to 16 or 32. Emitter constructs the remaining control bits | Common width-based shuffle is supported. Arbitrary packed or dynamic mask/clamp needs a lower-level form or equivalent decomposition. |
| Volatile load semantics | DPX load has `volatileOption`; HIVM indirect load defaults `isVolatile=true` | PTO load/ldg definitions expose no volatile control; `ConvertPtoLoadOp` emits an ordinary LLVM load | Preserve volatility through a supported representation/emitter. Cache hints alone do not encode volatile semantics. |
| Atomic INC/DEC | DPX atomic_inc/atomic_dec have lowering and intrinsic names | No dedicated PTO atomic_inc/atomic_dec | Potentially compose add/sub by one or CAS. Check native INC/DEC contract first; do not assume semantics from the name. |
| i8/i16 integer atomics | DPX lowering contains aligned i32 CAS loops for subword operations | PTO atomic value constraint excludes i8/i16 | Port the subword CAS algorithm; direct new hardware atomics are not required. Preserve neighboring bits, alignment, and returned old value. |
| Packed atomic old values | DPX atomics expose a returned old value | PTO verifier forbids consuming the result for vector<2xf16>/vector<2xbf16> atomics on beta.1 | Return-producing packed rows need a proven fallback or a target-specific implementation. An unused packed result is supported. |
| i64 UB atomics | DPX atomic lowering selects GM (`G`) or shared (`S`) intrinsic names from pointer address space | PTO rejects i64 VEC/UB atomics | Shared-memory i64 atomic cases need a reviewed fallback or target-specific support. GM i64 atomics already exist. |
| Scalar math type coverage | NPU-IR helper declarations include bf16 erf/rsqrt/tanh/sin/cos/pow and integer pow | PTO sin/cos are f32 only; exp/log/sqrt exclude scalar bf16; absf excludes scalar bf16; pow excludes scalar bf16 and integers | Add type-specific helpers/conversions with validated precision. Existing packed bf16 or vector ops do not establish scalar coverage. |
| Missing SIMT math-library families | DPX math declarations and `LibCallLowering` include erf family, tan/tanh, reciprocal/rsqrt, log1p/expm1, ldexp/ilogb, gamma/Bessel, norm and inverse-CDF families | No corresponding dedicated scalar SIMT PTO ops found; full per-op list is below | Port helpers or define standard-MLIR decompositions with the same numerical contract. Existing `pto.v*`/tile operations execute at different granularity. |
| Explicit arithmetic rounding | DPX declares add/sub/mul/div/fma/sqrt/rsqrt/rcp with rn/rz/rd/ru variants | PTO has conversion rounding, but scalar arithmetic ops do not expose these rounding selectors | Explicit directed rounding requires suitable LLVM intrinsics/constrained arithmetic or helpers. Ordinary default arithmetic is not a general substitute. |
| Fast-math variants | DPX declares fast_sinf, fast_cosf, fast_expf, fast_log*, fast_powf, fast_dividef, fast_tanf, fast_tanhf | No dedicated variants specifying those contracts | Validate any use of ordinary PTO math against the source approximation/accuracy contract. |
| Scalar call ABI | DPX `call_scalar` carries `use_shmem_offset` and argument/result attributes | Generic calls exist, but no matching dedicated boundary op | Preserve scalar/SIMT calling convention and shared-memory argument ABI during bridge lowering. A generic call alone does not prove ABI compatibility. |

## Already present: avoid filing these as absent instructions

- SIMT execution boundaries: `pto.section.simt`, `pto.simt_launch`, `pto.store_vfsimt_info`, and outlined `pto.simt_entry` functions.
- Thread IDs, block dimensions, grid dimensions, XYZ block indices, clocks; linear `pto.get_block_idx` and `pto.get_block_num` use SIMT-specific query lowering with an i64 PTO result.
- UB/local scalar loads/stores and GM loads/stores with cache controls; add control flow for masked memory accesses so disabled lanes never touch an invalid address.
- Nine atomic families: CAS, exchange, add, sub, min, max, and, or, xor. Unsigned min/max use signedness attributes rather than separate operation names.
- Four shuffle families (subject to controls above); add/min/max reductions, including unsigned integer selectors; votes and ballots.
- `pto.prmt`, `pto.mulhi`, `pto.mul_i32toi64`, floating min/max, abs, exp/log/sqrt, ceil/floor/rint/round, fma, and pow within their declared type slices.
- `pto.syncthreads`, `pto.threadfence`, `pto.threadfence_block`, keep/resume. NPU-IR complex barrier algorithms still need translation; a matching primitive does not establish full algorithm support.

PTO sin/cos use f32 PTODSL SoftOps expansion (`PTOExpandSoftLib`), so their presence is not evidence that every NPU-IR SIMT library helper is linked or available.

## Recommended approach (ranked by cleanness)

1. Reuse existing PTO SIMT primitives and standard arith/scf/LLVM operations for indexing, masks, subword CAS, bit manipulation, and ordinary casts. Keep these as bridge lowering work.
2. Extend PTO only where source semantics cannot be carried through its present interface: scheduling yield, volatile controls, raw shuffle control, required runtime queries, and target-supported rounding/result forms.
3. Port or link NPU-IR math helpers for algorithms without a direct primitive. Preserve their calling convention, address spaces and accuracy; a symbol declaration is not an implementation.

Before calling a row supported end to end, compile representative rank/type/mask cases through the actual bridge and selected PTOAS build, inspect emitted LLVM, and validate runtime results. This audit did not run a compiler rebuild, simulator, or hardware test.

## Complete AscendDPX correspondence matrix

Statuses describe source-level correspondence, not measured execution support:

- **Exists**: a corresponding dedicated PTO op exists for the stated slice.
- **Partial**: an equivalent family exists, with type, control, ABI, or algorithm limitations.
- **Compose**: standard MLIR/LLVM operations can express the operation; no dedicated PTO SIMT mnemonic is needed. These are candidate lowering routes, not claims that bridge patterns already exist.
- **Missing**: no dedicated scalar SIMT PTO counterpart was found; an extension/helper or reviewed decomposition is needed.

Counts: **Exists: 29**, **Partial: 63**, **Compose: 27**, **Missing: 104**. Total: **223**. Counts are per declared DPX operation, not per supported type instantiation.

| AscendDPX operation | Status | PTO correspondence / gap |
| --- | --- | --- |
| `thread_id_x` (DPX line 40) | Exists | `pto.get_tid_x` |
| `thread_id_y` (DPX line 41) | Exists | `pto.get_tid_y` |
| `thread_id_z` (DPX line 42) | Exists | `pto.get_tid_z` |
| `block_idx_x` (DPX line 44) | Exists | `pto.get_block_idx_x` |
| `block_idx_y` (DPX line 45) | Exists | `pto.get_block_idx_y` |
| `block_idx_z` (DPX line 46) | Exists | `pto.get_block_idx_z` |
| `block_idx` (DPX line 47) | Exists | `pto.get_block_idx` plus i64-to-i32 cast in SIMT scope. |
| `block_dim_x` (DPX line 49) | Exists | `pto.get_block_dim_x` |
| `block_dim_y` (DPX line 50) | Exists | `pto.get_block_dim_y` |
| `block_dim_z` (DPX line 51) | Exists | `pto.get_block_dim_z` |
| `grid_dim_x` (DPX line 53) | Exists | `pto.get_grid_dim_x` |
| `grid_dim_y` (DPX line 54) | Exists | `pto.get_grid_dim_y` |
| `grid_dim_z` (DPX line 55) | Exists | `pto.get_grid_dim_z` |
| `clock32` (DPX line 57) | Exists | `pto.get_clock32` |
| `clock64` (DPX line 58) | Exists | `pto.get_clock64` |
| `core_id` (DPX line 62) | Missing | DPX emits `tpe_get_COREID`; PTO exposes `get_veccoreid` / `tpe.get.VECCOREID`, whose equivalence is unproven. |
| `load` (DPX line 67) | Partial | `pto.load`/`pto.store` (UB/local), `pto.ldg`/`pto.stg` (GM); masks need control flow; volatile load is missing. |
| `store` (DPX line 96) | Partial | `pto.store` (UB/local), `pto.stg` (GM); masks require control flow and cache-policy translation. |
| `sync_threads` (DPX line 121) | Partial | `pto.syncthreads`, `pto.threadfence_block`; bridge must preserve NPU-IR barrier selection and complex-barrier lowering. |
| `yield` (DPX line 125) | Missing | No SIMT scheduling yield; `pto.yield` terminates a fusion region and is unrelated. |
| `shfl.up` (DPX line 139) | Partial | `pto.shuffle_up`; static width 16 or 32 only; no independent dynamic lane_mask/clamp. |
| `shfl.down` (DPX line 140) | Partial | `pto.shuffle_down`; static width 16 or 32 only; no independent dynamic lane_mask/clamp. |
| `shfl.bfly` (DPX line 141) | Partial | `pto.shuffle_bfly`; static width 16 or 32 only; no independent dynamic lane_mask/clamp. |
| `shfl.idx` (DPX line 142) | Partial | `pto.shuffle_idx`; static width 16 or 32 only; no independent dynamic lane_mask/clamp. |
| `permute` (DPX line 144) | Exists | `pto.prmt` for i32 operands. |
| `reduce.add` (DPX line 161) | Exists | `pto.redux_add`; i32/f16/f32 supported. |
| `reduce.max` (DPX line 162) | Exists | `pto.redux_max`; i32/f16/f32 supported. |
| `reduce.min` (DPX line 163) | Exists | `pto.redux_min`; i32/f16/f32 supported. |
| `reduce.umax` (DPX line 164) | Exists | `pto.redux_max` with unsigned signedness. |
| `reduce.umin` (DPX line 165) | Exists | `pto.redux_min` with unsigned signedness. |
| `atomic_and` (DPX line 182) | Partial | `pto.atomic_and`; see atomic type/result restrictions in the gap table above. |
| `atomic_or` (DPX line 183) | Partial | `pto.atomic_or`; see atomic type/result restrictions in the gap table above. |
| `atomic_xor` (DPX line 184) | Partial | `pto.atomic_xor`; see atomic type/result restrictions in the gap table above. |
| `atomic_inc` (DPX line 185) | Missing | No dedicated op; map to add/sub by one only after checking INC/DEC semantics, otherwise use CAS loop. |
| `atomic_dec` (DPX line 186) | Missing | No dedicated op; map to add/sub by one only after checking INC/DEC semantics, otherwise use CAS loop. |
| `atomic_max` (DPX line 187) | Partial | `pto.atomic_max`; see atomic type/result restrictions in the gap table above. |
| `atomic_min` (DPX line 188) | Partial | `pto.atomic_min`; see atomic type/result restrictions in the gap table above. |
| `atomic_add` (DPX line 189) | Partial | `pto.atomic_add`; see atomic type/result restrictions in the gap table above. |
| `atomic_sub` (DPX line 190) | Partial | `pto.atomic_sub`; see atomic type/result restrictions in the gap table above. |
| `atomic_umax` (DPX line 191) | Partial | `pto.atomic_max`; see atomic type/result restrictions in the gap table above. Set unsigned signedness. |
| `atomic_umin` (DPX line 192) | Partial | `pto.atomic_min`; see atomic type/result restrictions in the gap table above. Set unsigned signedness. |
| `atomic_exchange` (DPX line 193) | Partial | `pto.atomic_exch`; see atomic type/result restrictions in the gap table above. |
| `atomic_cas` (DPX line 194) | Partial | `pto.atomic_cas`; see atomic type/result restrictions in the gap table above. |
| `cast` (DPX line 206) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `abs` (DPX line 234) | Partial | `pto.absf` lacks scalar bf16 and integer forms. |
| `acos` (DPX line 235) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `acosh` (DPX line 236) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `asin` (DPX line 237) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `asinh` (DPX line 238) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `atan` (DPX line 239) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `atanh` (DPX line 240) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `brev` (DPX line 241) | Compose | LLVM bitreverse intrinsic. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `cbrt` (DPX line 242) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `ceil` (DPX line 243) | Exists | `pto.ceil`; preserve source type and numerical contract. |
| `clz` (DPX line 244) | Compose | LLVM ctlz intrinsic; define behavior for zero. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `cos` (DPX line 245) | Partial | `pto.cos` is f32 only and uses SoftOps expansion. |
| `cosh` (DPX line 246) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `cospi` (DPX line 247) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `cyl_bessel_i0` (DPX line 248) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `cyl_bessel_i1` (DPX line 249) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `erf` (DPX line 250) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `erfc` (DPX line 251) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `erfcinv` (DPX line 252) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `erfcx` (DPX line 253) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `erfinv` (DPX line 254) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `exp10` (DPX line 255) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `exp2` (DPX line 256) | Compose | `math.exp2` / LLVM exp2. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `exp` (DPX line 257) | Partial | `pto.exp` lacks scalar bf16. |
| `expm1` (DPX line 258) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `fast_cosf` (DPX line 259) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `fast_exp10f` (DPX line 260) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `fast_expf` (DPX line 261) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `fast_log10f` (DPX line 262) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `fast_log2f` (DPX line 263) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `fast_logf` (DPX line 264) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `fast_sinf` (DPX line 265) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `fast_tanf` (DPX line 266) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `fast_tanhf` (DPX line 267) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `float2half_rn` (DPX line 268) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2int_rn` (DPX line 269) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2int_rz` (DPX line 270) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2int_rd` (DPX line 271) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2int_ru` (DPX line 272) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2ll_rn` (DPX line 273) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2ll_rz` (DPX line 274) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2ll_rd` (DPX line 275) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2ll_ru` (DPX line 276) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2uint_rn` (DPX line 277) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2uint_rz` (DPX line 278) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2uint_rd` (DPX line 279) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2uint_ru` (DPX line 280) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2ull_rn` (DPX line 281) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2ull_rz` (DPX line 282) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2ull_rd` (DPX line 283) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float2ull_ru` (DPX line 284) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `float_as_int` (DPX line 285) | Compose | `arith.bitcast` (same-width bit reinterpretation). |
| `float_as_uint` (DPX line 286) | Compose | `arith.bitcast` (same-width bit reinterpretation). |
| `floor` (DPX line 287) | Exists | `pto.floor`; preserve source type and numerical contract. |
| `ffs` (DPX line 288) | Compose | cttz plus one, with explicit zero handling. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `gamma` (DPX line 289) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `half2float` (DPX line 290) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `ilogb` (DPX line 291) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `int2float_rn` (DPX line 292) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `int2float_rz` (DPX line 293) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `int2float_rd` (DPX line 294) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `int2float_ru` (DPX line 295) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `int_as_float` (DPX line 296) | Compose | `arith.bitcast` (same-width bit reinterpretation). |
| `j0` (DPX line 297) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `j1` (DPX line 298) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `ll2float_rn` (DPX line 299) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `ll2float_rz` (DPX line 300) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `ll2float_rd` (DPX line 301) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `ll2float_ru` (DPX line 302) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `llrint` (DPX line 303) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `llround` (DPX line 304) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `log10` (DPX line 305) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `log1p` (DPX line 306) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `log2` (DPX line 307) | Compose | `math.log2` / LLVM log2. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `log` (DPX line 308) | Partial | `pto.log` lacks scalar bf16. |
| `logb` (DPX line 309) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `lgamma` (DPX line 310) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `nearbyint` (DPX line 311) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `nanf` (DPX line 312) | Compose | NaN constant construction; preserve payload rules. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `normcdf` (DPX line 313) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `normcdfinv` (DPX line 314) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `popc` (DPX line 315) | Compose | LLVM ctpop intrinsic. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `rcp_rd` (DPX line 316) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `rcp_rn` (DPX line 317) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `rcp_ru` (DPX line 318) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `rcp_rz` (DPX line 319) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `rcbrt` (DPX line 320) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `recip` (DPX line 321) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `relu` (DPX line 322) | Compose | Comparison/select or float max with source NaN semantics. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `rint` (DPX line 323) | Exists | `pto.rint`; preserve source type and numerical contract. |
| `round` (DPX line 324) | Exists | `pto.round`; preserve source type and numerical contract. |
| `rsqrt` (DPX line 325) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `rsqrt_rd` (DPX line 326) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `rsqrt_rn` (DPX line 327) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `rsqrt_ru` (DPX line 328) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `rsqrt_rz` (DPX line 329) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `saturatef` (DPX line 330) | Compose | Clamp/select sequence with explicit NaN behavior. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `signbit` (DPX line 331) | Compose | Bitcast and extract sign bit (handles negative zero). No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `sin` (DPX line 332) | Partial | `pto.sin` is f32 only and uses SoftOps expansion. |
| `sinh` (DPX line 333) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `sinpi` (DPX line 334) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `sqrt` (DPX line 335) | Partial | `pto.sqrt` lacks scalar bf16. |
| `sqrt_rd` (DPX line 336) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `sqrt_rn` (DPX line 337) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `sqrt_ru` (DPX line 338) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `sqrt_rz` (DPX line 339) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `tan` (DPX line 340) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `tanh` (DPX line 341) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `tgamma` (DPX line 342) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `trunc` (DPX line 343) | Compose | `math.trunc` / LLVM trunc. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `uint2float_rn` (DPX line 344) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `uint2float_rz` (DPX line 345) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `uint2float_rd` (DPX line 346) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `uint2float_ru` (DPX line 347) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `uint_as_float` (DPX line 348) | Compose | `arith.bitcast` (same-width bit reinterpretation). |
| `ull2float_rn` (DPX line 349) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `ull2float_rz` (DPX line 350) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `ull2float_rd` (DPX line 351) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `ull2float_ru` (DPX line 352) | Partial | `pto.convert` for supported type pairs, rounding and signedness; generic `arith` casts for other legal cases. Not a missing mnemonic family. |
| `y0` (DPX line 353) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `y1` (DPX line 354) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `reciprocal` (DPX line 355) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `div` (DPX line 364) | Compose | `arith.divsi`/`arith.divf`/`arith.divui` or corresponding LLVM ops; preserve source types. |
| `fdiv` (DPX line 365) | Compose | `arith.divsi`/`arith.divf`/`arith.divui` or corresponding LLVM ops; preserve source types. |
| `udiv` (DPX line 366) | Compose | `arith.divsi`/`arith.divf`/`arith.divui` or corresponding LLVM ops; preserve source types. |
| `pow` (DPX line 367) | Partial | `pto.pow` supports f16/f32 and packed f16; lacks scalar bf16 and integer forms. |
| `umulhi` (DPX line 368) | Exists | `pto.mulhi` with signed/unsigned signedness (i32/i64). |
| `ldexp` (DPX line 369) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `copysign` (DPX line 370) | Compose | `math.copysign` / LLVM copysign. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `atan2` (DPX line 371) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `nextafter` (DPX line 372) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `hypot` (DPX line 373) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `max` (DPX line 374) | Partial | `pto.fmax` for floats; `arith` signed/unsigned min/max for integers; verify NaN and signed-zero rules. |
| `min` (DPX line 375) | Partial | `pto.fmin` for floats; `arith` signed/unsigned min/max for integers; verify NaN and signed-zero rules. |
| `mulhi` (DPX line 376) | Exists | `pto.mulhi` with signed/unsigned signedness (i32/i64). |
| `mul24` (DPX line 377) | Compose | Extend/truncate 24-bit operands then multiply. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `hadd` (DPX line 378) | Compose | Overflow-safe half-add sequence. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `rhadd` (DPX line 379) | Compose | Overflow-safe rounded half-add sequence. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `scalbn` (DPX line 380) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `fdim` (DPX line 381) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `fast_dividef` (DPX line 382) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `div_rn` (DPX line 383) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `div_rz` (DPX line 384) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `div_rd` (DPX line 385) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `div_ru` (DPX line 386) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `fmod` (DPX line 387) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `remainder` (DPX line 388) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `add_rn` (DPX line 389) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `add_rz` (DPX line 390) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `add_rd` (DPX line 391) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `add_ru` (DPX line 392) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `mul_rn` (DPX line 393) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `mul_rz` (DPX line 394) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `mul_rd` (DPX line 395) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `mul_ru` (DPX line 396) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `fast_powf` (DPX line 397) | Missing | No dedicated fast-math variant/contract. Some bases have PTO ops; validate approximation and accuracy before substitution. |
| `jn` (DPX line 398) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `rhypot` (DPX line 399) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `sub_rd` (DPX line 400) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `sub_rn` (DPX line 401) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `sub_ru` (DPX line 402) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `sub_rz` (DPX line 403) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `yn` (DPX line 404) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `byte_perm` (DPX line 413) | Exists | `pto.prmt` for i32 operands. |
| `fma_rn` (DPX line 414) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `fma_rz` (DPX line 415) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `fma_rd` (DPX line 416) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `fma_ru` (DPX line 417) | Missing | No explicit arithmetic rounding selector for this family; `pto.convert` rounding does not cover arithmetic. RN may use default arithmetic only after contract validation. |
| `fma` (DPX line 418) | Exists | `pto.fma`; preserve source type and numerical contract. |
| `norm3d` (DPX line 419) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `rnorm3d` (DPX line 420) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `sad` (DPX line 421) | Compose | Absolute difference plus add with correct signedness/width. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `norm4d` (DPX line 430) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `rnorm4d` (DPX line 431) | Missing | No dedicated scalar SIMT PTO op; requires a software helper or an explicit decomposition. |
| `finitef` (DPX line 440) | Compose | Exponent-bit check or compare/abs sequence. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `isfinite` (DPX line 441) | Compose | Exponent-bit check or compare/abs sequence. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `isinf` (DPX line 442) | Compose | Exponent/mantissa-bit check or compare sequence. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `isnan` (DPX line 443) | Compose | Unordered floating comparison. No dedicated PTO SIMT mnemonic; bridge lowering is required. |
| `call_scalar` (DPX line 449) | Partial | Ordinary function calls exist; scalar/SIMT calling-convention and shared-memory-offset ABI are not an equivalent dedicated PTO op. |

## Existing test evidence inspected

These are source fixtures inspected during the audit, not test executions:

- [Packed atomic old-value rejection](../../../AscendNPU-IR/third-party/ptoas/test/lit/vpto/simt_lowlevel_atomic_packed_verify_invalid.pto).
- [Shuffle width rejection](../../../AscendNPU-IR/third-party/ptoas/test/lit/vpto/simt_lowlevel_shuffle_verify_invalid.pto).
- [Scalar SIMT math emission](../../../AscendNPU-IR/third-party/ptoas/test/lit/vpto/simt_lowlevel_float_math_vpto_llvm.pto).
- [SIMT section round trip](../../../AscendNPU-IR/third-party/ptoas/test/lit/pto/section_simt_roundtrip.pto).
- [Packed GM loads/stores](../../../AscendNPU-IR/third-party/ptoas/test/lit/vpto/simt_lowlevel_vector_ldg_stg_packed_types_vpto_llvm.pto).

## Verification performed

- Enumerated concrete `def` operations in AscendDPXOps.td and checked each appears exactly once in the matrix (223 rows).
- Compared op definitions, verifier restrictions, and relevant LLVM emitter paths; checked representative existing PTO lit fixtures for atomics, shuffle validation, scalar math, SIMT sections and GM memory.
- Confirmed the AscendNPU-IR gitlink and checked-out PTOAS submodule HEAD match; baseline source references use that pinned submodule.
- Checked relative report links and whitespace. No compiler implementation was changed; no commit was created.

## Upstream reassessment — AscendNPU-IR submodule versus original repository, 2026-10-15

### Repository and exact revision

The earlier sections describe the **AscendNPU-IR PTOAS submodule baseline** (`third-party/ptoas` at `3c1c98fcc8ec0c49b8bededc9a0c7599c6a71a02`). This section reassesses instruction support against the freshly fetched original project's published upstream branch; it does not replace that historical matrix. Template coverage remains excluded.

The original GitHub repository, [hw-native-sys/PTOAS](https://github.com/hw-native-sys/PTOAS), now describes itself as the official read-only mirror of [cann/pto-as on GitCode](https://gitcode.com/cann/pto-as). Its default branch is **`master`**, and a live `git ls-remote` query returned **no `refs/heads/main`**. Accordingly, `master` is the upstream branch assessed here.

| Item | Value |
| --- | --- |
| Fetch source | `https://github.com/hw-native-sys/PTOAS.git`, `refs/heads/master` |
| Fetch/check time | 2026-10-15, approximately 18:00 UTC |
| Audited upstream HEAD | [`0770960cb874cad81301bcb06b8008349c5db416`](https://github.com/hw-native-sys/PTOAS/commit/0770960cb874cad81301bcb06b8008349c5db416) |
| HEAD commit time | 2026-10-15 17:15:44 +08:00 |
| HEAD subject | `fix(ptodsl): compile emitc ptoas output before the native fat link` |
| AscendNPU-IR submodule baseline | `third-party/ptoas` at `3c1c98fcc8ec0c49b8bededc9a0c7599c6a71a02` |
| Common ancestor | `0e980bc3ec03858a99de1786771377aebd075387` |
| History difference | 368 commits reachable only from upstream; 7 reachable only from the pinned submodule |

The fetch and detached source inspection were performed in a temporary audit checkout. The project's existing PTOAS working trees were not switched or merged. A direct GitCode query required credentials, so this assessment establishes the latest fetched **official GitHub mirror** tip; synchronization with the canonical GitCode tip could not be independently checked.

Because the submodule points to a fork, “new upstream changes” means changes absent from the pinned submodule's history, not necessarily commits dated after its latest commit. The submodule's latest commit is `3c1c98fcc` (`feat: add device object file emission support`, 2026-09-28); the important upstream scalar-interface change is dated September 17 but is absent from that pinned history.

### Result of reassessing the previous gap categories

**No entire gap category from the summary table is fully closed. Three categories have meaningful partial improvements; ten remain open without a corresponding new implementation.** Individual instruction/type cases have been resolved, as detailed in the next table.

| Previous gap category | Upstream result | Evidence and remaining limitation |
| --- | --- | --- |
| SIMT scheduling yield | Still open | No SIMT scheduling-yield op/emitter was found. `pto.yield` remains a fusion-region terminator. |
| Physical core ID | Still open | Runtime query still emits `llvm.hivm.tpe.get.VECCOREID`; no exact DPX `tpe_get_COREID` counterpart found. |
| Arbitrary shuffle control | Still open | The same one-SSA-operand plus static width interface remains; width must still be 16 or 32. No independent dynamic lane_mask/clamp or raw-control input. |
| Volatile load semantics | Still open | Public load/ldg operations still lack a volatile control, and scalar load emission creates a normal LLVM load. Scheduler recognition of generic volatile attributes does not establish propagation to LLVM memory operations. |
| Atomic INC/DEC | Still open | The nine atomic families remain; no dedicated atomic_inc/atomic_dec op was added. |
| i8/i16 integer atomics | Still open | Atomic value types are still i32/i64/f16/bf16/f32 and the two packed float pairs. No subword integer atomic interface or new software fallback found. |
| Packed atomic old values | Still open | The verifier still rejects any use of the packed atomic result on beta.1. |
| i64 UB atomics | Still open | The verifier still explicitly rejects i64 UB-space atomics. |
| Scalar math type coverage | Partially improved | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) adds integer `pto.absi`, generic `pto.absf` including scalar bf16, and broadens `pto.pow` to generic floating types including scalar bf16. Integer pow remains absent; exp/log/sqrt still exclude scalar bf16, and sin/cos remain f32 only. |
| Missing SIMT math-library families | Partially improved | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) adds `pto.remf`, covering the fmod-style remainder case. No dedicated erf/erfc, tan/tanh, reciprocal/rsqrt, log1p/expm1, ldexp/ilogb, gamma/Bessel, norm or inverse-CDF families were found. |
| Explicit arithmetic rounding | Partially improved | [`b023e0f5f`](https://github.com/hw-native-sys/PTOAS/commit/b023e0f5f20f7b0fc48c2b94a96c7452ae110d49) adds opt-in high-precision scalar f32 division with an RNE software path. No general arithmetic rounding selector, or rz/rd/ru arithmetic family, was added. Conversion rounding remains a separate interface. |
| Fast-math variants | Still open | Generic arithmetic now carries fastmath flags, but these do not provide the named DPX fast-math helper families or establish their approximation contracts. |
| Scalar call ABI | Still open | No dedicated counterpart preserving DPX call_scalar's `use_shmem_offset` and scalar/SIMT ABI was found. Generic function calls alone do not close this gap. |

Evidence for unchanged restrictions: [PTO SIMT definitions](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/include/PTO/IR/VPTOOps.td), [atomic verifier](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/lib/PTO/IR/VPTO/memop/VPTOMemOpInternal.h), [shuffle verifier](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/lib/PTO/IR/VPTO/vecop/detail/VPTOVecOpShuffleRedux.h), [scalar load emission](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/lib/PTO/Transforms/VPTO/VPTOCANN900LLVMEmitterTypePatterns.cpp), and [runtime-query emission](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/lib/PTO/Transforms/VPTO/VPTOCANN900LLVMEmitterTemplates.h). Definitions, verifiers and lowering paths were inspected together; merely moving an implementation to a different file was not counted as a resolution.

### Instruction-level changes relative to the original matrix

These are the **10 DPX rows whose source-correspondence status changes**. “Exists” here includes execution-domain-independent generic PTO operations usable inside SIMT, as well as dedicated SIMT operations; it does not imply a single native machine instruction or hardware validation.

| DPX operation | Baseline → upstream | What was added / resolved | Introducing upstream commit |
| --- | --- | --- | --- |
| `abs` | Partial → Exists | `pto.absi` now covers integer values; generic `pto.absf` includes scalar bf16 and lowers to `math.absf`. | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) |
| `float_as_int` | Compose → Exists | Scalar `pto.bitcast` handles equal-width f32-to-i32 reinterpretation. | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) |
| `float_as_uint` | Compose → Exists | Scalar `pto.bitcast` handles equal-width f32-to-i32 bits; preserve unsigned interpretation at consumers. | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) |
| `int_as_float` | Compose → Exists | Scalar `pto.bitcast` handles equal-width i32-to-f32 reinterpretation. | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) |
| `uint_as_float` | Compose → Exists | Scalar `pto.bitcast` handles equal-width integer-to-f32 bits. | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) |
| `div` | Compose → Exists | `pto.divi` with signedness for integer division; `pto.divf` for floating division. | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) |
| `fdiv` | Compose → Exists | `pto.divf` lowers to ordinary `arith.divf`; optional high-precision scalar f32 mode is also available. | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) |
| `udiv` | Compose → Exists | `pto.divi` with unsigned signedness lowers to `arith.divui`. | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) |
| `div_rn` | Missing → Partial | `pto.divf` with high_precision has a scalar f32 RNE SoftLib implementation. Special inputs use native division; full source-contract/runtime equivalence was not tested. Other directed modes remain unsupported. | [`b023e0f5f`](https://github.com/hw-native-sys/PTOAS/commit/b023e0f5f20f7b0fc48c2b94a96c7452ae110d49) |
| `fmod` | Missing → Exists | `pto.remf` lowers to `arith.remf`; this corresponds to fmod-style floating remainder. It does not implement the distinct `remainder` operation. | [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) |

Source evidence: [generic PTO definitions](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/include/PTO/IR/PTOGenericOps.td), [value/tile bitcast definition](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/include/PTO/IR/PTOOps.td), [generic lowering](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/lib/PTO/Transforms/Lowering/PTOLowerGenericOps.cpp), [generic verifiers](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/lib/PTO/IR/PTOGenericOps.cpp), [scalar division helper](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/lib/SoftOps/div_float.py), and [division SoftLib expansion](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/lib/PTO/Transforms/Lowering/PTOExpandSoftLib.cpp).

The other **213 DPX rows retain their classification**, although some use new operation names or interfaces. Reapplying the same classification scheme gives:

| Classification | AscendNPU-IR submodule | Upstream revision | Interpretation |
| --- | ---: | ---: | --- |
| Exists | 29 | 38 | Nine additional rows have a corresponding PTO operation/interface. |
| Partial | 63 | 63 | Abs moves to Exists; div_rn moves from Missing to Partial. Pow improves within Partial. |
| Compose | 27 | 20 | Three division rows and four bit-reinterpretation rows now have a direct PTO interface. |
| Missing | 104 | 102 | fmod moves to Exists; div_rn moves to Partial. |
| Total | 223 | 223 | Same AscendDPX operation inventory. |

The seven Compose → Exists rows were already expressible through standard MLIR/LLVM in the baseline. They are **PTO-interface additions**, not seven previously impossible computations becoming possible. Likewise, `pow`'s bf16 improvement does not resolve its integer form. Its broadened type definition and `math.powf` lowering establish IR-level correspondence; target-specific numerical behavior remains untested.

### Interface changes the bridge must account for

These changes preserve or broaden capabilities but require updating the old matrix's PTO spellings when targeting current upstream:

| Baseline PTO interface | Current upstream interface | Consequence |
| --- | --- | --- |
| `pto.convert` | `pto.ftof`, `pto.ftoi`, `pto.itof`; integer conversion helpers such as `pto.exti`/`pto.trunci` | Select the conversion by source/destination category. Explicit rounding, saturation and low-precision payloads use SIMT-specific verification/emission; an ordinary SIMT scope alone no longer forces hardware conversion semantics. |
| `pto.redux_add`, `pto.redux_min`, `pto.redux_max` | `pto.redux_addi/addf`, `pto.redux_mini/minf`, `pto.redux_maxi/maxf` | Integer/float forms are split; integer min/max require signedness. Capabilities were retained, not newly added reductions. |
| `pto.fmin`, `pto.fmax` | `pto.minf`, `pto.maxf`; plus `pto.minimum`, `pto.maximum`, `pto.mini`, `pto.maxi` | Choose minNum/maxNum versus NaN-propagating floating semantics explicitly; integer extrema have signedness. |
| `pto.bitcast` restricted to tile views | Same mnemonic also accepts scalar/builtin-vector values | The four DPX bit-reinterpretation operations can now use PTO directly. |
| signedness allowed on all atomic families | signedness retained for atomic min/max; rejected on other atomic families | Strip obsolete signedness attributes from other atomic operations. This does not expand their supported value types. |
| scalar-value operations marked as SIMT operations | execution-domain-independent scalar/generic families | Do not infer the execution domain solely from the old SimtOpInterface membership. Launches and SIMT-only operations still define the SIMT boundary. |

All rows in this interface table originate in [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed). Existing conversion support was reorganized; this is not evidence that every formerly unsupported conversion pair was implemented.

### Relevant upstream commits

| Commit | Date | Relevance to this audit |
| --- | --- | --- |
| [`fde3b353e`](https://github.com/hw-native-sys/PTOAS/commit/fde3b353ecc791861658e5c9a5503f61281553ed) | 2026-09-17 | Introduces generic scalar PTO operations and their legalization, scalar value bitcasts, split conversions/reductions/extrema, broader abs/pow interfaces. Main source of the instruction-level changes above. |
| [`b023e0f5f`](https://github.com/hw-native-sys/PTOAS/commit/b023e0f5f20f7b0fc48c2b94a96c7452ae110d49) | 2026-09-29 | Adds high-precision scalar f32 division, the scalar SoftLib helper, verification, and division tests. Resolves part of the div_rn requirement rather than the whole explicit-rounding family. |
| [`9d3bd3323`](https://github.com/hw-native-sys/PTOAS/commit/9d3bd3323ab48fdb1cf199d9a2e10d848afb5b0d) | 2026-09-16 | Adds non-negative range information to existing SIMT hardware-ID operations and range-driven arithmetic optimizations. It does not add the missing COREID query. |
| [`b772bfe11`](https://github.com/hw-native-sys/PTOAS/commit/b772bfe1198bd7b743790ad42c3380555963a7c6) | 2026-09-10 | Removes the legacy monolithic LLVM emitter; current assessment follows the active CANN900 emitter. File deletion here is not loss of all SIMT emission support. |
| [`e79a3393a`](https://github.com/hw-native-sys/PTOAS/commit/e79a3393ac097d1f27482a2e5e826f2f609fb1fa) / [`dc6834774`](https://github.com/hw-native-sys/PTOAS/commit/dc6834774cd83bce2ef955ba080125a4fbe71244) | 2026-09-10 / 2026-09-11 | Split IR/verifier/emitter implementation files. The old VPTO.cpp/VPTOLLVMEmitter.cpp locations are insufficient for assessing current support. |

### Verification and limits of the reassessment

- Fetched the original GitHub repository's actual default branch and compared its exact SHA with AscendNPU-IR's recorded PTOAS submodule pin; excluded other fork tips, feature branches and unmerged PRs.
- Rechecked the same 223 DPX operations against the fetched definitions, generic legalization, current SIMT verifiers and active LLVM emitter; classified ten status changes and 213 unchanged statuses.
- Traced the additions to the introducing commits using their diffs and checked that those commits are reachable from the fetched HEAD and absent from the pinned AscendNPU-IR PTOAS submodule.
- Inspected upstream regression fixtures for [scalar precision expansion](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/test/lit/vpto/scalar_div_precision.pto), [SIMT conversion routing](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/test/lit/vpto/simt_generic_convert_to_arith.pto), [packed atomic-result rejection](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/test/lit/vpto/simt_lowlevel_atomic_packed_verify_invalid.pto), and [invalid shuffle width](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/test/lit/vpto/simt_lowlevel_shuffle_verify_invalid.pto).
- Upstream also includes a [materialized scalar-division numeric test](https://github.com/hw-native-sys/PTOAS/blob/0770960cb874cad81301bcb06b8008349c5db416/ptodsl/tests/test_div_precision_numeric.py) comparing 100,000 random input pairs plus boundary cases against NumPy. This test was inspected, **not executed** in this audit.
- Checked report row accounting, source paths, relevant source invariants and commit ancestry. No upstream build, simulator or hardware test was run. New source-level support does not imply the existing bridge already emits or validates these interfaces.
