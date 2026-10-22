# Slide 2 validation: KKT predicate-valued memory

The supplied native dump reproduces the predicate store/reload pattern. The inspected VMI memory operations reject mask-valued payloads/results. The current bridge instead accepts a numeric-byte carrier route for the extracted producer and an isolated reload. No numerical equivalence of that route to packed predicate memory was tested.

## Source trigger and native evidence

In `../chunk_scaled_dot_kkt_fwd.py:91`, the kernel constructs:

```python
m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
b_A = tl.where(m_A, b_A, 0)
```

The original kernel has the same statements at `Q2TritonKernel/src/kernels/fla/ops/common/chunk_scaled_dot_kkt.py:75`. NPU-IR outlines predicate production and consumption into different helpers and materializes an intermediate UB predicate buffer. The Python source does not explicitly store a boolean tensor.

In [the supplied dump](../all-mlir.npuir.mlir):

| Location | Observation |
|---|---|
| 197421–197481, `...outlined_vf_4` | Producer helper; integer `GT` and validity predicates combine into the causal mask. |
| 197478 | `masked_store <NORM_B8>` writes a `vector<64xi1>` payload into `memref<64xi1>` with `functionType=pb32`; its separate controlling predicate is ALL. |
| 197482–197522, `...outlined_vf_5` | Consumer helper. |
| 197496 | PB32 `vload` reloads `vector<64xi1>`. |
| 197510 | `vsel` uses the reloaded predicate to select computed values or zero. |
| 199451 / 199454 | Native store lowers to `hivm_regbaseintrins.intr.hivm.pstu.b32` followed by `...vstas`, with initialized alignment state. |
| 199098–199103 | Native reload uses `...plds.b8` followed by two `...pintlv.b8` operations to recover the PB32 layout. |

These native instructions show physical predicate-memory support. They do not establish an equivalent first-class VMI interface.

## Fresh replay results

`helper-vf-4.mlir` and `helper-vf-5.mlir` copy the complete functions and aliases from the same `ExpandStridedMetadata` snapshot. `--lower-affine --canonicalize` produces the separately retained normalized inputs. [Source ranges and hash](source.json), [bridge commands/results](bridge-results.json), and [tool provenance](tool-provenance.json) identify the exact inputs and binaries.

| Check | Result |
|---|---|
| Full producer helper → bridge | Exit 0. [VMI output](helper-vf-4.vmi.mlir), lines 130–137, retains a PB32 pointer marker and stores a `vreg<64xui8>` carrier, generated from the predicate through select/truncate. |
| Derived predicate reload → bridge | Exit 0. [VMI output](predicate-reload-only.vmi.mlir) loads a `vreg<64xui8>` carrier and compares it against zero to reconstruct a mask. |
| Direct VMI predicate-valued load | Exit 1: result must be a VMI logical vector register, but was `!pto.vmi.mask<64xpred>`. [Diagnostic](predicate-vload.stderr) |
| Direct VMI predicate-valued store | Exit 1: payload operand must be a VMI logical vector register, but was `!pto.vmi.mask<64xpred>`. [Diagnostic](predicate-vstore.stderr) |
| Numeric VMI load/store control | Exit 0. [Input](numeric-memory-control.pto) |

The direct VMI probes use the existing upstream tool build from October 21 at `37f5f25e3b5b433e7cc760d148fea83dddef9934`; this is not a newly fetched upstream revision. [Interface commands/results](interface-results.json) retain all exits. The derived reload exposes the original computed memory view as an argument and returns the original load result; it is an interface probe, not an unchanged full helper or runtime-equivalence test.

## Workaround and separate blocker

The byte-carrier route is different from native packed predicate memory. The bridge preserves PB32 intent with `ui8 → ui32 → ui8` pointer markers and provides the physical repair pass `--finalize-ptoas-vpto-predicate-memory`. The inspected default pipeline leaves bridge finalizers opt-in. Successful VMI conversion alone therefore does not establish correct packed memory layout, address units, native-call ABI, or end-to-end execution. No new physical instruction is required by this finding.

The complete consumer helper fails bridge conversion on its later BF16 packed-store wrapper:

```mlir
%22 = builtin.unrealized_conversion_cast %19
    : vector<64xbf16> to vector<128xbf16>
```

See [the actual diagnostic](helper-vf-5.bridge.stderr). This is a separate existing BF16 wrapper limitation, not a predicate-load rejection. It was recorded to avoid attributing the full-helper failure to Slide 2; no compiler fix or unrelated issue investigation was performed.

Requested VMI capability: predicate-valued load/store with PB32 packing/address semantics, lowering to existing physical instructions and a supported buffer representation. Confidence is high for the reproduced native requirement and the tested direct-interface rejection. Full-kernel VMI/VPTO/LLVM compilation and numerical/device validation remain unestablished by these replays.
