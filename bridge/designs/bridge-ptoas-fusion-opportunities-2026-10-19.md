# AVE to VMI: performance opportunities with small examples

The useful question is: **can the bridge express the same useful calculation while giving PTOAS less machine work to do?**

There are three common savings:

- **Move less data:** keep an intermediate value in a register instead of writing and reading UB memory.
- **Do fewer instructions:** use a fused instruction or remove a packing step.
- **Repeat less setup:** share work across several groups or loop iterations.

Below are nine examples. They are not restricted to Q2 kernels. Some preserve an efficiency native AVE already has; others show a possible new rewrite. None establishes a measured AVE-versus-PTOAS speedup.

**How to read the examples:** these are simplified, MLIR-like excerpts using real operation names. Most types, attributes and address calculations are omitted so the transformation is visible. `...` abbreviates details; these are not standalone compiler inputs. `mask` means the active vector elements. UB is on-chip memory. A VMI operation may become several machine instructions, or disappear entirely.

**Status labels:**

- **Available:** upstream PTOAS has implementation and regression coverage for the stated kind of optimization. This does not mean every shape works or a new test run was performed here.
- **Bridge change needed:** the bridge must emit or preserve the right representation.
- **Future work:** current plain AVE-to-VMI output does not automatically receive this optimization.

Sources are pinned to the October 19 audit: NPU-IR `8b8f40f8`, upstream PTOAS `ec8d1810`. This rewrite simplifies that evidence; it does not claim a newer upstream audit. Full constraints and revisions are in the [technical reference](bridge-ptoas-fusion-reference-2026-10-19.md).

## Start with the savings

| Example | Work that could disappear | Status |
|---|---|---|
| 1. Keep a temporary in a register | One UB store and one UB reload per matched chunk | Available in eligible PTOAS tile fusion; future integration for plain VMI |
| 2. Multiply and add together | Two arithmetic instructions become one | VMI instruction available; bridge mapping needed |
| 3. Subtract and exponentiate together | Two arithmetic instructions become one | Bridge already preserves fused AVE expdif; later discovery is restricted |
| 4. Load and broadcast together | Scalar load plus vector duplication becomes one broadcast load | Available upstream; bridge interface migration needed |
| 5. Remove preparation before widening | An intermediate interleave and its temporary carriers | Bridge rewrite already exists; final instruction saving needs comparison |
| 6. Let the store handle the arrangement | A separate register-interleave instruction | Available for supported layouts |
| 7. Combine two sums before reducing | Two reductions become one, with the combine retained | Available for recognized equal masks and legal arithmetic |
| 8. Load several group values together | Repeated broadcast loads and selection work | Available for supported grouped shapes; bridge grouping needed |
| 9. Share store-stream setup | Repeated initialization and flushing | Available for eligible physical streams; not general point stores |

## 1. Keep a temporary value in a register

**The calculation:** `out = (a + b) * c`.

AVE may contain a temporary memory round trip, for example after two separately lowered stages:

~~~mlir
%t = ave.hir.vadd %a, %b, %mask
ave.hir.masked_store <NORM_B32> %tmp[%i], %mask, %t

%again = ave.hir.vload <NORM> %tmp[%i]
%out = ave.hir.vmul %again, %c, %mask
~~~

A better VMI representation, when the memory round trip is unnecessary:

~~~mlir
%t = pto.vmi.vadd %a, %b, %mask
%out = pto.vmi.vmul %t, %c, %mask
~~~

The multiply directly uses the add result. There is no intermediate store or reload.

**Concrete saving:** for a 64-element F32 chunk, removing a full store and reload avoids 256 bytes written plus 256 bytes read in UB. The final output store is still required. It can also remove address work and dependencies.

**What the bridge needs to know:** the later load reads exactly the just-written values, with no intervening modification; no other observer requires the temporary to be materialized. Matching addresses and compatible active lanes matter.

**Available today?** PTOAS can eliminate these round trips inside eligible tile-fusion regions. Its general raw-VMI route does not automatically perform this region-local optimization. The bridge could perform a verified local forwarding rewrite, or preserve a supported structured tile computation so PTOAS can do it later. Merely adding a fusion flag to the current VMI path is insufficient.

Evidence: [PTOAS test: remove intermediate stores and reloads](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/tile_fusion/op_fusion_backend_lifecycle_level3.pto#L99). See [reference C09/C10](bridge-ptoas-fusion-reference-2026-10-19.md#c09-tile-loop-fusion-requires-a-supported-structured-entry-point).

## 2. Turn multiply followed by add into one instruction

**The calculation:** `out = acc + a * b`.

AVE before fusion:

~~~mlir
%product = ave.hir.vmul %a, %b, %mask
%out = ave.hir.vadd %acc, %product, %mask
~~~

VMI can express the combined operation:

~~~mlir
%out = pto.vmi.vmula %acc, %a, %b, %mask
~~~

For a supported physical chunk, this becomes one multiply-add instruction.

**Concrete saving:** one fused arithmetic instruction instead of separate multiply and add; one intermediate product value is no longer needed when it has no other users.

**Important condition:** fused floating-point multiply-add rounds differently from separate multiplication and addition. This rewrite needs permission from the source numerical contract. Masks must also agree appropriately.

**Available today?** VMI supports explicit `vmula`. NPU-IR can already create an AVE `vmula` under its fusion option, but the assessed bridge has no mapping for that fused operation. A useful bridge change is to translate an already valid AVE `vmula` directly. Automatic discovery from separate VMI multiply/add was not found in the audited PTOAS transforms.

Evidence: [VMI fused multiply-add definition](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/include/PTO/IR/VMIOps.td#L1782); [Current bridge arithmetic mappings](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4720). See [reference C08](bridge-ptoas-fusion-reference-2026-10-19.md#c08-explicitly-map-existing-ave-fused-multiply-add).

## 3. Turn subtraction followed by exponential into one instruction

**The calculation:** `out = exp(a - b)`, common in normalization calculations.

AVE before fusion:

~~~mlir
%difference = ave.hir.vsub %a, %b, %mask
%out = ave.hir.vexp %difference, %mask
~~~

VMI can express it directly:

~~~mlir
%out = pto.vmi.vexpdif %a, %b, %mask
~~~

**Concrete saving:** a supported physical `vexpdif` replaces subtraction plus exponential. The intermediate difference can disappear if it is not otherwise needed.

**Available today?** The bridge already maps fused AVE `vexpdif` to VMI `vexpdif`. That is worth preserving. PTOAS can also discover a physical subtract/exp pair, but the audited pass requires F32 values, the same mask, a one-use subtraction, and a tile-fusion region. Separately emitting VMI sub/exp does not guarantee later fusion on the ordinary bridge path.

**Bridge opportunity:** retain NPU-IR's valid fused operation. If it sees separate operations, fusion needs a supported combine and the appropriate numerical contract; it should not assume PTOAS will always repair the sequence.

Evidence: [PTOAS subtraction/exponential fusion](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOVexpdifFusion.cpp#L30); [Current bridge expdif mapping](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3968). See [reference C07](bridge-ptoas-fusion-reference-2026-10-19.md#c07-keep-already-fused-expdif-do-not-assume-raw-subexp-is-re-fused).

## 4. Read one value directly into a broadcast vector

**The calculation:** every lane needs the same scale from UB.

AVE already has a broadcast load:

~~~mlir
%scale = ave.hir.vload <BRC_B32> %src[%zero]
    // Logical result: 64 F32 copies of one memory value.
~~~

The bridge can express that using the upstream VMI interface:

~~~mlir
%one = pto.vmi.vload %src[%zero]
    : !pto.ptr<f32, ub> -> !pto.vmi.vreg<1xf32>
%scale = pto.vmi.vbrc %one
    : !pto.vmi.vreg<1xf32> -> !pto.vmi.vreg<64xf32>
~~~

PTOAS can combine the two logical operations into a physical broadcast load:

~~~text
pto.vlds {dist = "BRC_B32"}
~~~

**Concrete benefit:** avoid the alternative of a scalar-pipeline load followed by a scalar-to-vector duplication. The memory read and physical broadcast happen together.

**This mainly preserves native efficiency:** AVE's BRC operation already requests a vector broadcast load. It is not automatically a speedup over good native AVE output.

**Available today?** Upstream has the combine. The assessed bridge still uses an older scalar-load operation and older broadcast-load interface, so migration is needed. A one-lane logical result needs only the singleton vector load, without a further logical broadcast. The exact BF16 64-lane broadcast followed by F32 widening in F02 remains unverified.

Evidence: [PTOAS test: scalar load and broadcast become BRC](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_pre_assignment_combine_scalar_ub_load_broadcast.pto#L50). See [reference C01](bridge-ptoas-fusion-reference-2026-10-19.md#c01-expose-the-supported-loadbroadcast-composition).

## 5. Widen the useful values without first duplicating them

**The calculation:** convert 128 BF16 values to F32, producing two groups of 64 results.

An AVE sequence prepares the narrow values for two conversions:

~~~mlir
%lo, %hi = ave.hir.vintlv %packed, %packed
    {layout_change = #ave<layout_change SPARSE>}
%a = ave.hir.vextf %lo, <part_even>, %mask
%b = ave.hir.vextf %hi, <part_even>, %mask
~~~

Here the interleave duplicates values into positions used by `part_even`. The combined meaning is simply to widen the original values and obtain their two halves.

The bridge can rewrite it as:

~~~mlir
%wide = pto.vmi.extf %packed
    : !pto.vmi.vreg<128xbf16> -> !pto.vmi.vreg<128xf32>
%a = pto.vmi.shuffle %wide[[0, 1, ..., 63]]
%b = pto.vmi.shuffle %wide[[64, 65, ..., 127]]
~~~

**Concrete benefit:** remove the narrow-value duplication/interleave step. The two shuffles select complete F32 register halves and may become simple references to those registers, with no executed shuffle instruction.

There is still enough conversion work to produce 128 F32 values. One VMI `extf` does **not** mean one physical conversion instruction.

**Available today?** This is the existing F08 bridge rewrite. It shows why the bridge can help: it recognizes that several AVE instructions jointly describe one logical conversion. The final saving versus native emitted code still needs instruction comparison and timing.

Evidence: [Bridge test: split and widen](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-sparse-interleave-extf.mlir:9); [Current bridge split/widen rewrite](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4081). See [reference C03](bridge-ptoas-fusion-reference-2026-10-19.md#c03-let-ptoas-fold-layout-work-into-loadsstores-or-move-it-after-arithmetic).

## 6. Let the store put values in order

**The calculation:** widen narrow values, multiply them, and write the results in the original logical order.

An AVE-style physical sequence may produce separate even/odd groups:

~~~mlir
// Schematic AVE sequence; the half-carrier preparation is omitted.
%even = ave.hir.vextf %even_input, <part_even>, %mask
%odd  = ave.hir.vextf %odd_input,  <part_even>, %mask
%even_product = ave.hir.vmul %even, %scale, %mask
%odd_product  = ave.hir.vmul %odd,  %scale, %mask
// Rearrange the products, then store them in logical order.
~~~

Instead of prescribing that physical rearrangement, the bridge can expose the whole logical calculation:

~~~mlir
%wide = pto.vmi.extf %packed
    : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
%product = pto.vmi.vmul %wide, %scales
pto.vmi.vstore %product, %dst[%zero]
~~~

PTOAS can retain the even/odd arrangement during arithmetic and use a store that performs the final ordering:

~~~text
Conversions -> multiplies -> interleaving store
                            (vstsx2)
~~~

rather than:

~~~text
Conversions -> multiplies -> separate interleave -> ordinary stores
~~~

**Concrete saving:** the standalone interleave can disappear. The memory write still happens; its supported instruction mode handles the ordering.

**Bridge help:** describe logical values and the real store destination/order. Avoid forcing an intermediate contiguous register arrangement when only memory needs contiguous order.

**Available today?** PTOAS has this layout-folding behavior for supported shapes. It is not a universal store transformation, and it does not establish that native AVE always emits the longer sequence.

Evidence: [PTOAS test: store without a separate interleave](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_layout_fold_store.pto#L37). See [reference C03](bridge-ptoas-fusion-reference-2026-10-19.md#c03-let-ptoas-fold-layout-work-into-loadsstores-or-move-it-after-arithmetic).

## 7. Combine the inputs before doing an expensive reduction

**The calculation:** `sum(a) + sum(b)`, with the same active lanes in both vectors.

AVE can express two reductions and addition of their lane-zero results:

~~~mlir
%ra = ave.hir.reduction <add>, %a, %mask
%rb = ave.hir.reduction <add>, %b, %mask
%sum = ave.hir.vadd %ra, %rb, %first_lane
~~~

A useful VMI representation retains the singleton results without unnecessary expansion:

~~~mlir
%ra = pto.vmi.vcadd %a, %mask {reassoc}  // One logical result.
%rb = pto.vmi.vcadd %b, %mask {reassoc}
%sum = pto.vmi.vadd %ra, %rb
~~~

For a legal matching tree, PTOAS can produce the physical equivalent of:

~~~text
combined = vector_add(a, b, mask)
sum      = reduction_sum(combined, mask)
~~~

**Concrete saving:** two reductions plus one add become one vector add plus one reduction. One reduction instruction disappears in the isolated two-chunk case.

**Conditions:** mask equivalence must be recognizable, types must agree, and the floating-point contract must permit the changed addition order. If other users still need the old reductions, those instructions may remain.

**Bridge help:** preserve reduction operations, shared predicates and direct result uses. Avoid inserting broadcast/projection or memory round trips that hide the reduction tree. This may require a checked consumer rewrite; the current F04 bridge restores wide result types with broadcasts.

**Available today?** PTOAS has the physical reduction combiner. Whether a particular AVE chain reaches it without intervening carrier operations must be tested. This pass does not itself promise to remove every F04 broadcast.

Evidence: [PTOAS test: combine reductions with equal masks](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_to_vpto_reduce_addf_multichunk.pto#L43). See [reference C04](bridge-ptoas-fusion-reference-2026-10-19.md#c04-preserve-reduction-structure-and-recognizable-mask-equivalence).

## 8. Load values for several groups together

**The calculation:** four groups each use their own BF16 scale:

~~~text
Memory scales:     [s0, s1, s2, s3]
Logical vector:   [s0 repeated 32 times,
                   s1 repeated 32 times,
                   s2 repeated 32 times,
                   s3 repeated 32 times]
~~~

A straightforward AVE representation might load/broadcast each group separately and later combine the groups:

~~~mlir
%g0 = ave.hir.vload <BRC_B16> %scales[%zero]
%g1 = ave.hir.vload <BRC_B16> %scales[%one]
%g2 = ave.hir.vload <BRC_B16> %scales[%two]
%g3 = ave.hir.vload <BRC_B16> %scales[%three]
// Then construct/use the corresponding groups.
~~~

If those groups belong to one logical computation, the bridge could express that relationship directly:

~~~mlir
%slots = pto.vmi.vload %scales[%aligned_off], %one {group = 4}
    : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<4xbf16>
%all_scales = pto.vmi.vbrc %slots {group = 4}
    : !pto.vmi.vreg<4xbf16> -> !pto.vmi.vreg<128xbf16>
~~~

For the supported layout in the cited test, PTOAS selects a packet-distribution load, `E2B_B16`, rather than a load/selector fallback. That instruction reads a packet and distributes its values into the required physical groups.

**Concrete opportunity:** replace repeated scalar-address broadcast loads and group-assembly work with a packet load. This is a stronger opportunity than simply renaming four separate loads.

**Bridge help:** preserve which lanes belong to each group, the scale addresses, and the common computation. The current bridge does not generally construct this grouped form.

**Available today?** Upstream has a positive four-group BF16 test with a deinterleaved result layout and aligned address. The example above leaves layout selection to PTOAS; the exact unconstrained consumer chain is not certified by that test. The packet also reads surplus values, so the physical read must be permitted by the memory contract. Other shapes use longer fallbacks.

Evidence: [PTOAS test: four BF16 groups use an E2B load](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_to_vpto_group_broadcast_load_e2b_four_group.pto#L20). See [reference C02](bridge-ptoas-fusion-reference-2026-10-19.md#c02-expose-groups-and-adjacent-rows-as-operations-not-vague-metadata).

## 9. Set up a continuous store stream once

**The calculation:** write several contiguous chunks to an unaligned destination.

AVE may carry this intent:

~~~mlir
// Schematic loop: offset = 1 + iteration * 64, four iterations.
scf.for ... {
  ave.hir.masked_store <NORM_B32> %dst[%offset], %all, %chunk
      {hivm.is_continuous}
}
~~~

The corresponding logical VMI stores can remain simple:

~~~mlir
scf.for ... {
  pto.vmi.vstore %chunk, %dst[%offset]
      : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>
}
~~~

Without stream fusion, physical lowering may repeat:

~~~text
For each chunk: initialize -> write -> flush
~~~

Eligible stream fusion changes it to:

~~~text
initialize once
For each chunk: write and carry the updated stream state
flush once
~~~

**Concrete saving:** for four eligible iterations, potentially remove three initializations and three flushes. The four data writes remain.

**Bridge help:** keep the loop bounds and address progression easy to analyze, preserve genuine continuity, and avoid unnecessary opaque operations between stores. Retain memory facts where a supported analysis can consume them.

**Available today?** PTOAS supports eligible physical streams and constant-trip loops. Native NPU-IR also performs stream hoisting, so the first goal is to preserve that efficiency through the bridge. A one-value `vsts` is not automatically a stream; this example does not solve F05's singleton stores. Real dependencies can prevent hoisting, and carrying state can increase register pressure.

Evidence: [PTOAS test: shared stream setup and final flush](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_to_vpto_stateful_stream_fusion.pto#L136). See [reference C06](bridge-ptoas-fusion-reference-2026-10-19.md#c06-make-continuity-provable-but-recognize-the-physical-stream-prerequisite).

## What should the bridge preserve?

| Information | Example of the benefit |
|---|---|
| An intermediate is private and need not be stored | Example 1: delete temporary UB traffic |
| Arithmetic is allowed to fuse or reassociate | Examples 2, 3 and 7: use combined arithmetic |
| Several physical carriers represent one logical vector | Examples 5 and 6: avoid packing/rearrangement |
| Lanes belong to known groups | Example 8: use grouped packet loads |
| Addresses are contiguous, aligned, or disjoint | Examples 1, 4, 8 and 9: enable safe memory combines |
| Predicates are genuinely equal | Examples 3 and 7: expose matching arithmetic |

Some facts already exist earlier in NPU-IR. They need to reach a representation or analysis PTOAS actually understands. An arbitrary new attribute is not enough.

## What I would try first

1. **Preserve and map existing fused arithmetic**, especially the missing AVE vmula mapping, rather than relying on later rediscovery.
2. **Use supported upstream load/broadcast forms** and test the exact consumers before retiring old bridge workarounds.
3. **Keep logical lane relationships visible**, preserving F08 and allowing load/store layout folding.
4. **Investigate temporary UB traffic and grouped computations** for larger gains, accepting that some require a structured bridge entry or additional compiler work.

The benefit is established only after the final physical code shows the saved instructions or traffic. Device timing then determines how much those savings matter.

For exact matcher conditions, source paths, pass order, negative tests and version differences, use the [technical reference](bridge-ptoas-fusion-reference-2026-10-19.md). In particular, plain VMI does not automatically enter PTOAS's tile-fusion framework; [PTOAS tile-fusion pipeline entry](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas_pipeline.cpp#L1729) shows that pipeline split.

Only documentation was changed. No new compiler execution or performance measurement is claimed.
