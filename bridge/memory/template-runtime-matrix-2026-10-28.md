# Existing template runtime matrix — 2026-10-28

Published 2026-10-29. Completed all **310 configurations / 620 flow attempts** in the matching installed source: MmadL1 226, ND2NZ 60, Fixpipe 24. Sixteen parallel workers ran native first for each configuration, then the unchanged kernel through the A5 PTODSL bridge with `FALLBACK=1`.

| Flow | Exact PTODSL passes | Native/fallback passes | Failed attempts |
| --- | ---: | ---: | ---: |
| Native NPU-IR | — | 298 | 12 |
| Bridge | 148 | 13 fallback-only | 149 |

**149 distinct instance/configuration combinations fail in at least one flow; 161 failed flow attempts are listed below and in the failure CSV.** Twelve combinations fail in both flows. No numerical mismatches, skips, timeouts or native compilation failures occurred; these failures stop at selection or compilation before numerical comparison. Replacement passes require the exact selected PTODSL call, materialized PTO body and original numerical assertions. Unrelated dependencies may fall back.

## Published records

- [Every configuration, paired results, selected/called symbols and dependency fallback](../validation/template-runtime-matrix-2026-10-28/cases.csv) — 310 rows.
- [Every failing instance/configuration/flow, exact pytest IDs, scalar options, diagnostics, log/IR paths and reproduction commands](../validation/template-runtime-matrix-2026-10-28/failures.csv) — 161 rows.
- [Toolchain provenance and audit summary](../validation/template-runtime-matrix-2026-10-28/provenance.json).

Large raw logs and IR remain on the simulator server under `$HOME/tmp/template-matrix-native-bridge-20261028.is6of753/`. Paths beginning `$HOME` in the CSVs are server-local evidence, not files uploaded to Planner. The CSVs retain full configuration JSON; sequence cases have their original `unit_flag`, `rounds` and caller-loop options.

## Provenance and interpretation

- NPU-IR `3bf5d9898e0c45447c81780bc35de7548f5cef99`; Ascend LLVM `95c4693447af2473be62bdee19522cbe596f9e96`; PTOAS `5ce40f710f7f83b8d1fb2d8c36647bb811236a49` / CLI 0.69; PTO LLVM `fa2fd1f75d742fec91ecf40d98c9c7961647ec42`.
- Existing Triton 3.6 candidate, simulator `Ascend950PR_9589`; CANN 9.2 compiler children and CANN 9.1 beta runtime. All 620 attempts have actual compiler-environment traces. All 459 passing flow attempts have runtime-split evidence; all 542 original numerical assertions are present. All 494 protected source/tool/resource hashes match.
- INT8-TB MmadL1 has no installed PTODSL replacement: its 13 bridge passes are native fallback baselines. The dual F32-to-I8 Fixpipe target is also native-only and has no passing exact-selection baseline in this run.
- SET2D is absent from this selected matching source/install; its older private prefix was not mixed in. Eight installed exports have no dedicated parameterized runtime target; their exact names are in provenance.json. This is the existing simulator suite, not physical A5 or full CI.
- Compiler, templates, test kernels/configurations/references/tolerances, shared installation and source branches were preserved. Earlier passing records refer to their recorded revisions and are not overwritten by this revision-specific result.

## Failure groups and investigation handoff

| ID | Flow | First observed restriction | Attempts | Native counterpart passes |
| --- | --- | --- | ---: | ---: |
| F1 | ptoas | `bridge-pointer-cast-contract` | 53 | 53 |
| F2 | ptoas | `PTOAS-masked-store-alignment` | 48 | 48 |
| F3 | ptoas | `ND2NZ-registry-name-mismatch` | 36 | 36 |
| F4 | npuir | `exact-target-selection` | 12 | 0 |
| F5 | ptoas | `bridge-vfptosi-legalization` | 12 | 0 |

ND2NZ selection: the early registry gate requests `__pto_<native-name>` while installed typed helpers export different names. Native ND2NZ calls remain. INT8/FP8 conversion: `hivm.hir.pointer_cast` rejects a native Cube representation; saved rank-four I8 forms include a 32×32 inner tile. These failures occur before template numerical execution.

INT8 Fixpipe: native lowering chooses F32-to-F32 dual Fixpipe plus an AIV vector cast, leaving the requested F32-to-I8 identity absent. Bridge compilation separately fails `ave.hir.vfptosi` legalization. This is an exact-instance coverage gap plus a vector-conversion limitation, not a demonstrated numerical defect in an INT8 Fixpipe replacement.

PTOAS alignment: `pto.vmi.masked_store` requires proven destination alignment in UB vector consumers. This affects four sequence cases per F16/BF16/F32 MmadL1 instance, four INT8-TB sequence cases, and four F32 Fixpipe configurations. Source/IR findings and bounded investigation pointers are retained in the evidence root’s `investigation-findings.json`. Preserve semantic write footprints; no compiler/template fixes or environment changes were made.

Next work belongs to registry/AVE-to-VMI/PTOAS contract investigators: use the exact failing pytest node and saved first IR, compare with its native counterpart, and coordinate any matching compiler build before retesting. Do not change references, assertions, shared installations or deferred HOST/workspace integration as a side effect.

## Every failing instance/configuration combination

`pass-native` and `pass-replacement` retain their distinct meaning. Group IDs refer to the table above. Artifact ID `NNNN` resolves to `runtime/npuir/NNNN/run.log` and `runtime/ptoas/NNNN/run.log` beneath the evidence root; their `evidence/` directories retain commands, diagnostics and IR. The CSV supplies exact per-attempt paths and original pytest node IDs.

### Fixpipe / nz2nd-dual-f32-i8-ub

Native identity: `fixpipe_nz2nd_dual_float_to_int8_t_4d_to_2d_ubuf`. Expected PTODSL identity: `__pto_fixpipe_nz2nd_dual_float_to_int8_t_4d_to_2d_ubuf`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/native-b64` | 64/64/64 | tile 64/64/64 | selection-failure | bridge-conversion-failure | F4, F5 | `0000` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/small-b16` | 13/17/9 | tile 16/16/16 | selection-failure | bridge-conversion-failure | F4, F5 | `0001` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/small-b64` | 13/17/9 | tile 64/64/64 | selection-failure | bridge-conversion-failure | F4, F5 | `0002` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/boundary-b16-32-16` | 63/65/67 | tile 16/32/16 | selection-failure | bridge-conversion-failure | F4, F5 | `0003` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/boundary-b32-16-32` | 63/65/67 | tile 32/16/32 | selection-failure | bridge-conversion-failure | F4, F5 | `0004` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/boundary-b32-64-64` | 63/65/67 | tile 32/64/64 | selection-failure | bridge-conversion-failure | F4, F5 | `0005` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/large-b64` | 257/257/65 | tile 64/64/64 | selection-failure | bridge-conversion-failure | F4, F5 | `0006` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/single-row-b16-64-32` | 1/129/65 | tile 16/64/32 | selection-failure | bridge-conversion-failure | F4, F5 | `0007` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/single-col-b64-16-16` | 129/1/33 | tile 64/16/16 | selection-failure | bridge-conversion-failure | F4, F5 | `0008` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/reduction-tail-b64` | 65/63/129 | tile 64/64/64 | selection-failure | bridge-conversion-failure | F4, F5 | `0009` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/large-b64-128-128` | 257/257/65 | tile 64/128/128 | selection-failure | bridge-conversion-failure | F4, F5 | `0010` |
| `Fixpipe/nz2nd-dual-f32-i8-ub/shape/large-b128-64-64` | 257/257/65 | tile 128/64/64 | selection-failure | bridge-conversion-failure | F4, F5 | `0011` |

### Fixpipe / nz2nd-f32-f32-ub

Native identity: `fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf`. Expected PTODSL identity: `__pto_fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `Fixpipe/nz2nd-f32-f32-ub/shape/small-b16` | 13/17/9 | tile 16/16/16 | pass-native | ptoas-compilation-failure | F2 | `0013` |
| `Fixpipe/nz2nd-f32-f32-ub/shape/boundary-b16-32-16` | 63/65/67 | tile 16/32/16 | pass-native | ptoas-compilation-failure | F2 | `0015` |
| `Fixpipe/nz2nd-f32-f32-ub/shape/boundary-b32-16-32` | 63/65/67 | tile 32/16/32 | pass-native | ptoas-compilation-failure | F2 | `0016` |
| `Fixpipe/nz2nd-f32-f32-ub/shape/single-col-b64-16-16` | 129/1/33 | tile 64/16/16 | pass-native | ptoas-compilation-failure | F2 | `0020` |

### MmadL1 / bf16-nn

Native identity: `mma_tile_bfloat16_t_to_float`. Expected PTODSL identity: `__pto_mma_tile_bfloat16_t_to_float`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/bf16-nn/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0210` |
| `MmadL1/bf16-nn/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0211` |
| `MmadL1/bf16-nn/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0213` |
| `MmadL1/bf16-nn/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0214` |

### MmadL1 / bf16-ta

Native identity: `mma_tile_bfloat16_t_to_float_ta`. Expected PTODSL identity: `__pto_mma_tile_bfloat16_t_to_float_ta`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/bf16-ta/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0215` |
| `MmadL1/bf16-ta/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0216` |
| `MmadL1/bf16-ta/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0218` |
| `MmadL1/bf16-ta/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0219` |

### MmadL1 / bf16-ta-tb

Native identity: `mma_tile_bfloat16_t_to_float_ta_tb`. Expected PTODSL identity: `__pto_mma_tile_bfloat16_t_to_float_ta_tb`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/bf16-ta-tb/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0225` |
| `MmadL1/bf16-ta-tb/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0226` |
| `MmadL1/bf16-ta-tb/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0228` |
| `MmadL1/bf16-ta-tb/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0229` |

### MmadL1 / bf16-tb

Native identity: `mma_tile_bfloat16_t_to_float_tb`. Expected PTODSL identity: `__pto_mma_tile_bfloat16_t_to_float_tb`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/bf16-tb/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0220` |
| `MmadL1/bf16-tb/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0221` |
| `MmadL1/bf16-tb/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0223` |
| `MmadL1/bf16-tb/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0224` |

### MmadL1 / f16-nn

Native identity: `mma_tile_half_to_float`. Expected PTODSL identity: `__pto_mma_tile_half_to_float`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/f16-nn/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0190` |
| `MmadL1/f16-nn/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0191` |
| `MmadL1/f16-nn/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0193` |
| `MmadL1/f16-nn/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0194` |

### MmadL1 / f16-ta

Native identity: `mma_tile_half_to_float_ta`. Expected PTODSL identity: `__pto_mma_tile_half_to_float_ta`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/f16-ta/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0195` |
| `MmadL1/f16-ta/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0196` |
| `MmadL1/f16-ta/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0198` |
| `MmadL1/f16-ta/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0199` |

### MmadL1 / f16-ta-tb

Native identity: `mma_tile_half_to_float_ta_tb`. Expected PTODSL identity: `__pto_mma_tile_half_to_float_ta_tb`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/f16-ta-tb/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0205` |
| `MmadL1/f16-ta-tb/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0206` |
| `MmadL1/f16-ta-tb/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0208` |
| `MmadL1/f16-ta-tb/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0209` |

### MmadL1 / f16-tb

Native identity: `mma_tile_half_to_float_tb`. Expected PTODSL identity: `__pto_mma_tile_half_to_float_tb`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/f16-tb/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0200` |
| `MmadL1/f16-tb/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0201` |
| `MmadL1/f16-tb/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0203` |
| `MmadL1/f16-tb/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0204` |

### MmadL1 / f32-hf32-nn

Native identity: `mma_tile_float_to_float_hf32`. Expected PTODSL identity: `__pto_mma_tile_float_to_float_hf32`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/f32-hf32-nn/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0240` |
| `MmadL1/f32-hf32-nn/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0241` |
| `MmadL1/f32-hf32-nn/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0243` |
| `MmadL1/f32-hf32-nn/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0244` |

### MmadL1 / f32-ieee-nn

Native identity: `mma_tile_float_to_float`. Expected PTODSL identity: `__pto_mma_tile_float_to_float`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/f32-ieee-nn/sequence/enabled` | 16/16/16 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0245` |
| `MmadL1/f32-ieee-nn/sequence/disabled` | 16/16/16 | block_k=16, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0246` |
| `MmadL1/f32-ieee-nn/sequence/caller-k` | 16/16/48 | block_k=16, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0248` |
| `MmadL1/f32-ieee-nn/sequence/caller-k-once` | 16/16/48 | block_k=16, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0249` |

### MmadL1 / f8e4m3-nn

Native identity: `mma_tile_float8_e4m3_t_to_float`. Expected PTODSL identity: `__pto_mma_tile_float8_e4m3_t_to_float`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/f8e4m3-nn/shape/small-b16` | 13/17/9 | tile 16/16/16 | pass-native | bridge-conversion-failure | F1 | `0178` |
| `MmadL1/f8e4m3-nn/shape/small-b64` | 13/17/9 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0179` |
| `MmadL1/f8e4m3-nn/shape/boundary-b16-32-16` | 63/65/67 | tile 16/32/16 | pass-native | bridge-conversion-failure | F1 | `0180` |
| `MmadL1/f8e4m3-nn/shape/boundary-b32-16-32` | 63/65/67 | tile 32/16/32 | pass-native | bridge-conversion-failure | F1 | `0181` |
| `MmadL1/f8e4m3-nn/shape/boundary-b32-64-64` | 63/65/67 | tile 32/64/64 | pass-native | bridge-conversion-failure | F1 | `0182` |
| `MmadL1/f8e4m3-nn/shape/native-b64` | 64/64/64 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0183` |
| `MmadL1/f8e4m3-nn/shape/large-b64` | 257/257/257 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0184` |
| `MmadL1/f8e4m3-nn/shape/large-b128` | 257/257/257 | tile 128/128/128 | pass-native | bridge-conversion-failure | F1 | `0185` |
| `MmadL1/f8e4m3-nn/shape/large-b128-256-128` | 257/257/257 | tile 128/256/128 | pass-native | bridge-conversion-failure | F1 | `0186` |
| `MmadL1/f8e4m3-nn/shape/large-b256-128-64` | 257/257/257 | tile 256/128/64 | pass-native | bridge-conversion-failure | F1 | `0187` |
| `MmadL1/f8e4m3-nn/shape/large-b64-128-128` | 257/257/257 | tile 64/128/128 | pass-native | bridge-conversion-failure | F1 | `0188` |
| `MmadL1/f8e4m3-nn/shape/large-b128-64-64` | 257/257/257 | tile 128/64/64 | pass-native | bridge-conversion-failure | F1 | `0189` |

### MmadL1 / i8-nn

Native identity: `mma_tile_int8_t_to_int32_t`. Expected PTODSL identity: `__pto_mma_tile_int8_t_to_int32_t`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/i8-nn/shape/small-b16` | 13/17/9 | tile 16/16/16 | pass-native | bridge-conversion-failure | F1 | `0130` |
| `MmadL1/i8-nn/shape/small-b64` | 13/17/9 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0131` |
| `MmadL1/i8-nn/shape/boundary-b16-32-16` | 63/65/67 | tile 16/32/16 | pass-native | bridge-conversion-failure | F1 | `0132` |
| `MmadL1/i8-nn/shape/boundary-b32-16-32` | 63/65/67 | tile 32/16/32 | pass-native | bridge-conversion-failure | F1 | `0133` |
| `MmadL1/i8-nn/shape/boundary-b32-64-64` | 63/65/67 | tile 32/64/64 | pass-native | bridge-conversion-failure | F1 | `0134` |
| `MmadL1/i8-nn/shape/native-b64` | 64/64/64 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0135` |
| `MmadL1/i8-nn/shape/large-b64` | 257/257/257 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0136` |
| `MmadL1/i8-nn/shape/large-b128` | 257/257/257 | tile 128/128/128 | pass-native | bridge-conversion-failure | F1 | `0137` |
| `MmadL1/i8-nn/shape/large-b128-256-128` | 257/257/257 | tile 128/256/128 | pass-native | bridge-conversion-failure | F1 | `0138` |
| `MmadL1/i8-nn/shape/large-b256-128-64` | 257/257/257 | tile 256/128/64 | pass-native | bridge-conversion-failure | F1 | `0139` |
| `MmadL1/i8-nn/shape/large-b64-128-128` | 257/257/257 | tile 64/128/128 | pass-native | bridge-conversion-failure | F1 | `0140` |
| `MmadL1/i8-nn/shape/large-b128-64-64` | 257/257/257 | tile 128/64/64 | pass-native | bridge-conversion-failure | F1 | `0141` |
| `MmadL1/i8-nn/sequence/enabled` | 16/16/32 | block_k=32, unit_flag=true, rounds=3 | pass-native | bridge-conversion-failure | F1 | `0230` |
| `MmadL1/i8-nn/sequence/disabled` | 16/16/32 | block_k=32, unit_flag=false, rounds=3 | pass-native | bridge-conversion-failure | F1 | `0231` |
| `MmadL1/i8-nn/sequence/split-k` | 64/64/1024 | block_k=1024, unit_flag=true, rounds=3 | pass-native | bridge-conversion-failure | F1 | `0232` |
| `MmadL1/i8-nn/sequence/caller-k` | 16/16/96 | block_k=32, unit_flag=true, rounds=3 | pass-native | bridge-conversion-failure | F1 | `0233` |
| `MmadL1/i8-nn/sequence/caller-k-once` | 16/16/96 | block_k=32, unit_flag=true, rounds=1 | pass-native | bridge-conversion-failure | F1 | `0234` |

### MmadL1 / i8-tb

Native identity: `mma_tile_int8_t_to_int32_t_tb`. Expected PTODSL identity: `__pto_mma_tile_int8_t_to_int32_t_tb`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `MmadL1/i8-tb/sequence/enabled` | 16/16/32 | block_k=32, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0235` |
| `MmadL1/i8-tb/sequence/disabled` | 16/16/32 | block_k=32, unit_flag=false, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0236` |
| `MmadL1/i8-tb/sequence/caller-k` | 16/16/96 | block_k=32, unit_flag=true, rounds=3 | pass-native | ptoas-compilation-failure | F2 | `0238` |
| `MmadL1/i8-tb/sequence/caller-k-once` | 16/16/96 | block_k=32, unit_flag=true, rounds=1 | pass-native | ptoas-compilation-failure | F2 | `0239` |

### Nd2Nz / bf16

Native identity: `nd2nz_bfloat16_t`. Expected PTODSL identity: `__pto_nd2nz_bf16_gm_l1`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `Nd2Nz/bf16/shape/small-b16` | 13/17/9 | tile 16/16/16 | pass-native | selection-failure | F3 | `0262` |
| `Nd2Nz/bf16/shape/small-b64` | 13/17/9 | tile 64/64/64 | pass-native | selection-failure | F3 | `0263` |
| `Nd2Nz/bf16/shape/boundary-b16-32-16` | 63/65/67 | tile 16/32/16 | pass-native | selection-failure | F3 | `0264` |
| `Nd2Nz/bf16/shape/boundary-b32-16-32` | 63/65/67 | tile 32/16/32 | pass-native | selection-failure | F3 | `0265` |
| `Nd2Nz/bf16/shape/boundary-b32-64-64` | 63/65/67 | tile 32/64/64 | pass-native | selection-failure | F3 | `0266` |
| `Nd2Nz/bf16/shape/native-b64` | 64/64/64 | tile 64/64/64 | pass-native | selection-failure | F3 | `0267` |
| `Nd2Nz/bf16/shape/large-b64` | 257/257/257 | tile 64/64/64 | pass-native | selection-failure | F3 | `0268` |
| `Nd2Nz/bf16/shape/large-b128` | 257/257/257 | tile 128/128/128 | pass-native | selection-failure | F3 | `0269` |
| `Nd2Nz/bf16/shape/large-b128-256-128` | 257/257/257 | tile 128/256/128 | pass-native | selection-failure | F3 | `0270` |
| `Nd2Nz/bf16/shape/large-b256-128-64` | 257/257/257 | tile 256/128/64 | pass-native | selection-failure | F3 | `0271` |
| `Nd2Nz/bf16/shape/large-b64-128-128` | 257/257/257 | tile 64/128/128 | pass-native | selection-failure | F3 | `0272` |
| `Nd2Nz/bf16/shape/large-b128-64-64` | 257/257/257 | tile 128/64/64 | pass-native | selection-failure | F3 | `0273` |

### Nd2Nz / f16

Native identity: `nd2nz_half`. Expected PTODSL identity: `__pto_nd2nz_f16_gm_l1`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `Nd2Nz/f16/shape/small-b16` | 13/17/9 | tile 16/16/16 | pass-native | selection-failure | F3 | `0250` |
| `Nd2Nz/f16/shape/small-b64` | 13/17/9 | tile 64/64/64 | pass-native | selection-failure | F3 | `0251` |
| `Nd2Nz/f16/shape/boundary-b16-32-16` | 63/65/67 | tile 16/32/16 | pass-native | selection-failure | F3 | `0252` |
| `Nd2Nz/f16/shape/boundary-b32-16-32` | 63/65/67 | tile 32/16/32 | pass-native | selection-failure | F3 | `0253` |
| `Nd2Nz/f16/shape/boundary-b32-64-64` | 63/65/67 | tile 32/64/64 | pass-native | selection-failure | F3 | `0254` |
| `Nd2Nz/f16/shape/native-b64` | 64/64/64 | tile 64/64/64 | pass-native | selection-failure | F3 | `0255` |
| `Nd2Nz/f16/shape/large-b64` | 257/257/257 | tile 64/64/64 | pass-native | selection-failure | F3 | `0256` |
| `Nd2Nz/f16/shape/large-b128` | 257/257/257 | tile 128/128/128 | pass-native | selection-failure | F3 | `0257` |
| `Nd2Nz/f16/shape/large-b128-256-128` | 257/257/257 | tile 128/256/128 | pass-native | selection-failure | F3 | `0258` |
| `Nd2Nz/f16/shape/large-b256-128-64` | 257/257/257 | tile 256/128/64 | pass-native | selection-failure | F3 | `0259` |
| `Nd2Nz/f16/shape/large-b64-128-128` | 257/257/257 | tile 64/128/128 | pass-native | selection-failure | F3 | `0260` |
| `Nd2Nz/f16/shape/large-b128-64-64` | 257/257/257 | tile 128/64/64 | pass-native | selection-failure | F3 | `0261` |

### Nd2Nz / f32

Native identity: `nd2nz_float`. Expected PTODSL identity: `__pto_nd2nz_f32_gm_l1`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `Nd2Nz/f32/shape/small-b16` | 13/17/9 | tile 16/16/16 | pass-native | selection-failure | F3 | `0286` |
| `Nd2Nz/f32/shape/small-b64` | 13/17/9 | tile 64/64/64 | pass-native | selection-failure | F3 | `0287` |
| `Nd2Nz/f32/shape/boundary-b16-32-16` | 63/65/67 | tile 16/32/16 | pass-native | selection-failure | F3 | `0288` |
| `Nd2Nz/f32/shape/boundary-b32-16-32` | 63/65/67 | tile 32/16/32 | pass-native | selection-failure | F3 | `0289` |
| `Nd2Nz/f32/shape/boundary-b32-64-64` | 63/65/67 | tile 32/64/64 | pass-native | selection-failure | F3 | `0290` |
| `Nd2Nz/f32/shape/native-b64` | 64/64/64 | tile 64/64/64 | pass-native | selection-failure | F3 | `0291` |
| `Nd2Nz/f32/shape/large-b64` | 257/257/257 | tile 64/64/64 | pass-native | selection-failure | F3 | `0292` |
| `Nd2Nz/f32/shape/large-b128` | 257/257/257 | tile 128/128/128 | pass-native | selection-failure | F3 | `0293` |
| `Nd2Nz/f32/shape/large-b128-256-128` | 257/257/257 | tile 128/256/128 | pass-native | selection-failure | F3 | `0294` |
| `Nd2Nz/f32/shape/large-b256-128-64` | 257/257/257 | tile 256/128/64 | pass-native | selection-failure | F3 | `0295` |
| `Nd2Nz/f32/shape/large-b64-128-128` | 257/257/257 | tile 64/128/128 | pass-native | selection-failure | F3 | `0296` |
| `Nd2Nz/f32/shape/large-b128-64-64` | 257/257/257 | tile 128/64/64 | pass-native | selection-failure | F3 | `0297` |

### Nd2Nz / f8e4m3

Native identity: `nd2nz_float8_e4m3_t`. Expected PTODSL identity: `__pto_nd2nz_float8_e4m3_t_gm_l1`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `Nd2Nz/f8e4m3/shape/small-b16` | 13/17/9 | tile 16/16/16 | pass-native | bridge-conversion-failure | F1 | `0298` |
| `Nd2Nz/f8e4m3/shape/small-b64` | 13/17/9 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0299` |
| `Nd2Nz/f8e4m3/shape/boundary-b16-32-16` | 63/65/67 | tile 16/32/16 | pass-native | bridge-conversion-failure | F1 | `0300` |
| `Nd2Nz/f8e4m3/shape/boundary-b32-16-32` | 63/65/67 | tile 32/16/32 | pass-native | bridge-conversion-failure | F1 | `0301` |
| `Nd2Nz/f8e4m3/shape/boundary-b32-64-64` | 63/65/67 | tile 32/64/64 | pass-native | bridge-conversion-failure | F1 | `0302` |
| `Nd2Nz/f8e4m3/shape/native-b64` | 64/64/64 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0303` |
| `Nd2Nz/f8e4m3/shape/large-b64` | 257/257/257 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0304` |
| `Nd2Nz/f8e4m3/shape/large-b128` | 257/257/257 | tile 128/128/128 | pass-native | bridge-conversion-failure | F1 | `0305` |
| `Nd2Nz/f8e4m3/shape/large-b128-256-128` | 257/257/257 | tile 128/256/128 | pass-native | bridge-conversion-failure | F1 | `0306` |
| `Nd2Nz/f8e4m3/shape/large-b256-128-64` | 257/257/257 | tile 256/128/64 | pass-native | bridge-conversion-failure | F1 | `0307` |
| `Nd2Nz/f8e4m3/shape/large-b64-128-128` | 257/257/257 | tile 64/128/128 | pass-native | bridge-conversion-failure | F1 | `0308` |
| `Nd2Nz/f8e4m3/shape/large-b128-64-64` | 257/257/257 | tile 128/64/64 | pass-native | bridge-conversion-failure | F1 | `0309` |

### Nd2Nz / i8

Native identity: `nd2nz_int8_t`. Expected PTODSL identity: `__pto_nd2nz_i8_gm_l1`.

| Complete combination name | M/N/K | Tile or sequence options | Native result | Bridge result | Groups | Artifact ID |
| --- | --- | --- | --- | --- | --- | --- |
| `Nd2Nz/i8/shape/small-b16` | 13/17/9 | tile 16/16/16 | pass-native | bridge-conversion-failure | F1 | `0274` |
| `Nd2Nz/i8/shape/small-b64` | 13/17/9 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0275` |
| `Nd2Nz/i8/shape/boundary-b16-32-16` | 63/65/67 | tile 16/32/16 | pass-native | bridge-conversion-failure | F1 | `0276` |
| `Nd2Nz/i8/shape/boundary-b32-16-32` | 63/65/67 | tile 32/16/32 | pass-native | bridge-conversion-failure | F1 | `0277` |
| `Nd2Nz/i8/shape/boundary-b32-64-64` | 63/65/67 | tile 32/64/64 | pass-native | bridge-conversion-failure | F1 | `0278` |
| `Nd2Nz/i8/shape/native-b64` | 64/64/64 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0279` |
| `Nd2Nz/i8/shape/large-b64` | 257/257/257 | tile 64/64/64 | pass-native | bridge-conversion-failure | F1 | `0280` |
| `Nd2Nz/i8/shape/large-b128` | 257/257/257 | tile 128/128/128 | pass-native | bridge-conversion-failure | F1 | `0281` |
| `Nd2Nz/i8/shape/large-b128-256-128` | 257/257/257 | tile 128/256/128 | pass-native | bridge-conversion-failure | F1 | `0282` |
| `Nd2Nz/i8/shape/large-b256-128-64` | 257/257/257 | tile 256/128/64 | pass-native | bridge-conversion-failure | F1 | `0283` |
| `Nd2Nz/i8/shape/large-b64-128-128` | 257/257/257 | tile 64/128/128 | pass-native | bridge-conversion-failure | F1 | `0284` |
| `Nd2Nz/i8/shape/large-b128-64-64` | 257/257/257 | tile 128/64/64 | pass-native | bridge-conversion-failure | F1 | `0285` |

## Reproduce one recorded failure

On the original simulator server, after checking that its tools/source still match the provenance record:

```bash
source "$HOME/.venv/npuir-sim-3.6-candidate/bin/activate"
python "$HOME/tmp/template-matrix-native-bridge-20261028.is6of753/reproduce.py" 0 ptoas
```

Replace `0` with the catalog/artifact ID (decimal) and `ptoas` with `npuir` for its native counterpart. The helper refuses changed source/tools and writes a fresh private reproduction directory. No new tests were run to publish this record.
