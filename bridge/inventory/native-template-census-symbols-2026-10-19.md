# Exact native template requirements observed through 2026-10-19

This is the 191-symbol union from two completed, disjoint 10,000-configuration
native A5 samples. See the [campaign report](native-template-census-2026-10-19.md) for scope, provenance,
failures and extrapolation; the [CSV](native-template-census-symbols-2026-10-19.csv) carries per-campaign counts
and one source testcase/compile identifier per symbol.

**Required so far** means a dependency demanded by observed lowered IR.
Compatibility needs an implementation or a documented fallback/dependency
strategy for these calls. This is not a claim that 191 PTODSL implementations
exist, every native definition links, or any numerical test passed. Controls
and descriptors within one symbol can still require additional work.

Names strip only `_mlir_ciface_`; suffixes such as `.cube`, `.vector`, `_ta`,
`_tb`, datatype and rank remain significant. Declarations alone do not count.
A/B denote observation in the first/second campaign. Static calls deduplicate
shared native compilation IDs globally; file counts deduplicate suite/file
addresses, independent of the number of functions/configurations.

## MmadL1 — 16 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `mma_tile_bfloat16_t_to_float` | A+B | 787 | 215 | yes |
| `mma_tile_bfloat16_t_to_float_ta` | A+B | 30 | 9 | yes |
| `mma_tile_bfloat16_t_to_float_ta_tb` | A+B | 9 | 5 | yes |
| `mma_tile_bfloat16_t_to_float_tb` | A | 8 | 5 | yes |
| `mma_tile_float_to_float` | A+B | 1678 | 262 | yes |
| `mma_tile_float_to_float_ta` | A+B | 30 | 12 | yes |
| `mma_tile_float_to_float_ta_tb` | A+B | 15 | 6 | yes |
| `mma_tile_float_to_float_tb` | A+B | 10 | 6 | yes |
| `mma_tile_half_to_float` | A+B | 609 | 225 | yes |
| `mma_tile_half_to_float_ta` | A+B | 13 | 6 | yes |
| `mma_tile_half_to_float_ta_tb` | A+B | 7 | 3 | yes |
| `mma_tile_half_to_float_tb` | A+B | 4 | 3 | yes |
| `mma_tile_half_to_half` | A | 1 | 1 | no |
| `mma_tile_int8_t_to_int32_t` | A+B | 627 | 211 | yes |
| `mma_tile_int8_t_to_int32_t_ta` | A+B | 14 | 11 | yes |
| `mma_tile_int8_t_to_int32_t_tb` | B | 5 | 3 | yes |

## Fixpipe — 8 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `fixpipe_normal_float_to_float_4d_to_4d_cbuf` | A+B | 59 | 5 | yes |
| `fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf` | A+B | 41 | 17 | yes |
| `fixpipe_nz2nd_dual_int32_t_to_int32_t_4d_to_2d_ubuf` | A+B | 13 | 9 | yes |
| `fixpipe_nz2nd_float_to_float_4d_to_2d_gm` | A+B | 658 | 176 | yes |
| `fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf` | A+B | 2559 | 283 | yes |
| `fixpipe_nz2nd_half_to_half_4d_to_2d_gm` | A | 1 | 1 | no |
| `fixpipe_nz2nd_int32_t_to_int32_t_4d_to_2d_gm` | A+B | 158 | 86 | yes |
| `fixpipe_nz2nd_int32_t_to_int32_t_4d_to_2d_ubuf` | A+B | 485 | 174 | yes |

## Nd2Nz — 4 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `nd2nz_bfloat16_t` | A+B | 1590 | 226 | yes |
| `nd2nz_float` | A+B | 2449 | 269 | yes |
| `nd2nz_half` | A+B | 1212 | 232 | yes |
| `nd2nz_int8_t` | A+B | 1254 | 220 | yes |

## Cube setup — 4 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `set_l1_2d_bfloat16_t` | A+B | 273 | 49 | yes |
| `set_l1_2d_float` | A+B | 192 | 49 | yes |
| `set_l1_2d_half` | A+B | 194 | 45 | yes |
| `set_l1_2d_int8_t` | A+B | 252 | 47 | yes |

## Cube copy — 1 instance

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `load_gm_to_cbuf_1d_float` | A | 6 | 1 | yes |

## Vector DMA — 52 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `copy_ubuf_to_cbuf_1d_bfloat16_t` | A+B | 30 | 10 | yes |
| `copy_ubuf_to_cbuf_1d_float` | A+B | 632 | 54 | yes |
| `copy_ubuf_to_cbuf_1d_half` | A+B | 21 | 8 | yes |
| `copy_ubuf_to_cbuf_1d_int8_t` | A+B | 9 | 3 | yes |
| `copy_ubuf_to_cbuf_2d_float` | A | 4 | 1 | yes |
| `copy_ubuf_to_ubuf_1d_bfloat16_t` | A+B | 13 | 3 | yes |
| `copy_ubuf_to_ubuf_1d_float` | A+B | 767 | 78 | yes |
| `copy_ubuf_to_ubuf_1d_half` | A+B | 11 | 6 | yes |
| `copy_ubuf_to_ubuf_1d_int16_t` | A+B | 9 | 2 | yes |
| `copy_ubuf_to_ubuf_1d_int32_t` | A+B | 79 | 19 | yes |
| `copy_ubuf_to_ubuf_1d_int64_t` | A+B | 45 | 17 | yes |
| `copy_ubuf_to_ubuf_1d_int8_t` | A+B | 29 | 6 | yes |
| `copy_ubuf_to_ubuf_2d_bfloat16_t` | A+B | 12 | 5 | yes |
| `copy_ubuf_to_ubuf_2d_bool` | A+B | 10 | 2 | yes |
| `copy_ubuf_to_ubuf_2d_float` | A+B | 757 | 59 | yes |
| `copy_ubuf_to_ubuf_2d_half` | A+B | 7 | 5 | yes |
| `copy_ubuf_to_ubuf_2d_int32_t` | A+B | 9 | 4 | yes |
| `copy_ubuf_to_ubuf_2d_int8_t` | A+B | 5 | 2 | yes |
| `load_gm_to_ubuf_1d_bfloat16_t` | A+B | 595 | 123 | yes |
| `load_gm_to_ubuf_1d_float` | A+B | 3950 | 401 | yes |
| `load_gm_to_ubuf_1d_half` | A+B | 641 | 157 | yes |
| `load_gm_to_ubuf_1d_int16_t` | A+B | 312 | 106 | yes |
| `load_gm_to_ubuf_1d_int32_t` | A+B | 986 | 258 | yes |
| `load_gm_to_ubuf_1d_int64_t` | A+B | 552 | 172 | yes |
| `load_gm_to_ubuf_1d_int8_t` | A+B | 2168 | 303 | yes |
| `load_gm_to_ubuf_2d_bfloat16_t` | A+B | 203 | 13 | yes |
| `load_gm_to_ubuf_2d_float` | A+B | 235 | 20 | yes |
| `load_gm_to_ubuf_2d_half` | A+B | 95 | 9 | yes |
| `load_gm_to_ubuf_2d_int16_t` | B | 1 | 1 | yes |
| `load_gm_to_ubuf_2d_int32_t` | A+B | 7 | 5 | yes |
| `load_gm_to_ubuf_2d_int64_t` | A+B | 2 | 2 | yes |
| `load_gm_to_ubuf_2d_int8_t` | A+B | 57 | 5 | yes |
| `load_gm_to_ubuf_3d_int32_t` | B | 1 | 1 | yes |
| `load_gm_to_ubuf_3d_int64_t` | B | 1 | 1 | yes |
| `store_ubuf_to_gm_1d_bfloat16_t` | A+B | 329 | 123 | yes |
| `store_ubuf_to_gm_1d_float` | A+B | 3030 | 431 | yes |
| `store_ubuf_to_gm_1d_half` | A+B | 470 | 159 | yes |
| `store_ubuf_to_gm_1d_int16_t` | A+B | 187 | 109 | yes |
| `store_ubuf_to_gm_1d_int32_t` | A+B | 1052 | 296 | yes |
| `store_ubuf_to_gm_1d_int64_t` | A+B | 329 | 172 | yes |
| `store_ubuf_to_gm_1d_int8_t` | A+B | 775 | 218 | yes |
| `store_ubuf_to_gm_2d_bfloat16_t` | A+B | 150 | 12 | yes |
| `store_ubuf_to_gm_2d_float` | A+B | 658 | 80 | yes |
| `store_ubuf_to_gm_2d_half` | A+B | 83 | 13 | yes |
| `store_ubuf_to_gm_2d_int16_t` | B | 2 | 2 | yes |
| `store_ubuf_to_gm_2d_int32_t` | A+B | 137 | 43 | yes |
| `store_ubuf_to_gm_2d_int64_t` | A+B | 4 | 4 | yes |
| `store_ubuf_to_gm_2d_int8_t` | A+B | 38 | 6 | yes |
| `store_ubuf_to_gm_3d_float` | A+B | 8 | 1 | yes |
| `store_ubuf_to_gm_3d_int32_t` | B | 1 | 1 | yes |
| `store_ubuf_to_gm_3d_int64_t` | B | 1 | 1 | yes |
| `store_ubuf_to_gm_3d_int8_t` | B | 2 | 1 | yes |

## Vector Integer64 — 46 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `cast_float_to_int64_t` | A+B | 16 | 7 | yes |
| `cast_int32_t_to_int64_t` | A+B | 5 | 2 | yes |
| `cast_int64_t_to_float` | A+B | 9 | 4 | yes |
| `cast_int64_t_to_int32_t` | A+B | 7 | 3 | yes |
| `cast_uint32_t_to_int64_t` | B | 2 | 1 | yes |
| `masked_store_NORM_B64_int64_t_rank1` | A+B | 232 | 124 | yes |
| `masked_store_NORM_B64_int64_t_unalign_rank1` | A+B | 18 | 7 | yes |
| `masked_store_ONEPT_B64_int64_t_rank0` | A+B | 21 | 13 | yes |
| `masked_store_ONEPT_B64_int64_t_rank1` | A+B | 38 | 30 | yes |
| `vabs_int64_t` | A+B | 12 | 10 | yes |
| `vadd_int64_t` | A+B | 112 | 59 | yes |
| `vadds_int64_t` | A+B | 5 | 3 | yes |
| `vand_int64_t` | A+B | 20 | 6 | yes |
| `vbr_int64_t` | A+B | 4 | 3 | yes |
| `vcadd_int64_t` | A | 4 | 1 | yes |
| `vci_int64_t` | B | 4 | 2 | yes |
| `vcmax_int64_t` | A | 1 | 1 | yes |
| `vcmin_int64_t` | B | 1 | 1 | yes |
| `vcmp_eq_int64_t` | A+B | 4 | 2 | yes |
| `vcmp_ge_int64_t` | A | 1 | 1 | yes |
| `vcmp_gt_int64_t` | A+B | 4 | 1 | yes |
| `vcmp_le_int64_t` | B | 1 | 1 | yes |
| `vcmp_lt_int64_t` | A | 1 | 1 | yes |
| `vcmp_ne_int64_t` | A | 2 | 1 | yes |
| `vcmp_ugt_uint64_t` | A | 1 | 1 | yes |
| `vdintlv_int64_t` | B | 1 | 1 | yes |
| `vdiv_int64_t` | A | 1 | 1 | yes |
| `vdups_int64_t` | A+B | 72 | 34 | yes |
| `vgather_int64_t_rank2` | B | 4 | 2 | yes |
| `vintlv_int64_t` | A+B | 92 | 12 | yes |
| `vload_BRC_B64_int64_t_rank0` | A+B | 13 | 7 | yes |
| `vload_BRC_B64_int64_t_rank1` | A+B | 85 | 38 | yes |
| `vload_BRC_B64_int64_t_rank3` | B | 1 | 1 | yes |
| `vload_NORM_int64_t_rank1` | A+B | 372 | 122 | yes |
| `vload_NORM_int64_t_unalign_rank1` | A+B | 20 | 12 | yes |
| `vload_NORM_int64_t_unalign_rank2` | A | 3 | 3 | yes |
| `vmax_int64_t` | A+B | 7 | 4 | yes |
| `vmin_int64_t` | A+B | 7 | 4 | yes |
| `vmul_int64_t` | A+B | 7 | 4 | yes |
| `vmuls_int64_t` | A+B | 17 | 10 | yes |
| `vor_int64_t` | A+B | 20 | 13 | yes |
| `vsel_int64_t` | A+B | 55 | 26 | yes |
| `vshls_int64_t` | B | 1 | 1 | yes |
| `vshrs_int64_t` | B | 2 | 1 | yes |
| `vsub_int64_t` | A+B | 16 | 11 | yes |
| `vxor_int64_t` | A+B | 115 | 18 | yes |

## Vector collective — 9 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `cumprod_2d_float_dim0` | B | 2 | 1 | yes |
| `cumsum_1d_float_dim0` | A | 17 | 3 | yes |
| `cumsum_1d_int32_t_dim0` | A+B | 2 | 2 | yes |
| `cumsum_2d_float_dim0` | A+B | 12 | 2 | yes |
| `cumsum_2d_int32_t_dim0` | A+B | 10 | 3 | yes |
| `cumsum_2d_int8_t_dim0` | B | 1 | 1 | yes |
| `flip_1d_int32_t` | B | 2 | 1 | yes |
| `flip_1d_int8_t` | A+B | 10 | 2 | yes |
| `sort_2d_float` | A | 1 | 1 | yes |

## Vector math — 3 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `vdiv_int32_t` | A+B | 6 | 1 | yes |
| `vdivfhp_float` | A+B | 13 | 1 | yes |
| `vmod_int32_t` | A+B | 6 | 1 | yes |

## SIMT indexing — 1 instance

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `gather_simt_2d_bfloat16_t_int32_t` | B | 1 | 1 | yes |

## Support: printing — 25 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `print_1d_bfloat16_t_ubuf` | B | 3 | 1 | yes |
| `print_1d_float_ubuf` | A+B | 15 | 4 | yes |
| `print_1d_half_ubuf` | A | 1 | 1 | yes |
| `print_1d_int16_t_ubuf` | B | 1 | 1 | yes |
| `print_1d_int32_t_ubuf` | A+B | 6 | 3 | yes |
| `print_1d_int64_t_ubuf` | A+B | 4 | 4 | yes |
| `print_1d_int8_t_ubuf` | A+B | 33 | 5 | yes |
| `print_2d_bfloat16_t_ubuf.vector` | A+B | 7 | 2 | yes |
| `print_2d_float_gm.cube` | A+B | 28 | 2 | yes |
| `print_2d_float_ubuf.vector` | A+B | 36 | 3 | yes |
| `print_2d_int16_t_ubuf` | B | 3 | 2 | yes |
| `print_2d_int32_t_gm.cube` | A | 2 | 1 | yes |
| `print_2d_int32_t_ubuf` | B | 2 | 1 | yes |
| `print_2d_int32_t_ubuf.vector` | B | 2 | 1 | yes |
| `print_2d_int64_t_ubuf` | A+B | 3 | 3 | yes |
| `print_2d_int8_t_ubuf` | B | 3 | 2 | yes |
| `print_2d_int8_t_ubuf.vector` | A+B | 4 | 1 | yes |
| `print_3d_float_ubuf` | A+B | 18 | 1 | yes |
| `print_3d_int32_t_ubuf` | A+B | 15 | 2 | yes |
| `print_3d_int64_t_ubuf` | A | 2 | 2 | yes |
| `print_3d_int8_t_ubuf` | A+B | 10 | 4 | yes |
| `print_5d_int32_t_ubuf` | B | 1 | 1 | yes |
| `print_5d_int64_t_ubuf` | B | 1 | 1 | yes |
| `print_scalar_bool_gm` | A+B | 7 | 1 | yes |
| `print_scalar_int32_t_gm` | A+B | 13 | 3 | yes |

## Support: cube debug — 4 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `finish_debug` | A+B | 141 | 10 | yes |
| `finish_debug.cube` | A+B | 30 | 2 | yes |
| `init_debug` | A+B | 43 | 10 | yes |
| `init_debug.cube` | A+B | 12 | 2 | yes |

## Support: vector debug — 2 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `finish_debug.vector` | A+B | 49 | 4 | yes |
| `init_debug.vector` | A+B | 17 | 4 | yes |

## Support: synchronization — 2 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `sync_block_lock` | A+B | 223 | 39 | yes |
| `sync_block_unlock` | A+B | 223 | 39 | yes |

## Other uncatalogued helpers — 14 instances

| Exact native symbol | Campaigns | Static calls | Test files | Catalogued |
|---|---|---:|---:|---|
| `broadcast_scalar_bfloat16_t_to_1d` | A+B | 12 | 4 | no |
| `broadcast_scalar_float_to_1d` | A+B | 483 | 59 | no |
| `broadcast_scalar_float_to_2d` | A+B | 89 | 14 | no |
| `broadcast_scalar_half_to_1d` | A+B | 9 | 5 | no |
| `broadcast_scalar_int32_t_to_1d` | A+B | 8 | 6 | no |
| `broadcast_scalar_int8_t_to_1d` | A+B | 5 | 2 | no |
| `transpose_3d_without_last_axis_bfloat16_t` | A+B | 30 | 10 | no |
| `transpose_3d_without_last_axis_float` | A+B | 636 | 54 | no |
| `transpose_3d_without_last_axis_half` | A+B | 21 | 8 | no |
| `transpose_3d_without_last_axis_int8_t` | A+B | 9 | 3 | no |
| `vadd_1d_float` | A+B | 199 | 43 | no |
| `vadd_1d_int32_t` | B | 9 | 6 | no |
| `vadd_2d_float` | A+B | 108 | 18 | no |
| `vadd_3d_float` | A+B | 10 | 1 | no |
