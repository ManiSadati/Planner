// -----// IR Dump After ConvertHIVMTemplatesToPTO (convert-hivm-templates-to-pto) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIC>} {
  func.func @matmul64_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c256_i64 = arith.constant 256 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %cst = arith.constant 1.000000e+00 : f16
    %c262144_i64 = arith.constant 262144 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca[%c0] : memref<1xi64>
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %1 = hivm.hir.pointer_cast(%c16384_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    %3 = hivm.hir.pointer_cast(%c16384_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    %4 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c24576_i64) : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
    %6 = arith.muli %arg5, %arg6 : i32
    %7 = arith.muli %6, %arg7 : i32
    %8 = hivm.hir.get_block_idx -> i64
    %9 = arith.trunci %8 : i64 to i32
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    scf.for %arg8 = %9 to %7 step %c32_i32  : i32 {
      %10 = memref.load %alloca[%c0] {markDCacheInvalidatePatternVisited = 0 : i32} : memref<1xi64>
      %11 = arith.remui %10, %c2_i64 : i64
      %12 = arith.index_cast %11 : i64 to index
      %13 = arith.cmpi eq, %11, %c1_i64 : i64
      %14 = arith.select %13, %1, %0 : memref<4x4x16x16xf32, #hivm.address_space<cc>>
      %15 = arith.select %13, %3, %2 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      %16 = arith.select %13, %5, %4 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>
      %17 = arith.index_cast %12 : index to i1
      %18 = arith.select %17, %c2_i64, %c3_i64 : i64
      %19 = arith.xori %17, %true : i1
      %20 = arith.extui %19 : i1 to i64
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %21 = arith.remsi %arg8, %arg5 : i32
      %22 = arith.cmpi ne, %21, %c0_i32 : i32
      scf.if %22 {
      } else {
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %20]
        %memspacecast = memref.memory_space_cast %reinterpret_cast : memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1]>, #pto.address_space<gm>>
        %24 = pto.tile_buf_addr %memspacecast : memref<64x64xf16, strided<[64, 1]>, #pto.address_space<gm>> -> !pto.ptr<f16, gm>
        %memspacecast_1 = memref.memory_space_cast %15 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>> to memref<4x4x16x16xf16, #pto.address_space<mat>>
        %25 = pto.tile_buf_addr %memspacecast_1 : memref<4x4x16x16xf16, #pto.address_space<mat>> -> !pto.ptr<f16, l1>
        func.call @__pto_nd2nz_f16_gm_l1(%24, %25, %c64_i64, %c64_i64, %c128_i64, %c64_i64) : (!pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %18]
        %memspacecast_2 = memref.memory_space_cast %reinterpret_cast_0 : memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1]>, #pto.address_space<gm>>
        %26 = pto.tile_buf_addr %memspacecast_2 : memref<64x64xf16, strided<[64, 1]>, #pto.address_space<gm>> -> !pto.ptr<f16, gm>
        %memspacecast_3 = memref.memory_space_cast %16 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>> to memref<4x4x16x16xf16, #pto.address_space<mat>>
        %27 = pto.tile_buf_addr %memspacecast_3 : memref<4x4x16x16xf16, #pto.address_space<mat>> -> !pto.ptr<f16, l1>
        func.call @__pto_nd2nz_f16_gm_l1(%26, %27, %c64_i64, %c64_i64, %c128_i64, %c64_i64) : (!pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64) -> ()
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %20]
        %28 = arith.select %13, %c262144_i64, %c0_i64 : i64
        %29 = arith.extui %13 : i1 to i64
        %memspacecast_4 = memref.memory_space_cast %15 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>> to memref<4x4x16x16xf16, #pto.address_space<mat>>
        %30 = pto.tile_buf_addr %memspacecast_4 : memref<4x4x16x16xf16, #pto.address_space<mat>> -> !pto.ptr<f16, l1>
        %memspacecast_5 = memref.memory_space_cast %16 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>> to memref<4x4x16x16xf16, #pto.address_space<mat>>
        %31 = pto.tile_buf_addr %memspacecast_5 : memref<4x4x16x16xf16, #pto.address_space<mat>> -> !pto.ptr<f16, l1>
        %memspacecast_6 = memref.memory_space_cast %14 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<4x4x16x16xf32, #pto.address_space<acc>>
        %32 = pto.tile_buf_addr %memspacecast_6 : memref<4x4x16x16xf32, #pto.address_space<acc>> -> !pto.ptr<f32, l0c>
        %33 = pto.castptr %28 : i64 -> !pto.ptr<f16, l0a>
        %34 = pto.castptr %28 : i64 -> !pto.ptr<f16, l0b>
        func.call @__pto_mma_tile_half_to_float(%30, %31, %32, %33, %34, %true, %c64_i64, %c64_i64, %c64_i64, %c0_i64, %c1_i64, %20, %18, %29, %c4_i64, %c4_i64, %c16_i64, %c16_i64, %c1024_i64, %c256_i64, %c16_i64, %c1_i64, %c4_i64, %c4_i64, %c16_i64, %c16_i64, %c1024_i64, %c256_i64, %c16_i64, %c1_i64, %c4_i64, %c4_i64, %c16_i64, %c16_i64, %c1024_i64, %c256_i64, %c16_i64, %c1_i64) : (!pto.ptr<f16, l1>, !pto.ptr<f16, l1>, !pto.ptr<f32, l0c>, !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, i1, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %reinterpret_cast_7 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_FIX>]
        %memspacecast_8 = memref.memory_space_cast %14 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<4x4x16x16xf32, #pto.address_space<acc>>
        %35 = pto.tile_buf_addr %memspacecast_8 : memref<4x4x16x16xf32, #pto.address_space<acc>> -> !pto.ptr<f32, l0c>
        %memspacecast_9 = memref.memory_space_cast %reinterpret_cast_7 : memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1]>, #pto.address_space<gm>>
        %36 = pto.tile_buf_addr %memspacecast_9 : memref<64x64xf16, strided<[64, 1]>, #pto.address_space<gm>> -> !pto.ptr<f16, gm>
        pto.mte_l0c_gm %35, %36, %c64_i64, %c64_i64, %c64_i64, %c64_i64, %c0_i64, %c0_i64, pre_quant(%cst, mode = f32_f16), nz2nd : !pto.ptr<f32, l0c>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64, f16
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, %20]
      }
      hivm.hir.set_ctrl true at ctrl[60]
      %23 = arith.addi %10, %c1_i64 : i64
      memref.store %23, %alloca[%c0] : memref<1xi64>
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func private @__pto_mma_tile_half_to_float(%arg0: !pto.ptr<f16, l1>, %arg1: !pto.ptr<f16, l1>, %arg2: !pto.ptr<f32, l0c>, %arg3: !pto.ptr<f16, l0a>, %arg4: !pto.ptr<f16, l0b>, %arg5: i1, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: i64, %arg37: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "mmadl1_f16_f32_nn", pto.ptodsl.logical_name = "mmadl1_f16_f32", pto.ptodsl.source = "mmadl1_f16_f32_nn.mlir"} {
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.cmpi sge, %arg13, %c0_i64 : i64
    scf.if %0 {
      %6 = arith.index_cast %arg13 : i64 to index
      pto.wait_flag_dyn[<PIPE_M>, <PIPE_MTE1>, %6]
    }
    %1 = arith.cmpi sge, %arg9, %c0_i64 : i64
    scf.if %1 {
      %6 = arith.index_cast %arg9 : i64 to index
      pto.wait_flag_dyn[<PIPE_MTE2>, <PIPE_MTE1>, %6]
    }
    pto.mte_l1_l0a %arg0, %arg3, %arg6, %arg7, %c0_i64, %c0_i64 {transpose = false} : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64, i64, i64
    %2 = arith.cmpi sge, %arg11, %c0_i64 : i64
    scf.if %2 {
      %6 = arith.index_cast %arg11 : i64 to index
      pto.set_flag_dyn[<PIPE_MTE1>, <PIPE_MTE2>, %6]
    }
    %3 = arith.cmpi sge, %arg10, %c0_i64 : i64
    scf.if %3 {
      %6 = arith.index_cast %arg10 : i64 to index
      pto.wait_flag_dyn[<PIPE_MTE2>, <PIPE_MTE1>, %6]
    }
    pto.mte_l1_l0b %arg1, %arg4, %arg7, %arg8, %c0_i64, %c0_i64 {transpose = true} : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64
    %4 = arith.cmpi sge, %arg12, %c0_i64 : i64
    scf.if %4 {
      %6 = arith.index_cast %arg12 : i64 to index
      pto.set_flag_dyn[<PIPE_MTE1>, <PIPE_MTE2>, %6]
    }
    pto.set_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
    pto.wait_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]
    scf.if %arg5 {
      pto.mad %arg3, %arg4, %arg2, %arg6, %arg8, %arg7 disable_gemv : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
    } else {
      pto.mad_acc %arg3, %arg4, %arg2, %arg6, %arg8, %arg7 disable_gemv : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
    }
    %5 = arith.cmpi sge, %arg13, %c0_i64 : i64
    scf.if %5 {
      %6 = arith.index_cast %arg13 : i64 to index
      pto.set_flag_dyn[<PIPE_M>, <PIPE_MTE1>, %6]
    }
    return
  }
  func.func private @__pto_nd2nz_f16_gm_l1(%arg0: !pto.ptr<f16, gm>, %arg1: !pto.ptr<f16, l1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "nd2nz_f16_gm_l1", pto.ptodsl.logical_name = "nd2nz_f16", pto.ptodsl.source = "nd2nz_f16_gm_l1.mlir"} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %false = arith.constant false
    pto.mte_gm_l1_frac %arg0, %arg1, nd2nz, shape(%arg2, %arg3), src_layout(%arg4), dst_group(%c1_i64, %c1_i64, %arg5, %c0_i64), ctrl(%c0_i64, %false) : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz, shape i64, i64, src_layout(i64), dst_group i64, i64, i64, i64, ctrl i64, i1
    return
  }
  func.func private @__pto_nd2nz_bf16_gm_l1(%arg0: !pto.ptr<bf16, gm>, %arg1: !pto.ptr<bf16, l1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "nd2nz_bf16_gm_l1", pto.ptodsl.logical_name = "nd2nz_bf16", pto.ptodsl.source = "nd2nz_bf16_gm_l1.mlir"} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %false = arith.constant false
    pto.mte_gm_l1_frac %arg0, %arg1, nd2nz, shape(%arg2, %arg3), src_layout(%arg4), dst_group(%c1_i64, %c1_i64, %arg5, %c0_i64), ctrl(%c0_i64, %false) : !pto.ptr<bf16, gm>, !pto.ptr<bf16, l1>, nd2nz, shape i64, i64, src_layout(i64), dst_group i64, i64, i64, i64, ctrl i64, i1
    return
  }
  func.func private @__pto_nd2nz_i8_gm_l1(%arg0: !pto.ptr<i8, gm>, %arg1: !pto.ptr<i8, l1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "nd2nz_i8_gm_l1", pto.ptodsl.logical_name = "nd2nz_i8", pto.ptodsl.source = "nd2nz_i8_gm_l1.mlir"} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %false = arith.constant false
    pto.mte_gm_l1_frac %arg0, %arg1, nd2nz, shape(%arg2, %arg3), src_layout(%arg4), dst_group(%c1_i64, %c1_i64, %arg5, %c0_i64), ctrl(%c0_i64, %false) : !pto.ptr<i8, gm>, !pto.ptr<i8, l1>, nd2nz, shape i64, i64, src_layout(i64), dst_group i64, i64, i64, i64, ctrl i64, i1
    return
  }
  func.func private @__pto_nd2nz_f32_gm_l1(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, l1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "nd2nz_f32_gm_l1", pto.ptodsl.logical_name = "nd2nz_f32", pto.ptodsl.source = "nd2nz_f32_gm_l1.mlir"} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %false = arith.constant false
    pto.mte_gm_l1_frac %arg0, %arg1, nd2nz, shape(%arg2, %arg3), src_layout(%arg4), dst_group(%c1_i64, %c1_i64, %arg5, %c0_i64), ctrl(%c0_i64, %false) : !pto.ptr<f32, gm>, !pto.ptr<f32, l1>, nd2nz, shape i64, i64, src_layout(i64), dst_group i64, i64, i64, i64, ctrl i64, i1
    return
  }
  func.func private @__pto_fixpipe_nz2nd_f32_f32_row_split_ub(%arg0: !pto.ptr<f32, l0c>, %arg1: !pto.ptr<f32, ub>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "fixpipe_nz2nd_f32_f32_row_split_ub", pto.ptodsl.logical_name = "fixpipe_nz2nd_f32_f32_row_split_ub", pto.ptodsl.source = "fixpipe_nz2nd_f32_f32_row_split_ub.mlir"} {
    %c2_i64 = arith.constant 2 : i64
    %0 = arith.muli %arg2, %c2_i64 : i64
    pto.mte_l0c_ub %arg0, %arg1, %0, %arg3, %arg4, %arg5, dst_mode(split_m), nz2nd : !pto.ptr<f32, l0c>, !pto.ptr<f32, ub>, i64, i64, i64, i64
    return
  }
  func.func private @__pto_fixpipe_nz2nd_f32_f16_row_split_ub(%arg0: !pto.ptr<f32, l0c>, %arg1: !pto.ptr<f16, ub>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "fixpipe_nz2nd_f32_f16_row_split_ub", pto.ptodsl.logical_name = "fixpipe_nz2nd_f32_f16_row_split_ub", pto.ptodsl.source = "fixpipe_nz2nd_f32_f16_row_split_ub.mlir"} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 1.000000e+00 : f16
    %c16_i64 = arith.constant 16 : i64
    %0 = arith.muli %arg2, %c16_i64 : i64
    %1 = arith.index_cast %0 : i64 to index
    %2 = pto.addptr %arg0, %1 : <f32, l0c> -> <f32, l0c>
    pto.mte_l0c_ub %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, dst_mode(%c0_i64), pre_quant(%cst, mode = f32_f16), nz2nd : !pto.ptr<f32, l0c>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64, f16
    pto.mte_l0c_ub %2, %arg1, %arg2, %arg3, %arg4, %arg5, dst_mode(%c1_i64), pre_quant(%cst, mode = f32_f16), nz2nd : !pto.ptr<f32, l0c>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64, f16
    return
  }
  func.func private @__pto_fixpipe_nz2nd_f32_bf16_row_split_ub(%arg0: !pto.ptr<f32, l0c>, %arg1: !pto.ptr<bf16, ub>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "fixpipe_nz2nd_f32_bf16_row_split_ub", pto.ptodsl.logical_name = "fixpipe_nz2nd_f32_bf16_row_split_ub", pto.ptodsl.source = "fixpipe_nz2nd_f32_bf16_row_split_ub.mlir"} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 1.000000e+00 : bf16
    %c16_i64 = arith.constant 16 : i64
    %0 = arith.muli %arg2, %c16_i64 : i64
    %1 = arith.index_cast %0 : i64 to index
    %2 = pto.addptr %arg0, %1 : <f32, l0c> -> <f32, l0c>
    pto.mte_l0c_ub %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, dst_mode(%c0_i64), pre_quant(%cst, mode = f32_bf16), nz2nd : !pto.ptr<f32, l0c>, !pto.ptr<bf16, ub>, i64, i64, i64, i64, i64, bf16
    pto.mte_l0c_ub %2, %arg1, %arg2, %arg3, %arg4, %arg5, dst_mode(%c1_i64), pre_quant(%cst, mode = f32_bf16), nz2nd : !pto.ptr<f32, l0c>, !pto.ptr<bf16, ub>, i64, i64, i64, i64, i64, bf16
    return
  }
  func.func private @__pto_fixpipe_nz2nd_f32_bf16_single_ub(%arg0: !pto.ptr<f32, l0c>, %arg1: !pto.ptr<bf16, ub>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "fixpipe_nz2nd_f32_bf16_single_ub", pto.ptodsl.logical_name = "fixpipe_nz2nd_f32_bf16_single_ub", pto.ptodsl.source = "fixpipe_nz2nd_f32_bf16_single_ub.mlir"} {
    %cst = arith.constant 1.000000e+00 : bf16
    pto.mte_l0c_ub %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, dst_mode(%arg6), pre_quant(%cst, mode = f32_bf16), nz2nd : !pto.ptr<f32, l0c>, !pto.ptr<bf16, ub>, i64, i64, i64, i64, i64, bf16
    return
  }
}
