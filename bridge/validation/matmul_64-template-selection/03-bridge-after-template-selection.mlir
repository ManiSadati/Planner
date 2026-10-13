// -----// IR Dump After SelectHIVMPTODSLTemplates (select-hivm-ptodsl-templates) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">, hivm.module_core_type = #hivm.module_core_type<AIC>} {
  func.func @matmul64_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, false, false, false]> : vector<8xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.storage_aligned, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
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
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>>) outs(%15 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %18]
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_0 : memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>>) outs(%16 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, %20]
        hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C, pto.ptodsl.native_template = "mma_tile_half_to_float", pto.ptodsl.template = "__pto_mma_tile_half_to_float"} ins(%15, %16, %true, %c64, %c64, %c64 : memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, memref<4x4x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%14 : memref<4x4x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c1_i64, %20, %18, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        hivm.hir.pipe_barrier[<PIPE_FIX>]
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>} ins(%14 : memref<4x4x16x16xf32, #hivm.address_space<cc>>) outs(%reinterpret_cast_1 : memref<64x64xf16, strided<[64, 1]>, #hivm.address_space<gm>>)
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
}
