module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9589">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  func.func @dma_native_kernel_outlined_vf_0(%arg0: memref<128xi16, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline, pto.bridge.compiler_owned_vector_helper} {
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
    %1 = ave.hir.broadcast %c0_i16, %0 : i16, vector<128xi1> -> vector<128xi16>
    %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
    ave.hir.masked_store <NORM_B16> %arg0[%c0], %2, %1 {functionType = #ave.func_dist_type<norm>} : memref<128xi16, #hivm.address_space<ub>>, vector<128xi1>, vector<128xi16>
    return
  }
  func.func private @load_gm_to_ubuf_1d_int16_t(memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i16, index, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, pto.native_template_fallback}
  func.func private @store_ubuf_to_gm_1d_int16_t(memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, pto.native_template_fallback}
  func.func @dma_native_kernel(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xi16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xi16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, false, false, false]> : vector<7xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "aiv", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i64 = arith.constant 0 : i64
    %c16 = arith.constant 16 : index
    %c128_i32 = arith.constant 128 : i32
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c0_i16 = arith.constant 0 : i16
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.muli %arg4, %arg5 : i32
    %1 = arith.muli %0, %arg6 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    scf.for %arg7 = %3 to %1 step %c64_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %4 = arith.remsi %arg7, %arg4 : i32
      %5 = arith.muli %4, %c128_i32 : i32
      %6 = arith.index_cast %5 : i32 to index
      %c32 = arith.constant 32 : index
      %7 = arith.addi %6, %c32 : index
      %8 = hivm.hir.pointer_cast(%c0_i64) : memref<128xi16, #hivm.address_space<ub>>
      %c128_0 = arith.constant 128 : index
      %9 = arith.addi %6, %c128_0 : index
      %10 = arith.maxsi %6, %c128 : index
      %11 = arith.minsi %9, %10 : index
      %c-1 = arith.constant -1 : index
      %12 = arith.muli %6, %c-1 : index
      %13 = arith.addi %11, %12 : index
      %14 = arith.maxsi %6, %c0 : index
      %15 = arith.minsi %9, %14 : index
      %c-1_1 = arith.constant -1 : index
      %16 = arith.muli %6, %c-1_1 : index
      %17 = arith.addi %15, %16 : index
      %18 = arith.maxsi %17, %c0 : index
      %19 = arith.minsi %13, %c128 : index
      %c-1_2 = arith.constant -1 : index
      %20 = arith.muli %18, %c-1_2 : index
      %21 = arith.addi %19, %20 : index
      %22 = arith.maxsi %21, %c0 : index
      %23 = arith.cmpi slt, %22, %c128 : index
      scf.if %23 {
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        func.call @dma_native_kernel_outlined_vf_0(%8) {hivm.vector_function, no_inline} : (memref<128xi16, #hivm.address_space<ub>>) -> ()
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<?xi16, #hivm.address_space<gm>> -> memref<i16, #hivm.address_space<gm>>, index, index, index
      %24 = arith.addi %7, %18 : index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%24], sizes: [%22], strides: [1] : memref<i16, #hivm.address_space<gm>> to memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %8 : memref<128xi16, #hivm.address_space<ub>> -> memref<i16, #hivm.address_space<ub>>, index, index, index
      %reinterpret_cast_7 = memref.reinterpret_cast %base_buffer_3 to offset: [%18], sizes: [%22], strides: [1] : memref<i16, #hivm.address_space<ub>> to memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %25 = arith.remui %18, %c16 : index
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      %cast = memref.cast %reinterpret_cast : memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      %cast_8 = memref.cast %reinterpret_cast_7 : memref<?xi16, strided<[1], offset: ?>, #hivm.address_space<ub>> to memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @load_gm_to_ubuf_1d_int16_t(%cast, %cast_8, %c2_i32, %c0_i16, %25, %c0_i32) {pto.native_template_fallback} : (memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32, i16, index, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %reinterpret_cast_9 = memref.reinterpret_cast %arg3 to offset: [%7], sizes: [128], strides: [1] : memref<?xi16, #hivm.address_space<gm>> to memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<gm>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
      %cast_10 = memref.cast %8 : memref<128xi16, #hivm.address_space<ub>> to memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<ub>>
      %cast_11 = memref.cast %reinterpret_cast_9 : memref<128xi16, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_int16_t(%cast_10, %cast_11, %c0_i32) {pto.native_template_fallback} : (memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi16, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}
