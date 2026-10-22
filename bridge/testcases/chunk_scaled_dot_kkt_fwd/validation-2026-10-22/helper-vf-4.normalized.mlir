#map = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @chunk_scaled_dot_kkt_fwd_kernel_mix_aiv_outlined_vf_4(%arg0: index, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<32x1xbf16, #hivm.address_space<ub>>, %arg3: memref<64xi32, #hivm.address_space<ub>>, %arg4: memref<64xf16, #hivm.address_space<ub>>, %arg5: memref<2x32x64xf32, #hivm.address_space<ub>>, %arg6: index, %arg7: memref<2x32x1xf32, #hivm.address_space<ub>>, %arg8: index, %arg9: memref<2x32x64xi1, strided<[8192, 256, 1]>, #hivm.address_space<ub>>, %arg10: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, hivm.vector_function, no_inline} {
    %c256 = arith.constant 256 : index
    %c8192 = arith.constant 8192 : index
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f16, vector<64xi1> -> vector<64xf16>
    %2 = ave.hir.pge <VL1> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %res = ave.hir.vload <NORM> %arg1[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %res_0 = ave.hir.vload <NORM> %arg3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xi32, #hivm.address_space<ub>> into vector<64xi32>
    %res_1 = ave.hir.vload <UNPK_B16> %arg4[%c0] {functionType = #ave.func_dist_type<pk>} : memref<64xf16, #hivm.address_space<ub>> into vector<64xf16>
    %3 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %4 = ave.hir.vcmp <NE> %res_1, %1, %3 : vector<64xf16>, vector<64xi1> -> vector<64xi1>
    scf.for %arg11 = %c0 to %c32 step %c1 {
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg1 : memref<64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
      %5 = arith.muli %arg0, %c32 : index
      %6 = arith.addi %arg11, %5 : index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%6], sizes: [1], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg2 : memref<32x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [%arg11], sizes: [1, 1], strides: [1, 1] : memref<bf16, #hivm.address_space<ub>> to memref<1x1xbf16, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg7 : memref<2x32x1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %7 = arith.muli %arg8, %c32 : index
      %8 = arith.addi %7, %arg11 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_7 to offset: [%8], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      %res_12 = ave.hir.vload <BRC_B16> %reinterpret_cast_6[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pk>} : memref<1x1xbf16, strided<[1, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xbf16>
      %9 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %10 = ave.hir.vextf %res_12, <part_even>, %9 {functionType = #ave.func_dist_type<even>} : vector<64xbf16>, vector<64xf32>, vector<64xi1>
      ave.hir.masked_store <ONEPT_B32> %reinterpret_cast_11[%c0, %c0], %2, %10 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %arg3 : memref<64xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
      %11 = arith.muli %arg0, %c32 : index
      %12 = arith.addi %arg11, %11 : index
      %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [%12], sizes: [1], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_18, %offset_19, %sizes_20, %strides_21 = memref.extract_strided_metadata %arg4 : memref<64xf16, #hivm.address_space<ub>> -> memref<f16, #hivm.address_space<ub>>, index, index, index
      %13 = arith.muli %arg0, %c32 : index
      %14 = arith.addi %arg11, %13 : index
      %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [%14], sizes: [1], strides: [1] : memref<f16, #hivm.address_space<ub>> to memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %res_23 = ave.hir.vload <BRC_B32> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %15 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %16 = ave.hir.vexpdif %res_23, %res, %15 : vector<64xf32>, vector<64xi1>
      %base_buffer_24, %offset_25, %sizes_26:3, %strides_27:3 = memref.extract_strided_metadata %arg5 : memref<2x32x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %17 = arith.muli %arg6, %c2048 : index
      %18 = arith.muli %arg11, %c64 : index
      %19 = arith.addi %17, %18 : index
      %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [%19], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %20 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %reinterpret_cast_28[%c0], %20, %16 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      %res_29 = ave.hir.vload <BRC_B32> %reinterpret_cast_17[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
      %res_30 = ave.hir.vload <BRC_B16> %reinterpret_cast_22[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pk>} : memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf16>
      %21 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %22 = ave.hir.vcmp <NE> %res_30, %1, %21 : vector<64xf16>, vector<64xi1> -> vector<64xi1>
      %23 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %24 = ave.hir.preg.and <b8> %22, %4, %23 : vector<64xi1>
      %25 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %26 = ave.hir.vcmp <GT> %res_29, %res_0, %25 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
      %27 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %28 = ave.hir.preg.and <b8> %26, %24, %27 : vector<64xi1>
      %base_buffer_31, %offset_32, %sizes_33:3, %strides_34:3 = memref.extract_strided_metadata %arg9 : memref<2x32x64xi1, strided<[8192, 256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %29 = arith.muli %arg10, %c8192 : index
      %30 = arith.muli %arg11, %c256 : index
      %31 = arith.addi %29, %30 : index
      %reinterpret_cast_35 = memref.reinterpret_cast %base_buffer_31 to offset: [%31], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, #map, #hivm.address_space<ub>>
      %32 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B8> %reinterpret_cast_35[%c0], %32, %28 {functionType = #ave.func_dist_type<pb32>} : memref<64xi1, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
    }
    return
  }
}

