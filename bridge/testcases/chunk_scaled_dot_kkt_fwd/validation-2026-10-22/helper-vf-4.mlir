#map = affine_map<(d0)[s0] -> ((-d0 + s0) ceildiv 32)>
#map1 = affine_map<()[s0] -> (s0 * 2048)>
#map2 = affine_map<(d0)[s0] -> (d0 * 32 + s0)>
#map3 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 256)>
#map4 = affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 256)>
#map5 = affine_map<()[s0] -> (-s0 + (s0 floordiv 256) * 256 + 128)>
#map6 = affine_map<()[s0, s1] -> (s0 - s1)>
#map7 = affine_map<()[s0, s1] -> ((s0 - s1) ceildiv 16)>
#map8 = affine_map<(d0, d1) -> (d0 + d1 * 16 + (d0 floordiv 16) * 1008)>
#map9 = affine_map<()[s0, s1] -> (s0 + s1 * 32)>
#map10 = affine_map<()[s0] -> (s0 * 32)>
#map11 = affine_map<()[s0, s1] -> (s0 + s1)>
#map12 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map13 = affine_map<(d0)[s0] -> (d0 + s0)>
#map14 = affine_map<()[s0] -> (s0 * 8192)>
#map15 = affine_map<()[s0, s1] -> (s0 + s1 * 256)>
#map16 = affine_map<()[s0, s1] -> (s0 * 32 + s1)>
#map17 = affine_map<()[s0, s1] -> (s0 * 8192 + s1 * 256)>
#map18 = affine_map<()[s0, s1] -> (s0 * 2048 + s1 * 64)>
#map19 = affine_map<()[s0] -> (s0 * 64)>
#map20 = affine_map<()[s0] -> (s0 * 32 + 32)>
#map21 = affine_map<()[s0, s1, s2] -> (s0 * 2 + s1 + s2)>
#map22 = affine_map<()[s0, s1, s2] -> (-s0 + s1 - s2)>
#map23 = affine_map<()[s0, s1] -> (s0 - s1 * 32)>
#map24 = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map25 = affine_map<()[s0, s1] -> (s0 * 64 - s1 * 2048)>
#map26 = affine_map<()[s0, s1] -> (s0 + s1 * 4096)>
module {
  func.func @chunk_scaled_dot_kkt_fwd_kernel_mix_aiv_outlined_vf_4(%arg0: index, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<32x1xbf16, #hivm.address_space<ub>>, %arg3: memref<64xi32, #hivm.address_space<ub>>, %arg4: memref<64xf16, #hivm.address_space<ub>>, %arg5: memref<2x32x64xf32, #hivm.address_space<ub>>, %arg6: index, %arg7: memref<2x32x1xf32, #hivm.address_space<ub>>, %arg8: index, %arg9: memref<2x32x64xi1, strided<[8192, 256, 1]>, #hivm.address_space<ub>>, %arg10: index) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, hivm.vector_function, no_inline} {
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
      %5 = affine.apply #map9()[%arg11, %arg0]
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%5], sizes: [1], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg2 : memref<32x1xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
      %reinterpret_cast_6 = memref.reinterpret_cast %base_buffer_2 to offset: [%arg11], sizes: [1, 1], strides: [1, 1] : memref<bf16, #hivm.address_space<ub>> to memref<1x1xbf16, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg7 : memref<2x32x1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %6 = affine.apply #map10()[%arg8]
      %7 = affine.apply #map11()[%6, %arg11]
      %reinterpret_cast_11 = memref.reinterpret_cast %base_buffer_7 to offset: [%7], sizes: [1, 1], strides: [1, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>
      %res_12 = ave.hir.vload <BRC_B16> %reinterpret_cast_6[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pk>} : memref<1x1xbf16, strided<[1, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xbf16>
      %8 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %9 = ave.hir.vextf %res_12, <part_even>, %8 {functionType = #ave.func_dist_type<even>} : vector<64xbf16>, vector<64xf32>, vector<64xi1>
      ave.hir.masked_store <ONEPT_B32> %reinterpret_cast_11[%c0, %c0], %2, %9 {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1x1xf32, strided<[1, 1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      %base_buffer_13, %offset_14, %sizes_15, %strides_16 = memref.extract_strided_metadata %arg3 : memref<64xi32, #hivm.address_space<ub>> -> memref<i32, #hivm.address_space<ub>>, index, index, index
      %10 = affine.apply #map9()[%arg11, %arg0]
      %reinterpret_cast_17 = memref.reinterpret_cast %base_buffer_13 to offset: [%10], sizes: [1], strides: [1] : memref<i32, #hivm.address_space<ub>> to memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_18, %offset_19, %sizes_20, %strides_21 = memref.extract_strided_metadata %arg4 : memref<64xf16, #hivm.address_space<ub>> -> memref<f16, #hivm.address_space<ub>>, index, index, index
      %11 = affine.apply #map9()[%arg11, %arg0]
      %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [%11], sizes: [1], strides: [1] : memref<f16, #hivm.address_space<ub>> to memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %res_23 = ave.hir.vload <BRC_B32> %reinterpret_cast[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %12 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %13 = ave.hir.vexpdif %res_23, %res, %12 : vector<64xf32>, vector<64xi1>
      %base_buffer_24, %offset_25, %sizes_26:3, %strides_27:3 = memref.extract_strided_metadata %arg5 : memref<2x32x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %14 = affine.apply #map1()[%arg6]
      %15 = affine.apply #map12()[%14, %arg11]
      %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [%15], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map13, #hivm.address_space<ub>>
      %16 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %reinterpret_cast_28[%c0], %16, %13 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, #map13, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
      %res_29 = ave.hir.vload <BRC_B32> %reinterpret_cast_17[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xi32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xi32>
      %res_30 = ave.hir.vload <BRC_B16> %reinterpret_cast_22[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<pk>} : memref<1xf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf16>
      %17 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %18 = ave.hir.vcmp <NE> %res_30, %1, %17 : vector<64xf16>, vector<64xi1> -> vector<64xi1>
      %19 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %20 = ave.hir.preg.and <b8> %18, %4, %19 : vector<64xi1>
      %21 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %22 = ave.hir.vcmp <GT> %res_29, %res_0, %21 : vector<64xi32>, vector<64xi1> -> vector<64xi1>
      %23 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %24 = ave.hir.preg.and <b8> %22, %20, %23 : vector<64xi1>
      %base_buffer_31, %offset_32, %sizes_33:3, %strides_34:3 = memref.extract_strided_metadata %arg9 : memref<2x32x64xi1, strided<[8192, 256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %25 = affine.apply #map14()[%arg10]
      %26 = affine.apply #map15()[%25, %arg11]
      %reinterpret_cast_35 = memref.reinterpret_cast %base_buffer_31 to offset: [%26], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, #map13, #hivm.address_space<ub>>
      %27 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B8> %reinterpret_cast_35[%c0], %27, %24 {functionType = #ave.func_dist_type<pb32>} : memref<64xi1, #map13, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi1>
    }
    return
  }
}
