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
  func.func @chunk_scaled_dot_kkt_fwd_kernel_mix_aiv_outlined_vf_5(%arg0: memref<2x32x1xf32, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<2x32x64xi1, strided<[8192, 256, 1]>, #hivm.address_space<ub>>, %arg3: memref<2x32x64xf32, #hivm.address_space<ub>>, %arg4: memref<2x32x64xf32, #hivm.address_space<ub>>, %arg5: memref<32x64xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
    scf.for %arg6 = %c0 to %c32 step %c1 {
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<2x32x1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %2 = affine.apply #map16()[%arg1, %arg6]
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%2], sizes: [1, 1], strides: [32, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_0, %offset_1, %sizes_2:3, %strides_3:3 = memref.extract_strided_metadata %arg2 : memref<2x32x64xi1, strided<[8192, 256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %3 = affine.apply #map17()[%arg1, %arg6]
      %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%3], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, #map13, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %reinterpret_cast_4[%c0] {functionType = #ave.func_dist_type<pb32>} : memref<64xi1, #map13, #hivm.address_space<ub>> into vector<64xi1>
      %base_buffer_5, %offset_6, %sizes_7:3, %strides_8:3 = memref.extract_strided_metadata %arg3 : memref<2x32x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %4 = affine.apply #map18()[%arg1, %arg6]
      %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%4], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map13, #hivm.address_space<ub>>
      %res_10 = ave.hir.vload <NORM> %reinterpret_cast_9[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map13, #hivm.address_space<ub>> into vector<64xf32>
      %base_buffer_11, %offset_12, %sizes_13:3, %strides_14:3 = memref.extract_strided_metadata %arg4 : memref<2x32x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %5 = affine.apply #map18()[%arg1, %arg6]
      %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [%5], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map13, #hivm.address_space<ub>>
      %res_16 = ave.hir.vload <NORM> %reinterpret_cast_15[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map13, #hivm.address_space<ub>> into vector<64xf32>
      %res_17 = ave.hir.vload <BRC_B32> %reinterpret_cast[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %6 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %7 = ave.hir.vmul %res_10, %res_16, %6 : vector<64xf32>, vector<64xi1>
      %8 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %9 = ave.hir.vmul %7, %res_17, %8 : vector<64xf32>, vector<64xi1>
      %10 = ave.hir.vsel %res, %9, %1 : vector<64xi1>, vector<64xf32>
      %11 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %12 = ave.hir.vtruncf %10, <rint>, false, <part_even>, %11 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xbf16>, vector<64xi1>
      %base_buffer_18, %offset_19, %sizes_20:2, %strides_21:2 = memref.extract_strided_metadata %arg5 : memref<32x64xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
      %13 = affine.apply #map19()[%arg6]
      %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [%13], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, #map13, #hivm.address_space<ub>>
      %14 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %15 = builtin.unrealized_conversion_cast %12 : vector<64xbf16> to vector<128xbf16>
      %16 = llvm.bitcast %15 : vector<128xbf16> to vector<64xf32>
      ave.hir.masked_store <PK_B32> %reinterpret_cast_22[%c0], %14, %16 {functionType = #ave.func_dist_type<pk>, hivm.is_continuous} : memref<64xbf16, #map13, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
}
