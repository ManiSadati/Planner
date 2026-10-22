#map = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @chunk_scaled_dot_kkt_fwd_kernel_mix_aiv_outlined_vf_5(%arg0: memref<2x32x1xf32, #hivm.address_space<ub>>, %arg1: index, %arg2: memref<2x32x64xi1, strided<[8192, 256, 1]>, #hivm.address_space<ub>>, %arg3: memref<2x32x64xf32, #hivm.address_space<ub>>, %arg4: memref<2x32x64xf32, #hivm.address_space<ub>>, %arg5: memref<32x64xbf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, hivm.vector_function, no_inline} {
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c8192 = arith.constant 8192 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
    scf.for %arg6 = %c0 to %c32 step %c1 {
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<2x32x1xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %2 = arith.muli %arg1, %c32 : index
      %3 = arith.addi %2, %arg6 : index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%3], sizes: [1, 1], strides: [32, 1] : memref<f32, #hivm.address_space<ub>> to memref<1x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      %base_buffer_0, %offset_1, %sizes_2:3, %strides_3:3 = memref.extract_strided_metadata %arg2 : memref<2x32x64xi1, strided<[8192, 256, 1]>, #hivm.address_space<ub>> -> memref<i1, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %4 = arith.muli %arg1, %c8192 : index
      %5 = arith.muli %arg6, %c256 : index
      %6 = arith.addi %4, %5 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%6], sizes: [64], strides: [1] : memref<i1, #hivm.address_space<ub>> to memref<64xi1, #map, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %reinterpret_cast_4[%c0] {functionType = #ave.func_dist_type<pb32>} : memref<64xi1, #map, #hivm.address_space<ub>> into vector<64xi1>
      %base_buffer_5, %offset_6, %sizes_7:3, %strides_8:3 = memref.extract_strided_metadata %arg3 : memref<2x32x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %7 = arith.muli %arg1, %c2048 : index
      %8 = arith.muli %arg6, %c64 : index
      %9 = arith.addi %7, %8 : index
      %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%9], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_10 = ave.hir.vload <NORM> %reinterpret_cast_9[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %base_buffer_11, %offset_12, %sizes_13:3, %strides_14:3 = memref.extract_strided_metadata %arg4 : memref<2x32x64xf32, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index, index, index, index, index
      %10 = arith.muli %arg1, %c2048 : index
      %11 = arith.muli %arg6, %c64 : index
      %12 = arith.addi %10, %11 : index
      %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [%12], sizes: [64], strides: [1] : memref<f32, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_16 = ave.hir.vload <NORM> %reinterpret_cast_15[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %res_17 = ave.hir.vload <BRC_B32> %reinterpret_cast[%c0, %c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1x1xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %13 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %14 = ave.hir.vmul %res_10, %res_16, %13 : vector<64xf32>, vector<64xi1>
      %15 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %16 = ave.hir.vmul %14, %res_17, %15 : vector<64xf32>, vector<64xi1>
      %17 = ave.hir.vsel %res, %16, %1 : vector<64xi1>, vector<64xf32>
      %18 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %19 = ave.hir.vtruncf %17, <rint>, false, <part_even>, %18 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xbf16>, vector<64xi1>
      %base_buffer_18, %offset_19, %sizes_20:2, %strides_21:2 = memref.extract_strided_metadata %arg5 : memref<32x64xbf16, #hivm.address_space<ub>> -> memref<bf16, #hivm.address_space<ub>>, index, index, index, index, index
      %20 = arith.muli %arg6, %c64 : index
      %reinterpret_cast_22 = memref.reinterpret_cast %base_buffer_18 to offset: [%20], sizes: [64], strides: [1] : memref<bf16, #hivm.address_space<ub>> to memref<64xbf16, #map, #hivm.address_space<ub>>
      %21 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %22 = builtin.unrealized_conversion_cast %19 : vector<64xbf16> to vector<128xbf16>
      %23 = llvm.bitcast %22 : vector<128xbf16> to vector<64xf32>
      ave.hir.masked_store <PK_B32> %reinterpret_cast_22[%c0], %21, %23 {functionType = #ave.func_dist_type<pk>, hivm.is_continuous} : memref<64xbf16, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
}

