// -----// IR Dump After AppendTargetDeviceSpec (hacc-append-device-spec) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @vector_elementwise_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c256 = arith.constant 256 : index
    %cst = arith.constant 5.000000e-01 : f32
    %c256_i32 = arith.constant 256 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    %3 = arith.muli %arg9, %c256_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%4], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<256xf32>
    %5 = arith.addi %4, %c256 : index
    %6 = arith.index_cast %arg5 : i32 to index
    %7 = arith.maxsi %4, %6 : index
    %8 = arith.minsi %5, %7 : index
    %9 = arith.subi %8, %4 : index
    %10 = arith.cmpi slt, %9, %c256 : index
    scf.if %10 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc : memref<256xf32>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0] [%9] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%9] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %11 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%4], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<256xf32>
    scf.if %10 {
      linalg.fill ins(%cst_0 : f32) outs(%alloc_3 : memref<256xf32>)
    } {hivm.unlikely_condition}
    %subview_4 = memref.subview %reinterpret_cast_2[0] [%9] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_5 = memref.subview %alloc_3[0] [%9] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_4, %subview_5 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %12 = bufferization.to_tensor %alloc_3 restrict writable : memref<256xf32>
    %13 = arith.addf %11, %12 : tensor<256xf32>
    %14 = arith.mulf %13, %2 : tensor<256xf32>
    %15 = arith.cmpf ogt, %14, %1 : tensor<256xf32>
    %16 = arith.select %15, %14, %1 : tensor<256xi1>, tensor<256xf32>
    %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%4], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %16[0] [%9] [1] : tensor<256xf32> to tensor<?xf32>
    %subview_7 = memref.subview %reinterpret_cast_6[0] [%9] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_7 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
