// -----// IR Dump After AppendTargetDeviceSpec (hacc-append-device-spec) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @vector_dma_pipeline_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c256 = arith.constant 256 : index
    %cst_0 = arith.constant -6.000000e+00 : f32
    %cst_1 = arith.constant 6.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %cst_2 = arith.constant 1.250000e+00 : f32
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    %2 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256xf32>) -> tensor<256xf32>
    %4 = arith.muli %arg10, %c256_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%5], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<256xf32>
    %6 = arith.addi %5, %c256 : index
    %7 = arith.index_cast %arg6 : i32 to index
    %8 = arith.maxsi %5, %7 : index
    %9 = arith.minsi %6, %8 : index
    %10 = arith.subi %9, %5 : index
    %11 = arith.cmpi slt, %10, %c256 : index
    scf.if %11 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<256xf32>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0] [%10] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_3 = memref.subview %alloc[0] [%10] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_3 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %12 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%5], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc_5 = memref.alloc() : memref<256xf32>
    scf.if %11 {
      linalg.fill ins(%cst : f32) outs(%alloc_5 : memref<256xf32>)
    } {hivm.unlikely_condition}
    %subview_6 = memref.subview %reinterpret_cast_4[0] [%10] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_7 = memref.subview %alloc_5[0] [%10] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_6, %subview_7 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %13 = bufferization.to_tensor %alloc_5 restrict writable : memref<256xf32>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%5], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc_9 = memref.alloc() : memref<256xf32>
    scf.if %11 {
      linalg.fill ins(%cst : f32) outs(%alloc_9 : memref<256xf32>)
    } {hivm.unlikely_condition}
    %subview_10 = memref.subview %reinterpret_cast_8[0] [%10] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_11 = memref.subview %alloc_9[0] [%10] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_10, %subview_11 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %14 = bufferization.to_tensor %alloc_9 restrict writable : memref<256xf32>
    %15 = arith.mulf %12, %1 : tensor<256xf32>
    %16 = arith.addf %15, %13 : tensor<256xf32>
    %17 = arith.addf %16, %14 : tensor<256xf32>
    %18 = arith.cmpf ogt, %17, %2 : tensor<256xf32>
    %19 = arith.select %18, %2, %17 : tensor<256xi1>, tensor<256xf32>
    %20 = arith.cmpf olt, %19, %3 : tensor<256xf32>
    %21 = arith.select %20, %3, %19 : tensor<256xi1>, tensor<256xf32>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg5 to offset: [%5], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %21[0] [%10] [1] : tensor<256xf32> to tensor<?xf32>
    %subview_13 = memref.subview %reinterpret_cast_12[0] [%10] [1] : memref<256xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_13 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
