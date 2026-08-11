// -----// IR Dump After AppendTargetDeviceSpec (hacc-append-device-spec) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @rmsnorm_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 2.560000e+02 : f32
    %c256_i32 = arith.constant 256 : i32
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = arith.muli %arg8, %c256_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<256xf32, strided<[1], offset: ?>> to memref<256xf32>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1]>>
    %alloc_4 = memref.alloc() : memref<256xf32>
    memref.copy %reinterpret_cast_3, %alloc_4 : memref<256xf32, strided<[1]>> to memref<256xf32>
    %5 = bufferization.to_tensor %alloc_4 restrict writable : memref<256xf32>
    %6 = arith.mulf %4, %4 : tensor<256xf32>
    %7 = bufferization.alloc_tensor() : tensor<f32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%6 : tensor<256xf32>) outs(%8 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.addf %in, %init : f32
        linalg.yield %17 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %9 = arith.divf %extracted, %cst_1 : f32
    %inserted = tensor.insert %9 into %0[%c0] : tensor<1xf32>
    %10 = arith.addf %inserted, %1 : tensor<1xf32>
    %extracted_5 = tensor.extract %10[%c0] : tensor<1xf32>
    %inserted_6 = tensor.insert %extracted_5 into %0[%c0] : tensor<1xf32>
    %11 = math.sqrt %inserted_6 : tensor<1xf32>
    %extracted_7 = tensor.extract %11[%c0] : tensor<1xf32>
    %12 = arith.divf %cst_0, %extracted_7 : f32
    %13 = tensor.empty() : tensor<256xf32>
    %14 = linalg.fill ins(%12 : f32) outs(%13 : tensor<256xf32>) -> tensor<256xf32>
    %15 = arith.mulf %4, %14 : tensor<256xf32>
    %16 = arith.mulf %15, %5 : tensor<256xf32>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%3], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %16 in writable %reinterpret_cast_8 : (tensor<256xf32>, memref<256xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
