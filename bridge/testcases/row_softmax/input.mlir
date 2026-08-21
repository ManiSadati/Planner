// -----// IR Dump After AppendTargetDeviceSpec (hacc-append-device-spec) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @row_softmax_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.muli %arg7, %c256_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<256xf32, strided<[1], offset: ?>> to memref<256xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %3 = bufferization.alloc_tensor() : tensor<f32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%2 : tensor<256xf32>) outs(%4 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %13 = arith.maxnumf %in, %init : f32
        linalg.yield %13 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %5 = tensor.empty() : tensor<256xf32>
    %6 = linalg.fill ins(%extracted : f32) outs(%5 : tensor<256xf32>) -> tensor<256xf32>
    %7 = arith.subf %2, %6 : tensor<256xf32>
    %8 = math.exp %7 : tensor<256xf32>
    %9 = bufferization.alloc_tensor() : tensor<f32>
    %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<f32>) -> tensor<f32>
    %reduced_1 = linalg.reduce ins(%8 : tensor<256xf32>) outs(%10 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %extracted_2 = tensor.extract %reduced_1[] : tensor<f32>
    %11 = linalg.fill ins(%extracted_2 : f32) outs(%5 : tensor<256xf32>) -> tensor<256xf32>
    %12 = arith.divf %8, %11 : tensor<256xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %12 in writable %reinterpret_cast_3 : (tensor<256xf32>, memref<256xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
