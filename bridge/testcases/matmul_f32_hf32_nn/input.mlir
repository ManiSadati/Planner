module {
  func.func @matmul_f32_hf32_nn_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<64x64xf32>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1]>>
    %alloc_1 = memref.alloc() : memref<64x64xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x64xf32, strided<[64, 1]>> to memref<64x64xf32>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x64xf32>
    %4 = linalg.matmul {input_precision = "hf32"} ins(%2, %3 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<64x64xf32>, memref<64x64xf32, strided<[64, 1]>>) -> ()
    return
  }
}
