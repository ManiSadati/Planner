module {
  func.func @matmul_i8_i32_nn_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<64x64xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xi8> to memref<64x64xi8, strided<[64, 1]>>
    %alloc = memref.alloc() : memref<64x64xi8>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xi8, strided<[64, 1]>> to memref<64x64xi8>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<64x64xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xi8> to memref<64x64xi8, strided<[64, 1]>>
    %alloc_1 = memref.alloc() : memref<64x64xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x64xi8, strided<[64, 1]>> to memref<64x64xi8>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x64xi8>
    %4 = linalg.matmul {input_precison = "ieee"} ins(%2, %3 : tensor<64x64xi8>, tensor<64x64xi8>) outs(%1 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xi32> to memref<64x64xi32, strided<[64, 1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_2 : (tensor<64x64xi32>, memref<64x64xi32, strided<[64, 1]>>) -> ()
    return
  }
}

