module {
  func.func @qk_matmul_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c4096_i32 = arith.constant 4096 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = arith.muli %arg8, %c4096_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%3], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [1, 64] : memref<?xf16> to memref<64x64xf16, strided<[1, 64]>>
    %alloc = memref.alloc() : memref<64x64xf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
    %4 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
    %alloc_1 = memref.alloc() : memref<64x64xf16>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x64xf16, strided<[1, 64]>> to memref<64x64xf16>
    annotation.mark %alloc_1 {MayImplicitTransposeWithLastAxis} : memref<64x64xf16>
    %5 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x64xf16>
    annotation.mark %5 {MayImplicitTransposeWithLastAxis} : tensor<64x64xf16>
    %6 = linalg.matmul {input_precison = "ieee"} ins(%4, %5 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%3], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %7 = arith.truncf %6 : tensor<64x64xf32> to tensor<64x64xf16>
    bufferization.materialize_in_destination %7 in writable %reinterpret_cast_2 : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}

