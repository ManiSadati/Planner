module {
  func.func @matmul64_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = arith.cmpi ne, %arg8, %c0_i32 : i32
    scf.if %2 {
    } else {
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1]>>
      %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1]>>
      %alloc = memref.alloc() : memref<64x64xf16>
      memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16>
      %3 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
      %alloc_1 = memref.alloc() : memref<64x64xf16>
      memref.copy %reinterpret_cast_0, %alloc_1 : memref<64x64xf16, strided<[64, 1]>> to memref<64x64xf16>
      %4 = bufferization.to_tensor %alloc_1 restrict writable : memref<64x64xf16>
      %5 = linalg.matmul {input_precison = "ieee"} ins(%3, %4 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1]>>
      %6 = arith.truncf %5 : tensor<64x64xf32> to tensor<64x64xf16>
      bufferization.materialize_in_destination %6 in writable %reinterpret_cast_2 : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1]>>) -> ()
    }
    return
  }
}

