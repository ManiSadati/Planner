module {
  func.func @vector_add_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<256xf32, strided<[1]>> to memref<256xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<256xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<256xf32, strided<[1]>> to memref<256xf32>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<256xf32>
    %2 = arith.addf %0, %1 : tensor<256xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1]>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_2 : (tensor<256xf32>, memref<256xf32, strided<[1]>>) -> ()
    return
  }
}

