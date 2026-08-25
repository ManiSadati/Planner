#map = affine_map<(d0) -> (d0)>
module {
  func.func @rmsnorm_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.560000e+02 : f32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %4 = tensor.empty() : tensor<256xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<256xf32>) -> tensor<256xf32>
    %6 = tensor.empty() : tensor<256xi32>
    %7 = linalg.fill ins(%c256_i32 : i32) outs(%6 : tensor<256xi32>) -> tensor<256xi32>
    %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%6 : tensor<256xi32>) {
    ^bb0(%out: i32):
      %25 = linalg.index 0 : index
      %26 = arith.index_cast %25 : index to i32
      linalg.yield %26 : i32
    } -> tensor<256xi32>
    %9 = arith.cmpi slt, %8, %7 : tensor<256xi32>
    %10 = arith.muli %arg8, %c256_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<256xf32, strided<[1], offset: ?>> to memref<256xf32>
    %12 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1]>>
    %alloc_4 = memref.alloc() : memref<256xf32>
    memref.copy %reinterpret_cast_3, %alloc_4 : memref<256xf32, strided<[1]>> to memref<256xf32>
    %13 = bufferization.to_tensor %alloc_4 restrict writable : memref<256xf32>
    %14 = arith.mulf %12, %12 : tensor<256xf32>
    %15 = arith.select %9, %14, %5 : tensor<256xi1>, tensor<256xf32>
    %16 = bufferization.alloc_tensor() : tensor<f32>
    %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%15 : tensor<256xf32>) outs(%17 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %25 = arith.addf %in, %init : f32
        linalg.yield %25 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %inserted = tensor.insert %extracted into %0[%c0] : tensor<1xf32>
    %18 = arith.divf %inserted, %3 : tensor<1xf32>
    %extracted_5 = tensor.extract %18[%c0] : tensor<1xf32>
    %inserted_6 = tensor.insert %extracted_5 into %0[%c0] : tensor<1xf32>
    %19 = arith.addf %inserted_6, %2 : tensor<1xf32>
    %extracted_7 = tensor.extract %19[%c0] : tensor<1xf32>
    %inserted_8 = tensor.insert %extracted_7 into %0[%c0] : tensor<1xf32>
    %20 = math.sqrt %inserted_8 : tensor<1xf32>
    %extracted_9 = tensor.extract %20[%c0] : tensor<1xf32>
    %inserted_10 = tensor.insert %extracted_9 into %0[%c0] : tensor<1xf32>
    %21 = arith.divf %1, %inserted_10 : tensor<1xf32>
    %extracted_11 = tensor.extract %21[%c0] : tensor<1xf32>
    %22 = linalg.fill ins(%extracted_11 : f32) outs(%4 : tensor<256xf32>) -> tensor<256xf32>
    %23 = arith.mulf %12, %22 : tensor<256xf32>
    %24 = arith.mulf %23, %13 : tensor<256xf32>
    %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%11], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %24 in writable %reinterpret_cast_12 : (tensor<256xf32>, memref<256xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

