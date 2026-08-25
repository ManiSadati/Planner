module {
  func.func @row_softmax_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant -1.000000e+09 : f32
    %c1600_i32 = arith.constant 1600 : i32
    %0 = arith.muli %arg7, %c1600_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<2048xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<2048xf32>)
    %subview = memref.subview %reinterpret_cast[0] [1600] [1] : memref<2048xf32, strided<[1], offset: ?>> to memref<1600xf32, strided<[1], offset: ?>>
    %subview_2 = memref.subview %alloc[0] [1600] [1] : memref<2048xf32> to memref<1600xf32, strided<[1]>>
    memref.copy %subview, %subview_2 : memref<1600xf32, strided<[1], offset: ?>> to memref<1600xf32, strided<[1]>>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32>
    %3 = bufferization.alloc_tensor() : tensor<f32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%2 : tensor<2048xf32>) outs(%4 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %13 = arith.maxnumf %in, %init : f32
        linalg.yield %13 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %5 = tensor.empty() : tensor<2048xf32>
    %6 = linalg.fill ins(%extracted : f32) outs(%5 : tensor<2048xf32>) -> tensor<2048xf32>
    %7 = arith.subf %2, %6 : tensor<2048xf32>
    %8 = math.exp %7 : tensor<2048xf32>
    %9 = bufferization.alloc_tensor() : tensor<f32>
    %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<f32>) -> tensor<f32>
    %reduced_3 = linalg.reduce ins(%8 : tensor<2048xf32>) outs(%10 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %13 = arith.addf %in, %init : f32
        linalg.yield %13 : f32
      }
    %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
    %11 = linalg.fill ins(%extracted_4 : f32) outs(%5 : tensor<2048xf32>) -> tensor<2048xf32>
    %12 = arith.divf %8, %11 : tensor<2048xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %12[0] [1600] [1] : tensor<2048xf32> to tensor<1600xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [1600] [1] : memref<2048xf32, strided<[1], offset: ?>> to memref<1600xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<1600xf32>, memref<1600xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

