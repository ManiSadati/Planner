#map = affine_map<(d0) -> (d0)>
module {
  func.func @rmsnorm_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c1600_i32 = arith.constant 1600 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.600000e+03 : f32
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %4 = tensor.empty() : tensor<2048xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<2048xf32>) -> tensor<2048xf32>
    %6 = tensor.empty() : tensor<2048xi32>
    %7 = linalg.fill ins(%c1600_i32 : i32) outs(%6 : tensor<2048xi32>) -> tensor<2048xi32>
    %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%6 : tensor<2048xi32>) {
    ^bb0(%out: i32):
      %25 = linalg.index 0 : index
      %26 = arith.index_cast %25 : index to i32
      linalg.yield %26 : i32
    } -> tensor<2048xi32>
    %9 = arith.cmpi slt, %8, %7 : tensor<2048xi32>
    %10 = arith.muli %arg8, %c1600_i32 : i32
    %11 = arith.index_cast %10 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<2048xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<2048xf32>)
    %subview = memref.subview %reinterpret_cast[0] [1600] [1] : memref<2048xf32, strided<[1], offset: ?>> to memref<1600xf32, strided<[1], offset: ?>>
    %subview_3 = memref.subview %alloc[0] [1600] [1] : memref<2048xf32> to memref<1600xf32, strided<[1]>>
    memref.copy %subview, %subview_3 : memref<1600xf32, strided<[1], offset: ?>> to memref<1600xf32, strided<[1]>>
    %12 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1]>>
    %alloc_5 = memref.alloc() : memref<2048xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_5 : memref<2048xf32>)
    %subview_6 = memref.subview %reinterpret_cast_4[0] [1600] [1] : memref<2048xf32, strided<[1]>> to memref<1600xf32, strided<[1]>>
    %subview_7 = memref.subview %alloc_5[0] [1600] [1] : memref<2048xf32> to memref<1600xf32, strided<[1]>>
    memref.copy %subview_6, %subview_7 : memref<1600xf32, strided<[1]>> to memref<1600xf32, strided<[1]>>
    %13 = bufferization.to_tensor %alloc_5 restrict writable : memref<2048xf32>
    %14 = arith.mulf %12, %12 : tensor<2048xf32>
    %15 = arith.select %9, %14, %5 : tensor<2048xi1>, tensor<2048xf32>
    %16 = bufferization.alloc_tensor() : tensor<f32>
    %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%15 : tensor<2048xf32>) outs(%17 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %25 = arith.addf %in, %init : f32
        linalg.yield %25 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %inserted = tensor.insert %extracted into %0[%c0] : tensor<1xf32>
    %18 = arith.divf %inserted, %3 : tensor<1xf32>
    %extracted_8 = tensor.extract %18[%c0] : tensor<1xf32>
    %inserted_9 = tensor.insert %extracted_8 into %0[%c0] : tensor<1xf32>
    %19 = arith.addf %inserted_9, %2 : tensor<1xf32>
    %extracted_10 = tensor.extract %19[%c0] : tensor<1xf32>
    %inserted_11 = tensor.insert %extracted_10 into %0[%c0] : tensor<1xf32>
    %20 = math.sqrt %inserted_11 : tensor<1xf32>
    %extracted_12 = tensor.extract %20[%c0] : tensor<1xf32>
    %inserted_13 = tensor.insert %extracted_12 into %0[%c0] : tensor<1xf32>
    %21 = arith.divf %1, %inserted_13 : tensor<1xf32>
    %extracted_14 = tensor.extract %21[%c0] : tensor<1xf32>
    %22 = linalg.fill ins(%extracted_14 : f32) outs(%4 : tensor<2048xf32>) -> tensor<2048xf32>
    %23 = arith.mulf %12, %22 : tensor<2048xf32>
    %24 = arith.mulf %23, %13 : tensor<2048xf32>
    %reinterpret_cast_15 = memref.reinterpret_cast %arg4 to offset: [%11], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %24[0] [1600] [1] : tensor<2048xf32> to tensor<1600xf32>
    %subview_16 = memref.subview %reinterpret_cast_15[0] [1600] [1] : memref<2048xf32, strided<[1], offset: ?>> to memref<1600xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_16 : (tensor<1600xf32>, memref<1600xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

