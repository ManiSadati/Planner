#map = affine_map<(d0) -> (d0)>
module {
  func.func @flash_atten_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0xFF800000 : f32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant 1.250000e-01 : f32
    %cst_1 = arith.constant -1.000000e+09 : f32
    %c4096_i32 = arith.constant 4096 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x1xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<64x1xf32>) -> tensor<64x1xf32>
    %2 = tensor.empty() : tensor<64xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %7 = tensor.empty() : tensor<1x64xi32>
    %8 = linalg.fill ins(%c64_i32 : i32) outs(%7 : tensor<1x64xi32>) -> tensor<1x64xi32>
    %9 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %10 = linalg.fill ins(%cst_2 : f32) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %11 = tensor.empty() : tensor<64x1xi32>
    %12 = linalg.fill ins(%c64_i32 : i32) outs(%11 : tensor<64x1xi32>) -> tensor<64x1xi32>
    %13 = tensor.empty() : tensor<64xi32>
    %14 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%13 : tensor<64xi32>) {
    ^bb0(%out: i32):
      %43 = linalg.index 0 : index
      %44 = arith.index_cast %43 : index to i32
      linalg.yield %44 : i32
    } -> tensor<64xi32>
    %expanded = tensor.expand_shape %14 [[0, 1]] output_shape [64, 1] : tensor<64xi32> into tensor<64x1xi32>
    %15 = arith.cmpi slt, %expanded, %12 : tensor<64x1xi32>
    %16 = arith.muli %arg9, %c4096_i32 : i32
    %17 = arith.index_cast %16 : i32 to index
    %expanded_3 = tensor.expand_shape %14 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
    %18 = tensor.empty() : tensor<64x64xi32>
    %broadcasted = linalg.broadcast ins(%14 : tensor<64xi32>) outs(%18 : tensor<64x64xi32>) dimensions = [0] 
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%17], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %19 = tensor.empty() : tensor<64x64xi1>
    %collapsed = tensor.collapse_shape %15 [[0, 1]] : tensor<64x1xi1> into tensor<64xi1>
    %broadcasted_4 = linalg.broadcast ins(%collapsed : tensor<64xi1>) outs(%19 : tensor<64x64xi1>) dimensions = [1] 
    %alloc = memref.alloc() : memref<64x64xf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
    %20 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
    %broadcasted_5 = linalg.broadcast ins(%14 : tensor<64xi32>) outs(%18 : tensor<64x64xi32>) dimensions = [1] 
    %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%17], sizes: [64, 64], strides: [1, 64] : memref<?xf16> to memref<64x64xf16, strided<[1, 64], offset: ?>>
    %21 = arith.cmpi slt, %expanded_3, %8 : tensor<1x64xi32>
    %collapsed_7 = tensor.collapse_shape %21 [[0, 1]] : tensor<1x64xi1> into tensor<64xi1>
    %broadcasted_8 = linalg.broadcast ins(%collapsed_7 : tensor<64xi1>) outs(%19 : tensor<64x64xi1>) dimensions = [0] 
    %alloc_9 = memref.alloc() : memref<64x64xf16>
    memref.copy %reinterpret_cast_6, %alloc_9 : memref<64x64xf16, strided<[1, 64], offset: ?>> to memref<64x64xf16>
    annotation.mark %alloc_9 {MayImplicitTransposeWithLastAxis} : memref<64x64xf16>
    %22 = bufferization.to_tensor %alloc_9 restrict writable : memref<64x64xf16>
    annotation.mark %22 {MayImplicitTransposeWithLastAxis} : tensor<64x64xf16>
    %23 = linalg.matmul {input_precison = "ieee"} ins(%20, %22 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %24 = arith.mulf %23, %6 : tensor<64x64xf32>
    %25 = arith.andi %broadcasted_4, %broadcasted_8 : tensor<64x64xi1>
    %26 = arith.cmpi sle, %broadcasted, %broadcasted_5 : tensor<64x64xi32>
    %27 = arith.andi %25, %26 : tensor<64x64xi1>
    %28 = arith.select %27, %24, %5 : tensor<64x64xi1>, tensor<64x64xf32>
    %29 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
    %reduced = linalg.reduce ins(%28 : tensor<64x64xf32>) outs(%29 : tensor<64xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %43 = arith.maxnumf %in, %init : f32
        linalg.yield %43 : f32
      }
    %30 = arith.maxnumf %reduced, %9 : tensor<64xf32>
    %31 = arith.subf %9, %30 : tensor<64xf32>
    %32 = math.exp %31 : tensor<64xf32>
    %broadcasted_10 = linalg.broadcast ins(%30 : tensor<64xf32>) outs(%4 : tensor<64x64xf32>) dimensions = [1] 
    %33 = arith.subf %28, %broadcasted_10 : tensor<64x64xf32>
    %34 = math.exp %33 : tensor<64x64xf32>
    %reduced_11 = linalg.reduce ins(%34 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [1] 
      (%in: f32, %init: f32) {
        %43 = arith.addf %in, %init : f32
        linalg.yield %43 : f32
      }
    %reinterpret_cast_12 = memref.reinterpret_cast %arg4 to offset: [%17], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %alloc_13 = memref.alloc() : memref<64x64xf16>
    memref.copy %reinterpret_cast_12, %alloc_13 : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
    %35 = bufferization.to_tensor %alloc_13 restrict writable : memref<64x64xf16>
    %expanded_14 = tensor.expand_shape %32 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %36 = arith.mulf %expanded_14, %1 : tensor<64x1xf32>
    %collapsed_15 = tensor.collapse_shape %36 [[0, 1]] : tensor<64x1xf32> into tensor<64xf32>
    %broadcasted_16 = linalg.broadcast ins(%collapsed_15 : tensor<64xf32>) outs(%4 : tensor<64x64xf32>) dimensions = [1] 
    %37 = arith.truncf %34 : tensor<64x64xf32> to tensor<64x64xf16>
    %38 = linalg.matmul {input_precison = "ieee"} ins(%37, %35 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%broadcasted_16 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %39 = arith.mulf %32, %3 : tensor<64xf32>
    %40 = arith.addf %39, %reduced_11 : tensor<64xf32>
    %broadcasted_17 = linalg.broadcast ins(%40 : tensor<64xf32>) outs(%4 : tensor<64x64xf32>) dimensions = [1] 
    %41 = arith.divf %38, %broadcasted_17 : tensor<64x64xf32>
    %reinterpret_cast_18 = memref.reinterpret_cast %arg5 to offset: [%17], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %42 = arith.truncf %41 : tensor<64x64xf32> to tensor<64x64xf16>
    bufferization.materialize_in_destination %42 in writable %reinterpret_cast_18 : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}

