#map = affine_map<(d0) -> (d0)>
module {
  func.func @flash_atten_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant 1.250000e-01 : f32
    %cst_2 = arith.constant -1.000000e+09 : f32
    %c16384_i32 = arith.constant 16384 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_3 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %5 = tensor.empty() : tensor<1x64xi32>
    %6 = linalg.fill ins(%c256_i32 : i32) outs(%5 : tensor<1x64xi32>) -> tensor<1x64xi32>
    %7 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %8 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %9 = tensor.empty() : tensor<64x1xi32>
    %10 = linalg.fill ins(%c256_i32 : i32) outs(%9 : tensor<64x1xi32>) -> tensor<64x1xi32>
    %11 = arith.divsi %arg9, %c4_i32 : i32
    %12 = arith.remsi %arg9, %c4_i32 : i32
    %13 = arith.divsi %11, %c2_i32 : i32
    %14 = arith.muli %12, %c64_i32 : i32
    %15 = tensor.empty() : tensor<64xi32>
    %16 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%15 : tensor<64xi32>) {
    ^bb0(%out: i32):
      %38 = linalg.index 0 : index
      %39 = arith.index_cast %38 : index to i32
      linalg.yield %39 : i32
    } -> tensor<64xi32>
    %17 = linalg.fill ins(%14 : i32) outs(%15 : tensor<64xi32>) -> tensor<64xi32>
    %18 = arith.addi %17, %16 : tensor<64xi32>
    %expanded = tensor.expand_shape %18 [[0, 1]] output_shape [64, 1] : tensor<64xi32> into tensor<64x1xi32>
    %19 = arith.cmpi slt, %expanded, %10 : tensor<64x1xi32>
    %20 = arith.muli %11, %c16384_i32 : i32
    %21 = arith.index_cast %20 : i32 to index
    %22 = arith.index_cast %14 : i32 to index
    %23 = arith.muli %22, %c64 : index
    %24 = arith.addi %21, %23 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %25 = tensor.empty() : tensor<64x64xi1>
    %collapsed = tensor.collapse_shape %19 [[0, 1]] : tensor<64x1xi1> into tensor<64xi1>
    %broadcasted = linalg.broadcast ins(%collapsed : tensor<64xi1>) outs(%25 : tensor<64x64xi1>) dimensions = [1] 
    %alloc = memref.alloc() : memref<64x64xf16>
    %26 = arith.addi %22, %c64 : index
    %27 = arith.maxsi %22, %c256 : index
    %28 = arith.minsi %26, %27 : index
    %29 = arith.subi %28, %22 : index
    %30 = arith.cmpi slt, %29, %c64 : index
    scf.if %30 {
      linalg.fill ins(%cst_0 : f16) outs(%alloc : memref<64x64xf16>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0, 0] [%29, 64] [1, 1] : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
    %subview_4 = memref.subview %alloc[0, 0] [%29, 64] [1, 1] : memref<64x64xf16> to memref<?x64xf16, strided<[64, 1]>>
    memref.copy %subview, %subview_4 : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>
    %31 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
    %32 = arith.muli %13, %c16384_i32 : i32
    %33 = arith.index_cast %32 : i32 to index
    %34 = tensor.empty() : tensor<64x64xi32>
    %broadcasted_5 = linalg.broadcast ins(%18 : tensor<64xi32>) outs(%34 : tensor<64x64xi32>) dimensions = [1] 
    %35:3 = scf.for %arg12 = %c0_i32 to %c256_i32 step %c64_i32 iter_args(%arg13 = %8, %arg14 = %1, %arg15 = %7) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)  : i32 {
      %38 = linalg.fill ins(%arg12 : i32) outs(%15 : tensor<64xi32>) -> tensor<64xi32>
      %39 = arith.addi %38, %16 : tensor<64xi32>
      %expanded_9 = tensor.expand_shape %39 [[0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
      %40 = arith.index_cast %arg12 : i32 to index
      %41 = arith.muli %40, %c64 : index
      %42 = arith.addi %33, %41 : index
      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [64, 64], strides: [1, 64] : memref<?xf16> to memref<64x64xf16, strided<[1, 64], offset: ?>>
      %43 = arith.cmpi slt, %expanded_9, %6 : tensor<1x64xi32>
      %collapsed_11 = tensor.collapse_shape %43 [[0, 1]] : tensor<1x64xi1> into tensor<64xi1>
      %broadcasted_12 = linalg.broadcast ins(%collapsed_11 : tensor<64xi1>) outs(%25 : tensor<64x64xi1>) dimensions = [0] 
      %alloc_13 = memref.alloc() : memref<64x64xf16>
      %44 = arith.addi %40, %c64 : index
      %45 = arith.maxsi %40, %c256 : index
      %46 = arith.minsi %44, %45 : index
      %47 = arith.subi %46, %40 : index
      %48 = arith.cmpi slt, %47, %c64 : index
      scf.if %48 {
        linalg.fill ins(%cst_0 : f16) outs(%alloc_13 : memref<64x64xf16>)
      } {hivm.unlikely_condition}
      %subview_14 = memref.subview %reinterpret_cast_10[0, 0] [64, %47] [1, 1] : memref<64x64xf16, strided<[1, 64], offset: ?>> to memref<64x?xf16, strided<[1, 64], offset: ?>>
      %subview_15 = memref.subview %alloc_13[0, 0] [64, %47] [1, 1] : memref<64x64xf16> to memref<64x?xf16, strided<[64, 1]>>
      memref.copy %subview_14, %subview_15 : memref<64x?xf16, strided<[1, 64], offset: ?>> to memref<64x?xf16, strided<[64, 1]>>
      annotation.mark %alloc_13 {MayImplicitTransposeWithLastAxis} : memref<64x64xf16>
      %49 = bufferization.to_tensor %alloc_13 restrict writable : memref<64x64xf16>
      annotation.mark %49 {MayImplicitTransposeWithLastAxis} : tensor<64x64xf16>
      %50 = linalg.matmul {input_precison = "ieee"} ins(%31, %49 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %51 = arith.mulf %50, %4 : tensor<64x64xf32>
      %52 = arith.andi %broadcasted, %broadcasted_12 : tensor<64x64xi1>
      %broadcasted_16 = linalg.broadcast ins(%39 : tensor<64xi32>) outs(%34 : tensor<64x64xi32>) dimensions = [0] 
      %53 = arith.cmpi sle, %broadcasted_16, %broadcasted_5 : tensor<64x64xi32>
      %54 = arith.andi %52, %53 : tensor<64x64xi1>
      %55 = arith.select %54, %51, %3 : tensor<64x64xi1>, tensor<64x64xf32>
      %56 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
      %reduced = linalg.reduce ins(%55 : tensor<64x64xf32>) outs(%56 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %68 = arith.maxnumf %in, %init : f32
          linalg.yield %68 : f32
        }
      %57 = arith.maxnumf %arg15, %reduced : tensor<64xf32>
      %58 = arith.subf %arg15, %57 : tensor<64xf32>
      %59 = math.exp %58 : tensor<64xf32>
      %broadcasted_17 = linalg.broadcast ins(%57 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %60 = arith.subf %55, %broadcasted_17 : tensor<64x64xf32>
      %61 = math.exp %60 : tensor<64x64xf32>
      %reduced_18 = linalg.reduce ins(%61 : tensor<64x64xf32>) outs(%1 : tensor<64xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %68 = arith.addf %in, %init : f32
          linalg.yield %68 : f32
        }
      %reinterpret_cast_19 = memref.reinterpret_cast %arg4 to offset: [%42], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %alloc_20 = memref.alloc() : memref<64x64xf16>
      scf.if %48 {
        linalg.fill ins(%cst_0 : f16) outs(%alloc_20 : memref<64x64xf16>)
      } {hivm.unlikely_condition}
      %subview_21 = memref.subview %reinterpret_cast_19[0, 0] [%47, 64] [1, 1] : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %subview_22 = memref.subview %alloc_20[0, 0] [%47, 64] [1, 1] : memref<64x64xf16> to memref<?x64xf16, strided<[64, 1]>>
      memref.copy %subview_21, %subview_22 : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>
      %62 = bufferization.to_tensor %alloc_20 restrict writable : memref<64x64xf16>
      %broadcasted_23 = linalg.broadcast ins(%59 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
      %63 = arith.mulf %arg13, %broadcasted_23 : tensor<64x64xf32>
      %64 = arith.truncf %61 : tensor<64x64xf32> to tensor<64x64xf16>
      %65 = linalg.matmul {input_precison = "ieee"} ins(%64, %62 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%63 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %66 = arith.mulf %arg14, %59 : tensor<64xf32>
      %67 = arith.addf %66, %reduced_18 : tensor<64xf32>
      scf.yield %65, %67, %57 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>
    }
    %broadcasted_6 = linalg.broadcast ins(%35#1 : tensor<64xf32>) outs(%2 : tensor<64x64xf32>) dimensions = [1] 
    %36 = arith.divf %35#0, %broadcasted_6 : tensor<64x64xf32>
    %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%24], sizes: [64, 64], strides: [64, 1] : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %37 = arith.truncf %36 : tensor<64x64xf32> to tensor<64x64xf16>
    %extracted_slice = tensor.extract_slice %37[0, 0] [%29, 64] [1, 1] : tensor<64x64xf16> to tensor<?x64xf16>
    %subview_8 = memref.subview %reinterpret_cast_7[0, 0] [%29, 64] [1, 1] : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16, strided<[64, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_8 : (tensor<?x64xf16>, memref<?x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}

