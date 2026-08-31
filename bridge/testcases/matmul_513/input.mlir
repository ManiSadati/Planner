module {
  func.func @matmul513_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c64 = arith.constant 64 : index
    %c513 = arith.constant 513 : index
    %c9_i32 = arith.constant 9 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = arith.divsi %arg8, %c9_i32 : i32
    %3 = arith.remsi %arg8, %c9_i32 : i32
    %4 = arith.muli %2, %c64_i32 : i32
    %5 = arith.muli %3, %c64_i32 : i32
    %6 = arith.index_cast %4 : i32 to index
    %7 = arith.muli %6, %c513 : index
    %8 = scf.for %arg11 = %c0_i32 to %c9_i32 step %c1_i32 iter_args(%arg12 = %1) -> (tensor<64x64xf32>)  : i32 {
      %22 = arith.muli %arg11, %c64_i32 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = arith.addi %7, %23 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [513, 1] : memref<?xf16> to memref<64x64xf16, strided<[513, 1], offset: ?>>
      %25 = arith.muli %23, %c513 : index
      %26 = arith.index_cast %5 : i32 to index
      %27 = arith.addi %25, %26 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [64, 64], strides: [513, 1] : memref<?xf16> to memref<64x64xf16, strided<[513, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x64xf16>
      %28 = arith.addi %6, %c64 : index
      %29 = arith.maxsi %6, %c513 : index
      %30 = arith.minsi %28, %29 : index
      %31 = arith.subi %30, %6 : index
      %32 = arith.addi %23, %c64 : index
      %33 = arith.maxsi %23, %c513 : index
      %34 = arith.minsi %32, %33 : index
      %35 = arith.subi %34, %23 : index
      %36 = arith.minsi %31, %c64 : index
      %37 = arith.minsi %35, %c64 : index
      %38 = arith.cmpi slt, %36, %c64 : index
      %39 = arith.cmpi slt, %37, %c64 : index
      %40 = arith.ori %38, %39 : i1
      scf.if %40 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<64x64xf16>)
      } {hivm.unlikely_condition}
      %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%36, %37] [1, 1] : memref<64x64xf16, strided<[513, 1], offset: ?>> to memref<?x?xf16, strided<[513, 1], offset: ?>>
      %subview_4 = memref.subview %alloc[0, 0] [%36, %37] [1, 1] : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1]>>
      memref.copy %subview_3, %subview_4 : memref<?x?xf16, strided<[513, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1]>>
      %41 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
      %alloc_5 = memref.alloc() : memref<64x64xf16>
      %42 = arith.addi %26, %c64 : index
      %43 = arith.maxsi %26, %c513 : index
      %44 = arith.minsi %42, %43 : index
      %45 = arith.subi %44, %26 : index
      %46 = arith.minsi %45, %c64 : index
      %47 = arith.cmpi slt, %46, %c64 : index
      %48 = arith.ori %39, %47 : i1
      scf.if %48 {
        linalg.fill ins(%cst : f16) outs(%alloc_5 : memref<64x64xf16>)
      } {hivm.unlikely_condition}
      %subview_6 = memref.subview %reinterpret_cast_2[0, 0] [%37, %46] [1, 1] : memref<64x64xf16, strided<[513, 1], offset: ?>> to memref<?x?xf16, strided<[513, 1], offset: ?>>
      %subview_7 = memref.subview %alloc_5[0, 0] [%37, %46] [1, 1] : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1]>>
      memref.copy %subview_6, %subview_7 : memref<?x?xf16, strided<[513, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1]>>
      %49 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x64xf16>
      %50 = linalg.matmul {input_precison = "ieee"} ins(%41, %49 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg12 : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %50 : tensor<64x64xf32>
    }
    %9 = arith.index_cast %5 : i32 to index
    %10 = arith.addi %7, %9 : index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%10], sizes: [64, 64], strides: [513, 1] : memref<?xf16> to memref<64x64xf16, strided<[513, 1], offset: ?>>
    %11 = arith.truncf %8 : tensor<64x64xf32> to tensor<64x64xf16>
    %12 = arith.addi %6, %c64 : index
    %13 = arith.maxsi %6, %c513 : index
    %14 = arith.minsi %12, %13 : index
    %15 = arith.subi %14, %6 : index
    %16 = arith.addi %9, %c64 : index
    %17 = arith.maxsi %9, %c513 : index
    %18 = arith.minsi %16, %17 : index
    %19 = arith.subi %18, %9 : index
    %20 = arith.minsi %15, %c64 : index
    %21 = arith.minsi %19, %c64 : index
    %extracted_slice = tensor.extract_slice %11[0, 0] [%20, %21] [1, 1] : tensor<64x64xf16> to tensor<?x?xf16>
    %subview = memref.subview %reinterpret_cast[0, 0] [%20, %21] [1, 1] : memref<64x64xf16, strided<[513, 1], offset: ?>> to memref<?x?xf16, strided<[513, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf16>, memref<?x?xf16, strided<[513, 1], offset: ?>>) -> ()
    return
  }
}

