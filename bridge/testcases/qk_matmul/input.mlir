module {
  func.func @qk_matmul_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c8192 = arith.constant 8192 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c128_i32 = arith.constant 128 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1048576_i32 = arith.constant 1048576 : i32
    %c2097152_i32 = arith.constant 2097152 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = arith.divsi %arg8, %c256_i32 : i32
    %3 = arith.remsi %arg8, %c256_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.remsi %3, %c128_i32 : i32
    %6 = arith.divsi %2, %c16_i32 : i32
    %7 = arith.muli %4, %c64_i32 : i32
    %8 = arith.muli %5, %c64_i32 : i32
    %9 = arith.muli %2, %c32768_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %11 = arith.index_cast %7 : i32 to index
    %12 = arith.muli %11, %c256 : index
    %13 = arith.addi %10, %12 : index
    %14 = arith.muli %6, %c2097152_i32 : i32
    %15 = arith.index_cast %14 : i32 to index
    %16 = arith.index_cast %8 : i32 to index
    %17 = arith.muli %16, %c256 : index
    %18 = scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg12 = %1) -> (tensor<64x64xf32>)  : i32 {
      %35 = arith.muli %arg11, %c64_i32 : i32
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.addi %13, %36 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%37], sizes: [64, 64], strides: [256, 1] : memref<?xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
      %38 = arith.addi %15, %36 : index
      %39 = arith.addi %38, %17 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [64, 64], strides: [1, 256] : memref<?xf16> to memref<64x64xf16, strided<[1, 256], offset: ?>>
      %alloc = memref.alloc() : memref<64x64xf16>
      %40 = arith.addi %11, %c64 : index
      %41 = arith.maxsi %11, %c128 : index
      %42 = arith.minsi %40, %41 : index
      %43 = arith.subi %42, %11 : index
      %44 = arith.addi %36, %c64 : index
      %45 = arith.maxsi %36, %c256 : index
      %46 = arith.minsi %44, %45 : index
      %47 = arith.subi %46, %36 : index
      %48 = arith.minsi %43, %c64 : index
      %49 = arith.minsi %47, %c64 : index
      %50 = arith.cmpi slt, %48, %c64 : index
      %51 = arith.cmpi slt, %49, %c64 : index
      %52 = arith.ori %50, %51 : i1
      scf.if %52 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<64x64xf16>)
      } {hivm.unlikely_condition}
      %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%48, %49] [1, 1] : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      %subview_4 = memref.subview %alloc[0, 0] [%48, %49] [1, 1] : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1]>>
      memref.copy %subview_3, %subview_4 : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1]>>
      %53 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
      %alloc_5 = memref.alloc() : memref<64x64xf16>
      %54 = arith.addi %16, %c64 : index
      %55 = arith.maxsi %16, %c8192 : index
      %56 = arith.minsi %54, %55 : index
      %57 = arith.subi %56, %16 : index
      %58 = arith.minsi %57, %c64 : index
      %59 = arith.cmpi slt, %58, %c64 : index
      %60 = arith.ori %51, %59 : i1
      scf.if %60 {
        linalg.fill ins(%cst : f16) outs(%alloc_5 : memref<64x64xf16>)
      } {hivm.unlikely_condition}
      %subview_6 = memref.subview %reinterpret_cast_2[0, 0] [%49, %58] [1, 1] : memref<64x64xf16, strided<[1, 256], offset: ?>> to memref<?x?xf16, strided<[1, 256], offset: ?>>
      %subview_7 = memref.subview %alloc_5[0, 0] [%49, %58] [1, 1] : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1]>>
      memref.copy %subview_6, %subview_7 : memref<?x?xf16, strided<[1, 256], offset: ?>> to memref<?x?xf16, strided<[64, 1]>>
      annotation.mark %alloc_5 {MayImplicitTransposeWithLastAxis} : memref<64x64xf16>
      %61 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x64xf16>
      annotation.mark %61 {MayImplicitTransposeWithLastAxis} : tensor<64x64xf16>
      %62 = linalg.matmul {input_precison = "ieee"} ins(%53, %61 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg12 : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %62 : tensor<64x64xf32>
    }
    %19 = arith.muli %2, %c1048576_i32 : i32
    %20 = arith.index_cast %19 : i32 to index
    %21 = arith.muli %11, %c8192 : index
    %22 = arith.addi %20, %21 : index
    %23 = arith.addi %22, %16 : index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%23], sizes: [64, 64], strides: [8192, 1] : memref<?xf16> to memref<64x64xf16, strided<[8192, 1], offset: ?>>
    %24 = arith.truncf %18 : tensor<64x64xf32> to tensor<64x64xf16>
    %25 = arith.addi %11, %c64 : index
    %26 = arith.maxsi %11, %c128 : index
    %27 = arith.minsi %25, %26 : index
    %28 = arith.subi %27, %11 : index
    %29 = arith.addi %16, %c64 : index
    %30 = arith.maxsi %16, %c8192 : index
    %31 = arith.minsi %29, %30 : index
    %32 = arith.subi %31, %16 : index
    %33 = arith.minsi %28, %c64 : index
    %34 = arith.minsi %32, %c64 : index
    %extracted_slice = tensor.extract_slice %24[0, 0] [%33, %34] [1, 1] : tensor<64x64xf16> to tensor<?x?xf16>
    %subview = memref.subview %reinterpret_cast[0, 0] [%33, %34] [1, 1] : memref<64x64xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[8192, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf16>, memref<?x?xf16, strided<[8192, 1], offset: ?>>) -> ()
    return
  }
}

