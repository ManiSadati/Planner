module {
  func.func @q_kt_matmul_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f16
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c8192 = arith.constant 8192 : index
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
    %16 = scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg12 = %1) -> (tensor<64x64xf32>)  : i32 {
      %34 = arith.muli %arg11, %c64_i32 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = arith.addi %13, %35 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [64, 64], strides: [256, 1] : memref<?xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
      %37 = arith.muli %35, %c8192 : index
      %38 = arith.addi %15, %37 : index
      %39 = arith.index_cast %8 : i32 to index
      %40 = arith.addi %38, %39 : index
      %reinterpret_cast_2 = memref.reinterpret_cast %arg3 to offset: [%40], sizes: [64, 64], strides: [8192, 1] : memref<?xf16> to memref<64x64xf16, strided<[8192, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x64xf16>
      %41 = arith.addi %11, %c64 : index
      %42 = arith.maxsi %11, %c128 : index
      %43 = arith.minsi %41, %42 : index
      %44 = arith.subi %43, %11 : index
      %45 = arith.addi %35, %c64 : index
      %46 = arith.maxsi %35, %c256 : index
      %47 = arith.minsi %45, %46 : index
      %48 = arith.subi %47, %35 : index
      %49 = arith.minsi %44, %c64 : index
      %50 = arith.minsi %48, %c64 : index
      %51 = arith.cmpi slt, %49, %c64 : index
      %52 = arith.cmpi slt, %50, %c64 : index
      %53 = arith.ori %51, %52 : i1
      scf.if %53 {
        linalg.fill ins(%cst : f16) outs(%alloc : memref<64x64xf16>)
      } {hivm.unlikely_condition}
      %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [%49, %50] [1, 1] : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      %subview_4 = memref.subview %alloc[0, 0] [%49, %50] [1, 1] : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1]>>
      memref.copy %subview_3, %subview_4 : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1]>>
      %54 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16>
      %alloc_5 = memref.alloc() : memref<64x64xf16>
      %55 = arith.addi %39, %c64 : index
      %56 = arith.maxsi %39, %c8192 : index
      %57 = arith.minsi %55, %56 : index
      %58 = arith.subi %57, %39 : index
      %59 = arith.minsi %58, %c64 : index
      %60 = arith.cmpi slt, %59, %c64 : index
      %61 = arith.ori %52, %60 : i1
      scf.if %61 {
        linalg.fill ins(%cst : f16) outs(%alloc_5 : memref<64x64xf16>)
      } {hivm.unlikely_condition}
      %subview_6 = memref.subview %reinterpret_cast_2[0, 0] [%50, %59] [1, 1] : memref<64x64xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[8192, 1], offset: ?>>
      %subview_7 = memref.subview %alloc_5[0, 0] [%50, %59] [1, 1] : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1]>>
      memref.copy %subview_6, %subview_7 : memref<?x?xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1]>>
      %62 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x64xf16>
      %63 = linalg.matmul {input_precison = "ieee"} ins(%54, %62 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg12 : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %63 : tensor<64x64xf32>
    }
    %17 = arith.muli %2, %c1048576_i32 : i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.muli %11, %c8192 : index
    %20 = arith.addi %18, %19 : index
    %21 = arith.index_cast %8 : i32 to index
    %22 = arith.addi %20, %21 : index
    %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%22], sizes: [64, 64], strides: [8192, 1] : memref<?xf16> to memref<64x64xf16, strided<[8192, 1], offset: ?>>
    %23 = arith.truncf %16 : tensor<64x64xf32> to tensor<64x64xf16>
    %24 = arith.addi %11, %c64 : index
    %25 = arith.maxsi %11, %c128 : index
    %26 = arith.minsi %24, %25 : index
    %27 = arith.subi %26, %11 : index
    %28 = arith.addi %21, %c64 : index
    %29 = arith.maxsi %21, %c8192 : index
    %30 = arith.minsi %28, %29 : index
    %31 = arith.subi %30, %21 : index
    %32 = arith.minsi %27, %c64 : index
    %33 = arith.minsi %31, %c64 : index
    %extracted_slice = tensor.extract_slice %23[0, 0] [%32, %33] [1, 1] : tensor<64x64xf16> to tensor<?x?xf16>
    %subview = memref.subview %reinterpret_cast[0, 0] [%32, %33] [1, 1] : memref<64x64xf16, strided<[8192, 1], offset: ?>> to memref<?x?xf16, strided<[8192, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf16>, memref<?x?xf16, strided<[8192, 1], offset: ?>>) -> ()
    return
  }
}

