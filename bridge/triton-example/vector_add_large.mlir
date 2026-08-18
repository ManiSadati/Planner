// -----// IR Dump After AppendTargetDeviceSpec (hacc-append-device-spec) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @vector_add_large_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c2000_i32 = arith.constant 2000 : i32
    %0 = arith.muli %arg8, %c2000_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<2048xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<2048xf32>)
    %subview = memref.subview %reinterpret_cast[0] [2000] [1] : memref<2048xf32, strided<[1], offset: ?>> to memref<2000xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [2000] [1] : memref<2048xf32> to memref<2000xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<2000xf32, strided<[1], offset: ?>> to memref<2000xf32, strided<[1]>>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<2048xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<2048xf32>
    linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<2048xf32>)
    %subview_3 = memref.subview %reinterpret_cast_1[0] [2000] [1] : memref<2048xf32, strided<[1], offset: ?>> to memref<2000xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [2000] [1] : memref<2048xf32> to memref<2000xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<2000xf32, strided<[1], offset: ?>> to memref<2000xf32, strided<[1]>>
    %3 = bufferization.to_tensor %alloc_2 restrict writable : memref<2048xf32>
    %4 = arith.addf %2, %3 : tensor<2048xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %4[0] [2000] [1] : tensor<2048xf32> to tensor<2000xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [2000] [1] : memref<2048xf32, strided<[1], offset: ?>> to memref<2000xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<2000xf32>, memref<2000xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}


