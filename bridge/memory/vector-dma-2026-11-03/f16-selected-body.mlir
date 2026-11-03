  func.func private @__pto_load_gm_to_ubuf_2d_half(%arg0: !pto.ptr<f16, gm>, %arg1: !pto.ptr<f16, ub>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: f16, %arg10: i64, %arg11: i64) attributes {pto.ptodsl.callable_kind = "func", pto.ptodsl.instantiation = "load_gm_to_ubuf_2d_half", pto.ptodsl.logical_name = "load_gm_to_ubuf_2d_half", pto.ptodsl.source = "load_gm_to_ubuf_2d_half.mlir"} {
    %c2_i64 = arith.constant 2 : i64
    %0 = arith.muli %arg5, %c2_i64 : i64
    %1 = arith.muli %arg7, %c2_i64 : i64
    %2 = arith.muli %arg4, %c2_i64 : i64
    %3 = arith.muli %arg6, %c2_i64 : i64
    pto.mte_gm_ub %arg0, %arg1, %arg11, %c2_i64 nburst(%arg3, %0, %1) loop(%arg2, %2, %3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>} : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64, loop i64, i64, i64
    return
  }
}
