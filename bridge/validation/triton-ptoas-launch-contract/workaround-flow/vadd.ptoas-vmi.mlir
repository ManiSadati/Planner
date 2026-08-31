module attributes {pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"} {
  func.func @vector_add_kernel_outlined_vf_0(%arg0: !pto.ptr<f32, ub>, %arg1: !pto.ptr<f32, ub>, %arg2: !pto.ptr<f32, ub>) attributes {no_inline} {
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c256 step %c64 {
      %c0_0 = arith.constant 0 : index
      %c256_1 = arith.constant 256 : index
      %c1 = arith.constant 1 : index
      %0 = pto.addptr %arg0, %arg3 : <f32, ub> -> <f32, ub>
      %c0_2 = arith.constant 0 : index
      %c256_3 = arith.constant 256 : index
      %c1_4 = arith.constant 1 : index
      %1 = pto.addptr %arg1, %arg3 : <f32, ub> -> <f32, ub>
      %c0_5 = arith.constant 0 : index
      %c256_6 = arith.constant 256 : index
      %c1_7 = arith.constant 1 : index
      %2 = pto.addptr %arg2, %arg3 : <f32, ub> -> <f32, ub>
      %3 = pto.vmi.load %0[%c0] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
      %4 = pto.vmi.load %1[%c0] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
      %5 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      %6 = pto.vmi.vadd %3, %4, %5 : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<64xf32>
      %7 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      pto.vmi.store %6, %2[%c0] : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>
    }
    return
  }
  func.func @vector_add_kernel(%arg0: !pto.ptr<ui8, gm>, %arg1: !pto.ptr<ui8, gm>, %arg2: !pto.ptr<f32, gm>, %arg3: !pto.ptr<f32, gm>, %arg4: !pto.ptr<f32, gm>, %arg5: i32, %arg6: i32, %arg7: i32) attributes {pto.kernel} {
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = pto.get_ctrl : i64
    %c-1152921504606846977_i64 = arith.constant -1152921504606846977 : i64
    %1 = arith.andi %0, %c-1152921504606846977_i64 : i64
    pto.set_ctrl %1 : i64
    %2 = pto.get_ctrl : i64
    %c281474976710656_i64 = arith.constant 281474976710656 : i64
    %3 = arith.ori %2, %c281474976710656_i64 : i64
    pto.set_ctrl %3 : i64
    %4 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
    pto.mte_gm_ub %arg2, %4, %c0_i64, %c1024_i64 nburst(%c1_i64, %c0_i64, %c0_i64) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0>} : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
    %5 = pto.castptr %c1024_i64 : i64 -> !pto.ptr<f32, ub>
    pto.mte_gm_ub %arg3, %5, %c0_i64, %c1024_i64 nburst(%c1_i64, %c0_i64, %c0_i64) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0>} : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
    pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %6 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
    pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    call @vector_add_kernel_outlined_vf_0(%4, %5, %6) : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>) -> ()
    pto.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    pto.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    pto.mte_ub_gm %6, %arg4, %c1024_i64 nburst(%c1_i64, %c0_i64, %c0_i64) l2_cache_ctl(%c0_i64) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>} : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64
    %7 = pto.get_ctrl : i64
    %c1152921504606846976_i64 = arith.constant 1152921504606846976 : i64
    %8 = arith.ori %7, %c1152921504606846976_i64 : i64
    pto.set_ctrl %8 : i64
    pto.barrier <PIPE_ALL>
    return
  }
}

