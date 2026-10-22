module attributes {pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"} {
  func.func @predicate_reload_only(%arg0: !pto.ptr<ui8, ub>) -> !pto.vmi.mask<64xpred> {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = arith.addi %c0_0, %c0 : index
    %1 = pto.addptr %arg0, %0 : <ui8, ub> -> <ui8, ub>
    %2 = pto.castptr %1 : !pto.ptr<ui8, ub> -> !pto.ptr<ui32, ub>
    %3 = pto.castptr %2 : !pto.ptr<ui32, ub> -> !pto.ptr<ui8, ub>
    %c0_1 = arith.constant 0 : index
    %4 = pto.vmi.load %3[%c0_1] : !pto.ptr<ui8, ub> -> !pto.vmi.vreg<64xui8>
    %5 = "pto.vmi.constant"() <{value = dense<0> : tensor<64xui8>}> : () -> !pto.vmi.vreg<64xui8>
    %6 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
    %7 = pto.vmi.vcmp %4, %5, %6 {cmp = "ne"} : !pto.vmi.vreg<64xui8>, !pto.vmi.vreg<64xui8>, !pto.vmi.mask<64xpred> -> !pto.vmi.mask<64xpred>
    return %7 : !pto.vmi.mask<64xpred>
  }
}

