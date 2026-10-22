module {
  func.func @numeric_memory(%arg0: !pto.ptr<ui8, ub>, %arg1: !pto.ptr<ui8, ub>) {
    %c0 = arith.constant 0 : index
    %0 = pto.vmi.vload %arg0[%c0] : !pto.ptr<ui8, ub> -> !pto.vmi.vreg<64xui8>
    pto.vmi.vstore %0, %arg1[%c0] : !pto.vmi.vreg<64xui8>, !pto.ptr<ui8, ub>
    return
  }
}

