#map = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @predicate_reload_only(%reinterpret_cast_4: memref<64xi1, #map, #hivm.address_space<ub>>) -> vector<64xi1> {
    %c0 = arith.constant 0 : index
    %res = ave.hir.vload <NORM> %reinterpret_cast_4[%c0] {functionType = #ave.func_dist_type<pb32>} : memref<64xi1, #map, #hivm.address_space<ub>> into vector<64xi1>
    return %res : vector<64xi1>
  }
}
