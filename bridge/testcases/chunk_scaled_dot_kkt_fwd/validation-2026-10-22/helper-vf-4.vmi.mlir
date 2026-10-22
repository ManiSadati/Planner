module attributes {pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"} {
  func.func @chunk_scaled_dot_kkt_fwd_kernel_mix_aiv_outlined_vf_4(%arg0: index, %arg1: !pto.ptr<f32, ub>, %arg2: !pto.ptr<bf16, ub>, %arg3: !pto.ptr<i32, ub>, %arg4: !pto.ptr<f16, ub>, %arg5: !pto.ptr<f32, ub>, %arg6: index, %arg7: !pto.ptr<f32, ub>, %arg8: index, %arg9: !pto.ptr<ui8, ub>, %arg10: index) attributes {no_inline} {
    %c256 = arith.constant 256 : index
    %c8192 = arith.constant 8192 : index
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
    %1 = "pto.vmi.constant"() <{value = dense<0.000000e+00> : tensor<64xf16>}> : () -> !pto.vmi.vreg<64xf16>
    %2 = pto.vmi.pge "PAT_VL1" : !pto.vmi.mask<64xpred>
    %c0_0 = arith.constant 0 : index
    %3 = arith.addi %c0_0, %c0 : index
    %4 = pto.vmi.load %arg1[%3] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
    %c0_1 = arith.constant 0 : index
    %5 = arith.addi %c0_1, %c0 : index
    %6 = pto.vmi.load %arg3[%5] : !pto.ptr<i32, ub> -> !pto.vmi.vreg<64xi32>
    %c0_2 = arith.constant 0 : index
    %7 = arith.addi %c0_2, %c0 : index
    %8 = pto.vmi.load %arg4[%7] : !pto.ptr<f16, ub> -> !pto.vmi.vreg<64xf16>
    %9 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
    %10 = pto.vmi.vcmp %8, %1, %9 {cmp = "ne"} : !pto.vmi.vreg<64xf16>, !pto.vmi.vreg<64xf16>, !pto.vmi.mask<64xpred> -> !pto.vmi.mask<64xpred>
    scf.for %arg11 = %c0 to %c32 step %c1 {
      %c0_3 = arith.constant 0 : index
      %c64_4 = arith.constant 64 : index
      %c1_5 = arith.constant 1 : index
      %11 = arith.muli %arg0, %c32 : index
      %12 = arith.addi %arg11, %11 : index
      %13 = pto.addptr %arg1, %12 : <f32, ub> -> <f32, ub>
      %c0_6 = arith.constant 0 : index
      %c32_7 = arith.constant 32 : index
      %c1_8 = arith.constant 1 : index
      %c1_9 = arith.constant 1 : index
      %c1_10 = arith.constant 1 : index
      %14 = pto.addptr %arg2, %arg11 : <bf16, ub> -> <bf16, ub>
      %c0_11 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c32_12 = arith.constant 32 : index
      %c1_13 = arith.constant 1 : index
      %c32_14 = arith.constant 32 : index
      %c1_15 = arith.constant 1 : index
      %c1_16 = arith.constant 1 : index
      %15 = arith.muli %arg8, %c32 : index
      %16 = arith.addi %15, %arg11 : index
      %17 = pto.addptr %arg7, %16 : <f32, ub> -> <f32, ub>
      %c0_17 = arith.constant 0 : index
      %18 = arith.addi %c0_17, %c0 : index
      %19 = arith.addi %18, %c0 : index
      %20 = pto.vmi.vload %14[%19] {dist_mode = "brc"} : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<128xbf16>
      %21 = pto.vmi.shuffle %20[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] : !pto.vmi.vreg<128xbf16> -> !pto.vmi.vreg<64xbf16>
      %22 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      %23 = pto.vmi.extf %21 : !pto.vmi.vreg<64xbf16> -> !pto.vmi.vreg<64xf32>
      %c0_18 = arith.constant 0 : index
      %24 = arith.addi %c0_18, %c0 : index
      %25 = arith.addi %24, %c0 : index
      %26 = pto.vmi.shuffle %23[[0]] : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<1xf32>
      pto.vmi.vstore %26, %17[%25] : !pto.vmi.vreg<1xf32>, !pto.ptr<f32, ub>
      %c0_19 = arith.constant 0 : index
      %c64_20 = arith.constant 64 : index
      %c1_21 = arith.constant 1 : index
      %27 = arith.muli %arg0, %c32 : index
      %28 = arith.addi %arg11, %27 : index
      %29 = pto.addptr %arg3, %28 : <i32, ub> -> <i32, ub>
      %c0_22 = arith.constant 0 : index
      %c64_23 = arith.constant 64 : index
      %c1_24 = arith.constant 1 : index
      %30 = arith.muli %arg0, %c32 : index
      %31 = arith.addi %arg11, %30 : index
      %32 = pto.addptr %arg4, %31 : <f16, ub> -> <f16, ub>
      %33 = pto.vmi.vload %13[%c0] {dist_mode = "brc"} : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
      %34 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      %35 = pto.vmi.vexpdif %33, %4, %34 : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<64xf32>
      %c0_25 = arith.constant 0 : index
      %c2_26 = arith.constant 2 : index
      %c32_27 = arith.constant 32 : index
      %c64_28 = arith.constant 64 : index
      %c2048_29 = arith.constant 2048 : index
      %c64_30 = arith.constant 64 : index
      %c1_31 = arith.constant 1 : index
      %36 = arith.muli %arg6, %c2048 : index
      %37 = arith.muli %arg11, %c64 : index
      %38 = arith.addi %36, %37 : index
      %39 = pto.addptr %arg5, %38 : <f32, ub> -> <f32, ub>
      %40 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      %c0_32 = arith.constant 0 : index
      %41 = arith.addi %c0_32, %c0 : index
      pto.vmi.store %35, %39[%41] : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>
      %42 = pto.vmi.vload %29[%c0] {dist_mode = "brc"} : !pto.ptr<i32, ub> -> !pto.vmi.vreg<64xi32>
      %c0_33 = arith.constant 0 : index
      %43 = arith.addi %c0_33, %c0 : index
      %44 = pto.vmi.vload %32[%43] {dist_mode = "brc"} : !pto.ptr<f16, ub> -> !pto.vmi.vreg<128xf16>
      %45 = pto.vmi.shuffle %44[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<64xf16>
      %46 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      %47 = pto.vmi.vcmp %45, %1, %46 {cmp = "ne"} : !pto.vmi.vreg<64xf16>, !pto.vmi.vreg<64xf16>, !pto.vmi.mask<64xpred> -> !pto.vmi.mask<64xpred>
      %48 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      %49 = "pto.vmi.constant"() <{value = dense<0> : tensor<64xi32>}> : () -> !pto.vmi.vreg<64xi32>
      %50 = "pto.vmi.constant"() <{value = dense<1> : tensor<64xi32>}> : () -> !pto.vmi.vreg<64xi32>
      %51 = pto.vmi.vsel %47, %50, %49 : !pto.vmi.mask<64xpred>, !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi32>
      %52 = pto.vmi.vsel %10, %50, %49 : !pto.vmi.mask<64xpred>, !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi32>
      %53 = pto.vmi.vand %51, %52, %48 : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<64xi32>
      %54 = pto.vmi.vcmp %53, %49, %48 {cmp = "ne"} : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64xpred> -> !pto.vmi.mask<64xpred>
      %55 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      %56 = pto.vmi.bitcast %42 : !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xsi32>
      %57 = pto.vmi.bitcast %6 : !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xsi32>
      %58 = pto.vmi.vcmp %56, %57, %55 {cmp = "gt"} : !pto.vmi.vreg<64xsi32>, !pto.vmi.vreg<64xsi32>, !pto.vmi.mask<64xpred> -> !pto.vmi.mask<64xpred>
      %59 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      %60 = "pto.vmi.constant"() <{value = dense<0> : tensor<64xi32>}> : () -> !pto.vmi.vreg<64xi32>
      %61 = "pto.vmi.constant"() <{value = dense<1> : tensor<64xi32>}> : () -> !pto.vmi.vreg<64xi32>
      %62 = pto.vmi.vsel %58, %61, %60 : !pto.vmi.mask<64xpred>, !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi32>
      %63 = pto.vmi.vsel %54, %61, %60 : !pto.vmi.mask<64xpred>, !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi32>
      %64 = pto.vmi.vand %62, %63, %59 : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<64xi32>
      %65 = pto.vmi.vcmp %64, %60, %59 {cmp = "ne"} : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>, !pto.vmi.mask<64xpred> -> !pto.vmi.mask<64xpred>
      %c0_34 = arith.constant 0 : index
      %c2_35 = arith.constant 2 : index
      %c32_36 = arith.constant 32 : index
      %c64_37 = arith.constant 64 : index
      %c8192_38 = arith.constant 8192 : index
      %c256_39 = arith.constant 256 : index
      %c1_40 = arith.constant 1 : index
      %66 = arith.muli %arg10, %c8192 : index
      %67 = arith.muli %arg11, %c256 : index
      %68 = arith.addi %66, %67 : index
      %69 = pto.addptr %arg9, %68 : <ui8, ub> -> <ui8, ub>
      %70 = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
      %c0_41 = arith.constant 0 : index
      %71 = arith.addi %c0_41, %c0 : index
      %72 = pto.addptr %69, %71 : <ui8, ub> -> <ui8, ub>
      %73 = pto.castptr %72 : !pto.ptr<ui8, ub> -> !pto.ptr<ui32, ub>
      %74 = pto.castptr %73 : !pto.ptr<ui32, ub> -> !pto.ptr<ui8, ub>
      %75 = "pto.vmi.constant"() <{value = dense<0> : tensor<64xi32>}> : () -> !pto.vmi.vreg<64xi32>
      %76 = "pto.vmi.constant"() <{value = dense<1> : tensor<64xi32>}> : () -> !pto.vmi.vreg<64xi32>
      %77 = pto.vmi.vsel %65, %76, %75 : !pto.vmi.mask<64xpred>, !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi32>
      %78 = pto.vmi.trunci %77 {saturate = "SAT"} : !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xui8>
      %c0_42 = arith.constant 0 : index
      pto.vmi.store %78, %74[%c0_42] : !pto.vmi.vreg<64xui8>, !pto.ptr<ui8, ub>
    }
    return
  }
}

