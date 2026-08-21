; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @vector_add_large_kernel_outlined_vf_0(ptr addrspace(6) noalias %0) #0 !dbg !4 {
  %2 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %3 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %2, i32 1)
  br label %4, !dbg !7

4:                                                ; preds = %7, %1
  %5 = phi i32 [ %11, %7 ], [ 0, %1 ]
  %6 = icmp slt i32 %5, 2048, !dbg !7
  br i1 %6, label %7, label %12, !dbg !7

7:                                                ; preds = %4
  %8 = sext i32 %5 to i64, !dbg !7
  %9 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0), !dbg !8
  %10 = getelementptr float, ptr addrspace(6) %0, i64 %8, !dbg !8
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %3, ptr addrspace(6) %10, i32 0, i32 2, i32 0, <256 x i1> %9), !dbg !8
  %11 = add nsw i32 %5, 64, !dbg !7
  br label %4, !dbg !7

12:                                               ; preds = %4
  ret void, !dbg !7
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @vector_add_large_kernel_outlined_vf_1(ptr addrspace(6) noalias %0) #0 !dbg !9 {
  %2 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %3 = call <64 x float> @llvm.hivm.vdups.z.v64f32(float 0.000000e+00, <256 x i1> %2, i32 1)
  br label %4, !dbg !10

4:                                                ; preds = %7, %1
  %5 = phi i32 [ %11, %7 ], [ 0, %1 ]
  %6 = icmp slt i32 %5, 2048, !dbg !10
  br i1 %6, label %7, label %12, !dbg !10

7:                                                ; preds = %4
  %8 = sext i32 %5 to i64, !dbg !10
  %9 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0), !dbg !11
  %10 = getelementptr float, ptr addrspace(6) %0, i64 %8, !dbg !11
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %3, ptr addrspace(6) %10, i32 0, i32 2, i32 0, <256 x i1> %9), !dbg !11
  %11 = add nsw i32 %5, 64, !dbg !10
  br label %4, !dbg !10

12:                                               ; preds = %4
  ret void, !dbg !10
}

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @vector_add_large_kernel_outlined_vf_2(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2) #0 !dbg !12 {
  br label %4, !dbg !13

4:                                                ; preds = %7, %3
  %5 = phi i32 [ %17, %7 ], [ 0, %3 ]
  %6 = icmp slt i32 %5, 2048, !dbg !13
  br i1 %6, label %7, label %18, !dbg !13

7:                                                ; preds = %4
  %8 = sext i32 %5 to i64, !dbg !13
  %9 = getelementptr float, ptr addrspace(6) %0, i64 %8, !dbg !13
  %10 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %9, i32 0, i32 0, i32 0), !dbg !13
  %11 = getelementptr float, ptr addrspace(6) %1, i64 %8, !dbg !13
  %12 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %11, i32 0, i32 0, i32 0), !dbg !13
  %13 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %14 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %10, <64 x float> %12, <256 x i1> %13)
  %15 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %16 = getelementptr float, ptr addrspace(6) %2, i64 %8
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %14, ptr addrspace(6) %16, i32 0, i32 2, i32 0, <256 x i1> %15)
  %17 = add nsw i32 %5, 64, !dbg !13
  br label %4, !dbg !13

18:                                               ; preds = %4
  ret void, !dbg !13
}

; Function Attrs: alwaysinline
define private void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, float %11, i64 %12, i32 %13) #1 !dbg !14 {
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0, !dbg !16
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, ptr addrspace(1) %1, 1, !dbg !16
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %2, 2, !dbg !16
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %3, 3, 0, !dbg !16
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, i64 %4, 4, 0, !dbg !16
  %20 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8, !dbg !16
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, ptr %20, align 8, !dbg !16
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0, !dbg !16
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %6, 1, !dbg !16
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %7, 2, !dbg !16
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0, !dbg !16
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0, !dbg !16
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8, !dbg !16
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8, !dbg !16
  call void @_mlir_ciface_load_gm_to_ubuf_1d_float(ptr %20, ptr %26, i32 %10, float %11, i64 %12, i32 %13), !dbg !16
  ret void, !dbg !16
}

; Function Attrs: alwaysinline
declare !dbg !17 dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_float(ptr, ptr, i32, float, i64, i32) #1

; Function Attrs: alwaysinline
define private void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i32 %10) #1 !dbg !18 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0, !dbg !19
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1, !dbg !19
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2, !dbg !19
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0, !dbg !19
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0, !dbg !19
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8, !dbg !19
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8, !dbg !19
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %5, 0, !dbg !19
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(1) %6, 1, !dbg !19
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2, !dbg !19
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0, !dbg !19
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0, !dbg !19
  %23 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8, !dbg !19
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8, !dbg !19
  call void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr %17, ptr %23, i32 %10), !dbg !19
  ret void, !dbg !19
}

; Function Attrs: alwaysinline
declare !dbg !20 dso_local void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr, ptr, i32) #1

define dso_local void @vector_add_large_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7) #2 !dbg !21 {
  %9 = alloca i64, i64 1, align 8
  store i64 0, ptr %9, align 4
  %10 = mul i32 %5, %6, !dbg !22
  %11 = mul i32 %10, %7, !dbg !22
  %12 = call i64 @llvm.hivm.GET.BLOCK.IDX(), !dbg !22
  %13 = trunc i64 %12 to i32
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  br label %14

14:                                               ; preds = %17, %8
  %15 = phi i32 [ %48, %17 ], [ %13, %8 ]
  %16 = icmp slt i32 %15, %11
  br i1 %16, label %17, label %49

17:                                               ; preds = %14
  %18 = load i64, ptr %9, align 4
  %19 = urem i64 %18, 2
  %20 = icmp eq i64 %19, 1, !dbg !23
  %21 = select i1 %20, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 40960 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 40960 to ptr addrspace(6)), i64 0, [1 x i64] [i64 2048], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6)), i64 0, [1 x i64] [i64 2048], [1 x i64] [i64 1] }, !dbg !23
  %22 = select i1 %20, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), i64 0, [1 x i64] [i64 2048], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6)), i64 0, [1 x i64] [i64 2048], [1 x i64] [i64 1] }, !dbg !24
  %23 = select i1 %20, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) inttoptr (i64 24576 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 24576 to ptr addrspace(6)), i64 0, [1 x i64] [i64 2048], [1 x i64] [i64 1] }, { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } { ptr addrspace(6) null, ptr addrspace(6) null, i64 0, [1 x i64] [i64 2048], [1 x i64] [i64 1] }, !dbg !25
  %24 = trunc i64 %19 to i1
  %25 = xor i1 %24, true
  %26 = zext i1 %25 to i64
  %27 = call i64 @llvm.hivm.GET.CTRL(), !dbg !26
  %28 = call i64 @llvm.hivm.SBITSET0(i64 %27, i64 60), !dbg !26
  call void @llvm.hivm.SET.CTRL(i64 %28), !dbg !26
  %29 = call i64 @llvm.hivm.GET.CTRL(), !dbg !26
  %30 = call i64 @llvm.hivm.SBITSET1(i64 %29, i64 48), !dbg !26
  call void @llvm.hivm.SET.CTRL(i64 %30), !dbg !26
  %31 = srem i32 %15, %5, !dbg !22
  %32 = mul i32 %31, 2000, !dbg !27
  %33 = sext i32 %32 to i64, !dbg !28
  %34 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, 1, !dbg !29
  call ptc_simdvf void @vector_add_large_kernel_outlined_vf_0(ptr addrspace(6) %34), !dbg !29
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0), !dbg !29
  %35 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, 0, !dbg !30
  %36 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, 1, !dbg !30
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0), !dbg !31
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 %33, i64 2000, i64 1, ptr addrspace(6) %35, ptr addrspace(6) %36, i64 0, i64 2000, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0), !dbg !31
  %37 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, 1, !dbg !32
  call ptc_simdvf void @vector_add_large_kernel_outlined_vf_1(ptr addrspace(6) %37), !dbg !32
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0), !dbg !32
  %38 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, 0, !dbg !33
  %39 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, 1, !dbg !33
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0), !dbg !34
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 %33, i64 2000, i64 1, ptr addrspace(6) %38, ptr addrspace(6) %39, i64 0, i64 2000, i64 1, i32 2, float 0.000000e+00, i64 0, i32 0), !dbg !34
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0), !dbg !34
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 1, i64 %26), !dbg !23
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0), !dbg !23
  %40 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, 1, !dbg !23
  %41 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, 1, !dbg !23
  %42 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, 1, !dbg !23
  call ptc_simdvf void @vector_add_large_kernel_outlined_vf_2(ptr addrspace(6) %40, ptr addrspace(6) %41, ptr addrspace(6) %42), !dbg !23
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0), !dbg !23
  %43 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, 0, !dbg !35
  %44 = extractvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, 1, !dbg !35
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0), !dbg !36
  call void @llvm.hivm.BARRIER(i64 5), !dbg !36
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %43, ptr addrspace(6) %44, i64 0, i64 2000, i64 1, ptr addrspace(1) %4, ptr addrspace(1) %4, i64 %33, i64 2000, i64 1, i32 0), !dbg !36
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 1, i64 %26), !dbg !36
  %45 = call i64 @llvm.hivm.GET.CTRL(), !dbg !37
  %46 = call i64 @llvm.hivm.SBITSET1(i64 %45, i64 60), !dbg !37
  call void @llvm.hivm.SET.CTRL(i64 %46), !dbg !37
  %47 = add i64 %18, 1
  store i64 %47, ptr %9, align 4
  %48 = add nsw i32 %15, 64
  br label %14

49:                                               ; preds = %14
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.BARRIER(i64 6), !dbg !37
  ret void, !dbg !37
}

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b32(i32, i32) #3

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vdups.z.v64f32(float, <256 x i1>, i32) #4

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #5

; Function Attrs: nounwind readonly 
declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) nocapture readonly, i32, i32, i32) #6

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #4

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.GET.BLOCK.IDX() #4

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #3

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #7

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET1(i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #3

attributes #0 = { noduplicate noinline willreturn }
attributes #1 = { alwaysinline }
attributes #2 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #3 = { nounwind }
attributes #4 = { nounwind readnone  }
attributes #5 = { nounwind writeonly  }
attributes #6 = { nounwind readonly  }
attributes #7 = { nounwind  inaccessiblememonly }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!hivm.annotations = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!2 = !DIFile(filename: "vadd_large.ttadapter.mlir", directory: "/home/m84446336/tmp/npuir-ptoas-comparison/vadd_large-20260821T132319Z/early-ir")
!3 = !{ptr @vector_add_large_kernel, !"kernel", i32 1}
!4 = distinct !DISubprogram(name: "vector_add_large_kernel_outlined_vf_0", linkageName: "vector_add_large_kernel_outlined_vf_0", scope: !2, file: !2, line: 9, type: !5, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{}
!7 = !DILocation(line: 9, column: 5, scope: !4)
!8 = !DILocation(line: 3, column: 12, scope: !4)
!9 = distinct !DISubprogram(name: "vector_add_large_kernel_outlined_vf_1", linkageName: "vector_add_large_kernel_outlined_vf_1", scope: !2, file: !2, line: 16, type: !5, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!10 = !DILocation(line: 16, column: 5, scope: !9)
!11 = !DILocation(line: 3, column: 12, scope: !9)
!12 = distinct !DISubprogram(name: "vector_add_large_kernel_outlined_vf_2", linkageName: "vector_add_large_kernel_outlined_vf_2", scope: !2, file: !2, line: 21, type: !5, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!13 = !DILocation(line: 21, column: 10, scope: !12)
!14 = distinct !DISubprogram(name: "load_gm_to_ubuf_1d_float", linkageName: "load_gm_to_ubuf_1d_float", scope: !15, file: !15, type: !5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!15 = !DIFile(filename: "internal", directory: "")
!16 = !DILocation(line: 0, scope: !14)
!17 = !DISubprogram(name: "_mlir_ciface_load_gm_to_ubuf_1d_float", linkageName: "_mlir_ciface_load_gm_to_ubuf_1d_float", scope: !15, file: !15, type: !5, spFlags: DISPFlagOptimized)
!18 = distinct !DISubprogram(name: "store_ubuf_to_gm_1d_float", linkageName: "store_ubuf_to_gm_1d_float", scope: !15, file: !15, type: !5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!19 = !DILocation(line: 0, scope: !18)
!20 = !DISubprogram(name: "_mlir_ciface_store_ubuf_to_gm_1d_float", linkageName: "_mlir_ciface_store_ubuf_to_gm_1d_float", scope: !15, file: !15, type: !5, spFlags: DISPFlagOptimized)
!21 = distinct !DISubprogram(name: "vector_add_large_kernel", linkageName: "vector_add_large_kernel", scope: !2, file: !2, line: 2, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!22 = !DILocation(line: 3, column: 12, scope: !21)
!23 = !DILocation(line: 21, column: 10, scope: !21)
!24 = !DILocation(line: 15, column: 16, scope: !21)
!25 = !DILocation(line: 8, column: 14, scope: !21)
!26 = !DILocation(line: 2, column: 3, scope: !21)
!27 = !DILocation(line: 5, column: 10, scope: !21)
!28 = !DILocation(line: 6, column: 10, scope: !21)
!29 = !DILocation(line: 9, column: 5, scope: !21)
!30 = !DILocation(line: 11, column: 18, scope: !21)
!31 = !DILocation(line: 12, column: 5, scope: !21)
!32 = !DILocation(line: 16, column: 5, scope: !21)
!33 = !DILocation(line: 18, column: 18, scope: !21)
!34 = !DILocation(line: 19, column: 5, scope: !21)
!35 = !DILocation(line: 23, column: 24, scope: !21)
!36 = !DILocation(line: 25, column: 5, scope: !21)
!37 = !DILocation(line: 26, column: 5, scope: !21)

