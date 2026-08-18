; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

; Function Attrs: noduplicate noinline willreturn
define internal ptc_simdvf void @vector_add_kernel_outlined_vf_0(ptr addrspace(6) noalias %0, ptr addrspace(6) noalias %1, ptr addrspace(6) noalias %2) #0 !dbg !4 {
  br label %4, !dbg !7

4:                                                ; preds = %7, %3
  %5 = phi i32 [ %17, %7 ], [ 0, %3 ]
  %6 = icmp slt i32 %5, 256, !dbg !7
  br i1 %6, label %7, label %18, !dbg !7

7:                                                ; preds = %4
  %8 = sext i32 %5 to i64, !dbg !7
  %9 = getelementptr float, ptr addrspace(6) %0, i64 %8, !dbg !7
  %10 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %9, i32 0, i32 0, i32 0), !dbg !7
  %11 = getelementptr float, ptr addrspace(6) %1, i64 %8, !dbg !7
  %12 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %11, i32 0, i32 0, i32 0), !dbg !7
  %13 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %14 = call <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float> %10, <64 x float> %12, <256 x i1> %13)
  %15 = call <256 x i1> @llvm.hivm.pge.b32(i32 0, i32 0)
  %16 = getelementptr float, ptr addrspace(6) %2, i64 %8
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %14, ptr addrspace(6) %16, i32 0, i32 2, i32 0, <256 x i1> %15)
  %17 = add nsw i32 %5, 64, !dbg !7
  br label %4, !dbg !7

18:                                               ; preds = %4
  ret void, !dbg !7
}

; Function Attrs: alwaysinline
define private void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %0, ptr addrspace(1) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(6) %5, ptr addrspace(6) %6, i64 %7, i64 %8, i64 %9, i32 %10, float %11, i64 %12, i32 %13) #1 !dbg !8 {
  %15 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %0, 0, !dbg !10
  %16 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %15, ptr addrspace(1) %1, 1, !dbg !10
  %17 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %16, i64 %2, 2, !dbg !10
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %17, i64 %3, 3, 0, !dbg !10
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, i64 %4, 4, 0, !dbg !10
  %20 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8, !dbg !10
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, ptr %20, align 8, !dbg !10
  %21 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %5, 0, !dbg !10
  %22 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %21, ptr addrspace(6) %6, 1, !dbg !10
  %23 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %22, i64 %7, 2, !dbg !10
  %24 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0, !dbg !10
  %25 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0, !dbg !10
  %26 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8, !dbg !10
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %25, ptr %26, align 8, !dbg !10
  call void @_mlir_ciface_load_gm_to_ubuf_1d_float(ptr %20, ptr %26, i32 %10, float %11, i64 %12, i32 %13), !dbg !10
  ret void, !dbg !10
}

; Function Attrs: alwaysinline
declare !dbg !11 dso_local void @_mlir_ciface_load_gm_to_ubuf_1d_float(ptr, ptr, i32, float, i64, i32) #1

; Function Attrs: alwaysinline
define private void @store_ubuf_to_gm_1d_float(ptr addrspace(6) %0, ptr addrspace(6) %1, i64 %2, i64 %3, i64 %4, ptr addrspace(1) %5, ptr addrspace(1) %6, i64 %7, i64 %8, i64 %9, i32 %10) #1 !dbg !12 {
  %12 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(6) %0, 0, !dbg !13
  %13 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %12, ptr addrspace(6) %1, 1, !dbg !13
  %14 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %13, i64 %2, 2, !dbg !13
  %15 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %14, i64 %3, 3, 0, !dbg !13
  %16 = insertvalue { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %15, i64 %4, 4, 0, !dbg !13
  %17 = alloca { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] }, i64 1, align 8, !dbg !13
  store { ptr addrspace(6), ptr addrspace(6), i64, [1 x i64], [1 x i64] } %16, ptr %17, align 8, !dbg !13
  %18 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } undef, ptr addrspace(1) %5, 0, !dbg !13
  %19 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %18, ptr addrspace(1) %6, 1, !dbg !13
  %20 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %19, i64 %7, 2, !dbg !13
  %21 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %20, i64 %8, 3, 0, !dbg !13
  %22 = insertvalue { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %21, i64 %9, 4, 0, !dbg !13
  %23 = alloca { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] }, i64 1, align 8, !dbg !13
  store { ptr addrspace(1), ptr addrspace(1), i64, [1 x i64], [1 x i64] } %22, ptr %23, align 8, !dbg !13
  call void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr %17, ptr %23, i32 %10), !dbg !13
  ret void, !dbg !13
}

; Function Attrs: alwaysinline
declare !dbg !14 dso_local void @_mlir_ciface_store_ubuf_to_gm_1d_float(ptr, ptr, i32) #1

define dso_local void @vector_add_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7) #2 !dbg !15 {
  %9 = call i64 @llvm.hivm.GET.CTRL(), !dbg !16
  %10 = call i64 @llvm.hivm.SBITSET0(i64 %9, i64 60), !dbg !16
  call void @llvm.hivm.SET.CTRL(i64 %10), !dbg !16
  %11 = call i64 @llvm.hivm.GET.CTRL(), !dbg !16
  %12 = call i64 @llvm.hivm.SBITSET1(i64 %11, i64 48), !dbg !16
  call void @llvm.hivm.SET.CTRL(i64 %12), !dbg !16
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %2, ptr addrspace(1) %2, i64 0, i64 256, i64 1, ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 256, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0), !dbg !17
  call void @load_gm_to_ubuf_1d_float(ptr addrspace(1) %3, ptr addrspace(1) %3, i64 0, i64 256, i64 1, ptr addrspace(6) inttoptr (i64 1024 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 1024 to ptr addrspace(6)), i64 0, i64 256, i64 1, i32 0, float 0.000000e+00, i64 0, i32 0), !dbg !18
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0), !dbg !18
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0), !dbg !19
  call ptc_simdvf void @vector_add_kernel_outlined_vf_0(ptr addrspace(6) null, ptr addrspace(6) inttoptr (i64 1024 to ptr addrspace(6)), ptr addrspace(6) null), !dbg !19
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0), !dbg !19
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0), !dbg !20
  call void @store_ubuf_to_gm_1d_float(ptr addrspace(6) null, ptr addrspace(6) null, i64 0, i64 256, i64 1, ptr addrspace(1) %4, ptr addrspace(1) %4, i64 0, i64 256, i64 1, i32 0), !dbg !20
  %13 = call i64 @llvm.hivm.GET.CTRL(), !dbg !21
  %14 = call i64 @llvm.hivm.SBITSET1(i64 %13, i64 60), !dbg !21
  call void @llvm.hivm.SET.CTRL(i64 %14), !dbg !21
  call void @llvm.hivm.BARRIER(i64 6), !dbg !21
  ret void, !dbg !21
}

; Function Attrs: nounwind readonly 
declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) nocapture readonly, i32, i32, i32) #3

; Function Attrs: nounwind
declare <256 x i1> @llvm.hivm.pge.b32(i32, i32) #4

; Function Attrs: nounwind readnone 
declare <64 x float> @llvm.hivm.vadd.s.x.v64f32(<64 x float>, <64 x float>, <256 x i1>) #5

; Function Attrs: nounwind writeonly 
declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6) nocapture readonly, i32, i32, i32, <256 x i1>) #6

; Function Attrs: nounwind  inaccessiblememonly
declare i64 @llvm.hivm.GET.CTRL() #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET0(i64, i64) #5

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.SET.CTRL(i64) #7

; Function Attrs: nounwind readnone 
declare i64 @llvm.hivm.SBITSET1(i64, i64) #5

; Function Attrs: nounwind
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #4

; Function Attrs: nounwind  inaccessiblememonly
declare void @llvm.hivm.BARRIER(i64) #7

attributes #0 = { noduplicate noinline willreturn }
attributes #1 = { alwaysinline }
attributes #2 = { "target-cpu"="dav-c310" "target-features"="+dav-c310" }
attributes #3 = { nounwind readonly  }
attributes #4 = { nounwind }
attributes #5 = { nounwind readnone  }
attributes #6 = { nounwind writeonly  }
attributes #7 = { nounwind  inaccessiblememonly }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!hivm.annotations = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!2 = !DIFile(filename: "kernel.ttadapter.mlir", directory: "/home/m84446336/tmp/npuir-sim-vector-add-wrapper/dump/1Ccieb4dbQawqKqKBm7HVRN8BrI0YfZWGzX_dahGmnE")
!3 = !{ptr @vector_add_kernel, !"kernel", i32 1}
!4 = distinct !DISubprogram(name: "vector_add_kernel_outlined_vf_0", linkageName: "vector_add_kernel_outlined_vf_0", scope: !2, file: !2, line: 11, type: !5, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{}
!7 = !DILocation(line: 11, column: 10, scope: !4)
!8 = distinct !DISubprogram(name: "load_gm_to_ubuf_1d_float", linkageName: "load_gm_to_ubuf_1d_float", scope: !9, file: !9, type: !5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!9 = !DIFile(filename: "internal", directory: "")
!10 = !DILocation(line: 0, scope: !8)
!11 = !DISubprogram(name: "_mlir_ciface_load_gm_to_ubuf_1d_float", linkageName: "_mlir_ciface_load_gm_to_ubuf_1d_float", scope: !9, file: !9, type: !5, spFlags: DISPFlagOptimized)
!12 = distinct !DISubprogram(name: "store_ubuf_to_gm_1d_float", linkageName: "store_ubuf_to_gm_1d_float", scope: !9, file: !9, type: !5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!13 = !DILocation(line: 0, scope: !12)
!14 = !DISubprogram(name: "_mlir_ciface_store_ubuf_to_gm_1d_float", linkageName: "_mlir_ciface_store_ubuf_to_gm_1d_float", scope: !9, file: !9, type: !5, spFlags: DISPFlagOptimized)
!15 = distinct !DISubprogram(name: "vector_add_kernel", linkageName: "vector_add_kernel", scope: !2, file: !2, line: 2, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!16 = !DILocation(line: 2, column: 3, scope: !15)
!17 = !DILocation(line: 5, column: 5, scope: !15)
!18 = !DILocation(line: 9, column: 5, scope: !15)
!19 = !DILocation(line: 11, column: 10, scope: !15)
!20 = !DILocation(line: 13, column: 5, scope: !15)
!21 = !DILocation(line: 14, column: 5, scope: !15)

