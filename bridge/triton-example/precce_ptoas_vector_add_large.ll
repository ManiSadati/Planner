; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

declare <256 x i1> @llvm.hivm.pset.b32(i32)

declare <64 x float> @llvm.hivm.vdups.v64f32.z(float, <256 x i1>, i32)

declare void @llvm.hivm.vstsx1.v64f32(<64 x float>, ptr addrspace(6), i32, i32, i32, <256 x i1>)

declare <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6), i32, i32, i32)

declare <64 x float> @llvm.hivm.vadd.v64f32.x(<64 x float>, <64 x float>, <256 x i1>)

declare i64 @llvm.hivm.GET.BLOCK.IDX()

declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64)

declare i64 @llvm.hivm.GET.CTRL()

declare void @llvm.hivm.SET.CTRL(i64)

declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64)

declare void @llvm.hivm.SET.MOV.PAD.VAL(i64)

declare void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6), ptr addrspace(1), i64, i64)

declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64)

declare void @llvm.hivm.BARRIER(i64)

declare void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV(ptr addrspace(1), ptr addrspace(6), i64, i64)

declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64)

; Function Attrs: noinline
define void @vector_add_large_kernel_outlined_vf_0(ptr addrspace(6) %0) #0 {
  br label %2

2:                                                ; preds = %aivscope.latch, %1
  %3 = phi i64 [ %16, %aivscope.latch ], [ 0, %1 ]
  %4 = icmp slt i64 %3, 1
  br i1 %4, label %5, label %17

5:                                                ; preds = %2
  %6 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %7 = call <64 x float> @llvm.hivm.vdups.v64f32.z(float 0.000000e+00, <256 x i1> %6, i32 1)
  br label %8

8:                                                ; preds = %11, %5
  %9 = phi i16 [ %14, %11 ], [ 0, %5 ]
  %10 = icmp slt i16 %9, 2048
  br i1 %10, label %11, label %15

11:                                               ; preds = %8
  %12 = sext i16 %9 to i64
  %13 = getelementptr float, ptr addrspace(6) %0, i64 %12
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %7, ptr addrspace(6) %13, i32 0, i32 2, i32 0, <256 x i1> %6)
  %14 = add i16 %9, 64
  br label %8

15:                                               ; preds = %8
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %15
  %16 = add i64 %3, 1
  br label %2, !llvm.loop !3

17:                                               ; preds = %2
  ret void
}

; Function Attrs: noinline
define void @vector_add_large_kernel_outlined_vf_1(ptr addrspace(6) %0) #0 {
  br label %2

2:                                                ; preds = %aivscope.latch, %1
  %3 = phi i64 [ %16, %aivscope.latch ], [ 0, %1 ]
  %4 = icmp slt i64 %3, 1
  br i1 %4, label %5, label %17

5:                                                ; preds = %2
  %6 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  %7 = call <64 x float> @llvm.hivm.vdups.v64f32.z(float 0.000000e+00, <256 x i1> %6, i32 1)
  br label %8

8:                                                ; preds = %11, %5
  %9 = phi i16 [ %14, %11 ], [ 0, %5 ]
  %10 = icmp slt i16 %9, 2048
  br i1 %10, label %11, label %15

11:                                               ; preds = %8
  %12 = sext i16 %9 to i64
  %13 = getelementptr float, ptr addrspace(6) %0, i64 %12
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %7, ptr addrspace(6) %13, i32 0, i32 2, i32 0, <256 x i1> %6)
  %14 = add i16 %9, 64
  br label %8

15:                                               ; preds = %8
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %15
  %16 = add i64 %3, 1
  br label %2, !llvm.loop !5

17:                                               ; preds = %2
  ret void
}

; Function Attrs: noinline
define void @vector_add_large_kernel_outlined_vf_2(ptr addrspace(6) %0, ptr addrspace(6) %1, ptr addrspace(6) %2) #0 {
  br label %4

4:                                                ; preds = %aivscope.latch, %3
  %5 = phi i64 [ %22, %aivscope.latch ], [ 0, %3 ]
  %6 = icmp slt i64 %5, 1
  br i1 %6, label %7, label %23

7:                                                ; preds = %4
  %8 = call <256 x i1> @llvm.hivm.pset.b32(i32 0)
  br label %9

9:                                                ; preds = %12, %7
  %10 = phi i16 [ %20, %12 ], [ 0, %7 ]
  %11 = icmp slt i16 %10, 2048
  br i1 %11, label %12, label %21

12:                                               ; preds = %9
  %13 = sext i16 %10 to i64
  %14 = getelementptr float, ptr addrspace(6) %0, i64 %13
  %15 = getelementptr float, ptr addrspace(6) %1, i64 %13
  %16 = getelementptr float, ptr addrspace(6) %2, i64 %13
  %17 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %14, i32 0, i32 0, i32 0)
  %18 = call <64 x float> @llvm.hivm.vldsx1.v64f32(ptr addrspace(6) %15, i32 0, i32 0, i32 0)
  %19 = call <64 x float> @llvm.hivm.vadd.v64f32.x(<64 x float> %17, <64 x float> %18, <256 x i1> %8)
  call void @llvm.hivm.vstsx1.v64f32(<64 x float> %19, ptr addrspace(6) %16, i32 0, i32 2, i32 0, <256 x i1> %8)
  %20 = add i16 %10, 64
  br label %9

21:                                               ; preds = %9
  br label %aivscope.latch

aivscope.latch:                                   ; preds = %21
  %22 = add i64 %5, 1
  br label %4, !llvm.loop !6

23:                                               ; preds = %4
  ret void
}

define void @vector_add_large_kernel_mix_aiv(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7) #1 {
  %9 = alloca i64, i64 1, align 8
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %9, 0
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, ptr %9, 1
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 0, 2
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 1, 3, 0
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 1, 4, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 1
  %16 = getelementptr i64, ptr %15, i64 0
  store i64 0, ptr %16, align 4
  %17 = mul i32 %5, %6
  %18 = mul i32 %17, %7
  %19 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %20 = trunc i64 %19 to i32
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  br label %21

21:                                               ; preds = %24, %8
  %22 = phi i32 [ %51, %24 ], [ %20, %8 ]
  %23 = icmp slt i32 %22, %18
  br i1 %23, label %24, label %52

24:                                               ; preds = %21
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 1
  %26 = getelementptr i64, ptr %25, i64 0
  %27 = load i64, ptr %26, align 4
  %28 = urem i64 %27, 2
  %29 = icmp eq i64 %28, 1
  %30 = select i1 %29, ptr addrspace(6) inttoptr (i64 40960 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 16384 to ptr addrspace(6))
  %31 = select i1 %29, ptr addrspace(6) inttoptr (i64 32768 to ptr addrspace(6)), ptr addrspace(6) inttoptr (i64 8192 to ptr addrspace(6))
  %32 = select i1 %29, ptr addrspace(6) inttoptr (i64 24576 to ptr addrspace(6)), ptr addrspace(6) null
  %33 = trunc i64 %28 to i1
  %34 = xor i1 %33, true
  %35 = zext i1 %34 to i64
  %36 = call i64 @llvm.hivm.GET.CTRL()
  %37 = and i64 %36, -1152921504606846977
  call void @llvm.hivm.SET.CTRL(i64 %37)
  %38 = call i64 @llvm.hivm.GET.CTRL()
  %39 = or i64 %38, 281474976710656
  call void @llvm.hivm.SET.CTRL(i64 %39)
  %40 = srem i32 %22, %5
  %41 = mul i32 %40, 2000
  %42 = sext i32 %41 to i64
  call void @vector_add_large_kernel_outlined_vf_0(ptr addrspace(6) %32)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  %43 = getelementptr float, ptr addrspace(1) %2, i64 %42
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.SET.MOV.PAD.VAL(i64 0)
  call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6) %32, ptr addrspace(1) %43, i64 288230644587167760, i64 0)
  call void @vector_add_large_kernel_outlined_vf_1(ptr addrspace(6) %31)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  %44 = getelementptr float, ptr addrspace(1) %3, i64 %42
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.SET.MOV.PAD.VAL(i64 0)
  call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6) %31, ptr addrspace(1) %44, i64 288230644587167760, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 1, i64 %35)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 0)
  call void @vector_add_large_kernel_outlined_vf_2(ptr addrspace(6) %32, ptr addrspace(6) %31, ptr addrspace(6) %30)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 5, i64 0)
  %45 = getelementptr float, ptr addrspace(1) %4, i64 %42
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 5, i64 0)
  call void @llvm.hivm.BARRIER(i64 5)
  call void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV(ptr addrspace(1) %45, ptr addrspace(6) %30, i64 268435456016, i64 0)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 1, i64 %35)
  %46 = call i64 @llvm.hivm.GET.CTRL()
  %47 = or i64 %46, 1152921504606846976
  call void @llvm.hivm.SET.CTRL(i64 %47)
  %48 = add i64 %27, 1
  %49 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, 1
  %50 = getelementptr i64, ptr %49, i64 0
  store i64 %48, ptr %50, align 4
  %51 = add i32 %22, 64
  br label %21

52:                                               ; preds = %21
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.BARRIER(i64 6)
  ret void
}

attributes #0 = { noinline "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #1 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1, !2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @vector_add_large_kernel_mix_aiv, !"kernel", i32 1}
!2 = !{ptr @vector_add_large_kernel_mix_aiv, !"kernel_with_simd", i32 1}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.aivector_scope"}
!5 = distinct !{!5, !4}
!6 = distinct !{!6, !4}

