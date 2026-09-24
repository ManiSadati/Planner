Timing of the compilation pipeline of `bishengir-compile` on the same input but two different flows: 
  1. Compeletly in NPU-IR
  2. NPU-IR until HIVMAVToPTASVMI (right before ToHIVMIntrin)


Aggregated timings for 1:
```
===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 1.1612 seconds

  ----Wall Time----  ----Name----
    1.1612 (100.0%)  root
    0.5277 ( 45.4%)  BiShengHIR
    0.4644 ( 40.0%)  'func.func' Pipeline
    0.4465 ( 38.5%)  Pipeline Collection : ['func.func']
    0.3929 ( 33.8%)  buildFinalHIVMPipelines
    0.1982 ( 17.1%)  Canonicalizer
    0.1273 ( 11.0%)  hivmc-a5
    0.1049 (  9.0%)  MarkRealCoreType
    0.0910 (  7.8%)  BiShengSIMD
    0.0355 (  3.1%)  InlineScope
    0.0226 (  1.9%)  AutoVectorizeV2
    0.0212 (  1.8%)  CanonicalizeIterArg
    0.0206 (  1.8%)  InsertLoadStoreForMixCV
    0.0193 (  1.7%)  NormalizeVector
    0.0179 (  1.5%)  NormalizeTensorOps
    0.0157 (  1.4%)  Inliner
    0.0152 (  1.3%)  HIVMOptSinglePointOp
    0.0148 (  1.3%)  CSE
    0.0137 (  1.2%)  SCFForLoopCanonicalization
    0.0131 (  1.1%)  HIVMAggregatedDecomposeOp
    0.0118 (  1.0%)  CloneTensorEmpty
    0.0112 (  1.0%)  NormalizeMatmul
    0.0109 (  0.9%)  GraphSyncSolver
    0.0108 (  0.9%)  ConvertArithToAffine
    0.0089 (  0.8%)  SimplifyVFArgs
    0.0082 (  0.7%)  ConvertHIVMAVEToAVEIntrin
    0.0081 (  0.7%)  MemrefDeadStoreEliminationOp
    0.0081 (  0.7%)  InferHIVMMemScope
    0.0077 (  0.7%)  OneShotBufferize
    0.0076 (  0.7%)  TensorCopyInsertion
    0.0075 (  0.6%)  PreVectorizationFusion
    0.0071 (  0.6%)  Normalize
    0.0066 (  0.6%)  SyncBlockHoisting
    0.0060 (  0.5%)  ConvertToHIVMOp
    0.0053 (  0.5%)  PlanMemoryRegBase
    0.0053 (  0.5%)  InsertFixpipe
    0.0051 (  0.4%)  InlineFixpipe
    0.0050 (  0.4%)  AdaptTritonKernel
    0.0048 (  0.4%)  EnableStrideAlign
    0.0047 (  0.4%)  LiftLowestStride
    0.0047 (  0.4%)  CanonicalizeTensorReshape
    0.0046 (  0.4%)  SymbolDCE
    0.0045 (  0.4%)  FoldTensorEmpty
    0.0043 (  0.4%)  AnnotationLowering
    0.0043 (  0.4%)  ConvertVectorToHIVMAVE
    0.0043 (  0.4%)  VFFusion
    0.0043 (  0.4%)  LegalizeBoolPass
    0.0042 (  0.4%)  ConvertArithToHFusion
    0.0042 (  0.4%)  InsertWorkSpaceForMixCV
    0.0040 (  0.3%)  ConvertHIVMAVEToStandard
    0.0037 (  0.3%)  LoopInvariantSubsetHoisting
    0.0035 (  0.3%)  ConvertArithToHIVMAVE
    0.0035 (  0.3%)  HFusionInlineBrc
    0.0035 (  0.3%)  MarkMultiBuffer
    0.0034 (  0.3%)  FlattenOps
    0.0034 (  0.3%)  HoistVstas
    0.0032 (  0.3%)  TileAndBindSubBlock
    0.0031 (  0.3%)  ConvertHIVMToStandard
    0.0030 (  0.3%)  LowerVectorMaskPass
    0.0030 (  0.3%)  FoldExtractInsertPair
    0.0030 (  0.3%)  DropEquivalentBufferResults
    0.0029 (  0.3%)  HIVMDecomposeOp
    0.0029 (  0.2%)  LiftZeroRank
    0.0029 (  0.2%)  SinkOpToConsumerInLoop
    0.0029 (  0.2%)  ConvertMathToHFusion
    0.0028 (  0.2%)  Decompose
    0.0028 (  0.2%)  HIVMNormalize
    0.0028 (  0.2%)  InsertCVTightCoupledBuffer
    0.0027 (  0.2%)  HIVMFlattenOps
    0.0027 (  0.2%)  OutlineVectorFunction
    0.0027 (  0.2%)  AutoVectorizeVerifier
    0.0026 (  0.2%)  MergeVecScope
    0.0025 (  0.2%)  ConvertHFusionToVector
    0.0024 (  0.2%)  I1opSoftImpl
    0.0024 (  0.2%)  ReduceRankSubview
    0.0024 (  0.2%)  HIVMLowerToLoops
    0.0024 (  0.2%)  EnableMultiBuffer
    0.0023 (  0.2%)  ConvertTensorToHFusion
    0.0023 (  0.2%)  AVENormalizeOps
    0.0023 (  0.2%)  ExpandStridedMetadata
    0.0022 (  0.2%)  HFusionFoldUnitDims
    0.0022 (  0.2%)  InjectIR
    0.0022 (  0.2%)  ConvertHFusionToHIVM
    0.0022 (  0.2%)  ConvertAffineToStandard
    0.0021 (  0.2%)  PullSliceIntoVectorFunction
    0.0021 (  0.2%)  PropagateConvertLayout
    0.0020 (  0.2%)  SCFToControlFlow
    0.0020 (  0.2%)  LoopInvariantPromotion
    0.0020 (  0.2%)  ProcessVsstb
    0.0020 (  0.2%)  FuseTransposeIntoLoad
    0.0019 (  0.2%)  ReplaceWithVectorScalar
    0.0019 (  0.2%)  OptimizeDpsOpWithYieldedInsertSlice
    0.0019 (  0.2%)  TreeReduceV2
    0.0019 (  0.2%)  LegalizeLoopIterArgs
    0.0018 (  0.2%)  ConvertAscendDPXToHIVMRegbaseIntrins
    0.0018 (  0.2%)  LowerMemRefExt
    0.0017 (  0.1%)  OptimizeReductionLoopHIVMAVE
    0.0017 (  0.1%)  BindSyncBlockLockArg
    0.0017 (  0.1%)  SimplifyOps
    0.0017 (  0.1%)  OutlineCopyInVF
    0.0017 (  0.1%)  LowerCreateSyncBlockLock
    0.0017 (  0.1%)  VectorizeOps
    0.0017 (  0.1%)  AnalyzeVectorLayout
    0.0016 (  0.1%)  LegalizeBF16Pass
    0.0016 (  0.1%)  HIVMComplexReductionIntermediateLowering
    0.0016 (  0.1%)  AveLoopOptimize
    0.0016 (  0.1%)  CombineOptimizedConvertLayout
    0.0015 (  0.1%)  CombineAVEOPs
    0.0015 (  0.1%)  PropagateReshape
    0.0015 (  0.1%)  LegalizeOptHIVMAVE
    0.0015 (  0.1%)  BindBuffer
    0.0015 (  0.1%)  ProcessMembar
    0.0015 (  0.1%)  InsertL12UBForDebug
    0.0015 (  0.1%)  InsertFreeLockVarBeforeReturn
    0.0015 (  0.1%)  NormalizeLastDimUnalignedTensorOp
    0.0014 (  0.1%)  RemoveRedundantWriteAndReadPair
    0.0014 (  0.1%)  ArithExpandOpsPass
    0.0014 (  0.1%)  MapForToForall
    0.0014 (  0.1%)  ScalarBroadcastToVLoad
    0.0014 (  0.1%)  PLTToPLTM
    0.0014 (  0.1%)  PLTToPGE
    0.0014 (  0.1%)  ConvertLinalgToHFusion
    0.0014 (  0.1%)  HIVMInlineOTFLoadStore
    0.0014 (  0.1%)  RemoveRedundantLoopInit
    0.0014 (  0.1%)  FoldAllocReshapeOp
    0.0014 (  0.1%)  ConstantizeBufferSize
    0.0014 (  0.1%)  InlineOTFBroadcast
    0.0013 (  0.1%)  BindWorkSpaceArg
    0.0013 (  0.1%)  NormalizeArith
    0.0013 (  0.1%)  ConvertLayoutToTranspose
    0.0013 (  0.1%)  HFusionGeneralizePass
    0.0013 (  0.1%)  HIVMMapForallToBlocks
    0.0013 (  0.1%)  InsertConvertLayout
    0.0013 (  0.1%)  AllocToAlloca
    0.0012 (  0.1%)  TileBatchMMIntoLoop
    0.0012 (  0.1%)  LiftArithIndexCast
    0.0012 (  0.1%)  MarkSyncBlockLockWithSubblock
    0.0011 (  0.1%)  InsertAnchorsAndBackup
    0.0011 (  0.1%)  AppendTargetDeviceSpec
    0.0011 (  0.1%)  DecomposeFRem
    0.0011 (  0.1%)  ConvertGPUToHFusion
    0.0011 (  0.1%)  AlignAllocSize
    0.0010 (  0.1%)  OutlineAllocInVF
    0.0010 (  0.1%)  AddFFTSAddr
    0.0010 (  0.1%)  PrepareI1Nx1ForVectorization
    0.0010 (  0.1%)  HoistTensorEmpty
    0.0010 (  0.1%)  MarkDisableLoad
    0.0010 (  0.1%)  OutlineScope
    0.0010 (  0.1%)  RemoveVectorLayoutAttr
    0.0010 (  0.1%)  LegalizeFP8Pass
    0.0010 (  0.1%)  MaterializeVectorWriteToDestination
    0.0010 (  0.1%)  CanonicalizeModule
    0.0009 (  0.1%)  ConvertTensorToHIVM
    0.0009 (  0.1%)  LoopInvariantCodeMotion
    0.0009 (  0.1%)  ConvertGenericToNamedOp
    0.0008 (  0.1%)  NormalizeSliceOps
    0.0008 (  0.1%)  AnalyzeArithVectorMask
    0.0008 (  0.1%)  LowerMultiBufferCounter
    0.0008 (  0.1%)  GenericUnroller
    0.0008 (  0.1%)  TritonGlobalKernelArgsToHIVMOp
    0.0008 (  0.1%)  AutoBlockifyParallelLoop
    0.0008 (  0.1%)  InitEntryKernel
    0.0008 (  0.1%)  FixCallUnknownLoc
    0.0007 (  0.1%)  LegalizeScalarPass
    0.0007 (  0.1%)  InferHIVMDataLayout
    0.0007 (  0.1%)  InsertInferSyncBlockLockNumAndInitFunc
    0.0007 (  0.1%)  InsertInitAndFinishForDebug
    0.0007 (  0.1%)  RemoveMaskFromUnalignedReductionLoop
    0.0007 (  0.1%)  ExposeMemrefWriteToTensor
    0.0007 (  0.1%)  MarkStrideAlign
    0.0007 (  0.1%)  InsertInferWorkSpaceSizeFunc
    0.0007 (  0.1%)  UpliftWhileToFor
    0.0007 (  0.1%)  PreMarkStrideAlign
    0.0007 (  0.1%)  DelayedCrossCoreGSS
    0.0007 (  0.1%)  MarkTightlyCoupledBuffer
    0.0007 (  0.1%)  CrossCoreGSS
    0.0007 (  0.1%)  SplitMixKernel
    0.0007 (  0.1%)  PeelLoopsContainingTranspose
    0.0006 (  0.1%)  InferVFMode
    0.0006 (  0.1%)  InferFuncCoreType
    0.0006 (  0.1%)  EraseSymbol
    0.0006 (  0.0%)  AutoInferBufferSize
    0.0006 (  0.0%)  HoistTightlyCoupledAlloc
    0.0006 (  0.0%)  AllocExtraBuffer
    0.0006 (  0.0%)  SetBufferSize
    0.0001 (  0.0%)  (A) DominanceInfo
    0.0001 (  0.0%)  (A) CallGraph
    0.0001 (  0.0%)  Pipeline Collection : ['builtin.module']
    0.0224 (  1.9%)  Rest
    1.1612 (100.0%)  Total

real    0m1.195s
user    0m1.113s
sys     0m0.082s
```

Aggregated timing for 2:
```

===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.2030 seconds

  ----Wall Time----  ----Name----
    0.2030 (100.0%)  root
    0.1084 ( 53.4%)  BiShengHIR
    0.0900 ( 44.3%)  Pipeline Collection : ['func.func']
    0.0899 ( 44.3%)  'func.func' Pipeline
    0.0753 ( 37.1%)  buildFinalHIVMPipelines
    0.0395 ( 19.5%)  Canonicalizer
    0.0197 (  9.7%)  MarkRealCoreType
    0.0157 (  7.7%)  BiShengSIMD
    0.0066 (  3.3%)  InlineScope
    0.0045 (  2.2%)  AutoVectorizeV2
    0.0042 (  2.1%)  CanonicalizeIterArg
    0.0041 (  2.0%)  InsertLoadStoreForMixCV
    0.0036 (  1.7%)  NormalizeTensorOps
    0.0031 (  1.5%)  NormalizeVector
    0.0030 (  1.5%)  HIVMOptSinglePointOp
    0.0029 (  1.4%)  SCFForLoopCanonicalization
    0.0027 (  1.3%)  Inliner
    0.0026 (  1.3%)  HIVMAggregatedDecomposeOp
    0.0023 (  1.2%)  CloneTensorEmpty
    0.0023 (  1.1%)  GraphSyncSolver
    0.0023 (  1.1%)  InferHIVMMemScope
    0.0022 (  1.1%)  NormalizeMatmul
    0.0021 (  1.0%)  SimplifyVFArgs
    0.0020 (  1.0%)  CSE
    0.0019 (  0.9%)  ConvertArithToAffine
    0.0016 (  0.8%)  MemrefDeadStoreEliminationOp
    0.0016 (  0.8%)  OneShotBufferize
    0.0015 (  0.8%)  Normalize
    0.0015 (  0.8%)  TensorCopyInsertion
    0.0015 (  0.7%)  PreVectorizationFusion
    0.0013 (  0.6%)  PlanMemoryRegBase
    0.0013 (  0.6%)  SyncBlockHoisting
    0.0012 (  0.6%)  InsertWorkSpaceForMixCV
    0.0011 (  0.6%)  EnableStrideAlign
    0.0010 (  0.5%)  ConvertHIVMAVEToPTOASVMI
    0.0010 (  0.5%)  InsertFixpipe
    0.0010 (  0.5%)  SymbolDCE
    0.0010 (  0.5%)  CanonicalizeTensorReshape
    0.0009 (  0.5%)  ConvertToHIVMOp
    0.0009 (  0.5%)  VFFusion
    0.0009 (  0.4%)  FoldTensorEmpty
    0.0009 (  0.4%)  LiftLowestStride
    0.0008 (  0.4%)  ConvertArithToHFusion
    0.0008 (  0.4%)  LegalizeBoolPass
    0.0007 (  0.4%)  ConvertHIVMAVEToStandard
    0.0007 (  0.3%)  MarkMultiBuffer
    0.0007 (  0.3%)  ConvertVectorToHIVMAVE
    0.0007 (  0.3%)  HFusionInlineBrc
    0.0007 (  0.3%)  FlattenOps
    0.0007 (  0.3%)  DropEquivalentBufferResults
    0.0006 (  0.3%)  ExpandStridedMetadata
    0.0006 (  0.3%)  FoldExtractInsertPair
    0.0006 (  0.3%)  LowerVectorMaskPass
    0.0006 (  0.3%)  ConvertArithToHIVMAVE
    0.0006 (  0.3%)  HIVMDecomposeOp
    0.0006 (  0.3%)  HIVMNormalize
    0.0006 (  0.3%)  ConvertHIVMTemplatesToPTO
    0.0006 (  0.3%)  SinkOpToConsumerInLoop
    0.0006 (  0.3%)  InsertCVTightCoupledBuffer
    0.0006 (  0.3%)  Decompose
    0.0005 (  0.3%)  ConvertMathToHFusion
    0.0005 (  0.3%)  HIVMFlattenOps
    0.0005 (  0.3%)  LiftZeroRank
    0.0005 (  0.3%)  LoopInvariantSubsetHoisting
    0.0005 (  0.3%)  ConvertHFusionToVector
    0.0005 (  0.3%)  InlineFixpipe
    0.0005 (  0.2%)  ConvertHFusionToHIVM
    0.0005 (  0.2%)  ReduceRankSubview
    0.0005 (  0.2%)  HIVMLowerToLoops
    0.0005 (  0.2%)  OutlineVectorFunction
    0.0005 (  0.2%)  ConvertHIVMToStandard
    0.0004 (  0.2%)  HFusionFoldUnitDims
    0.0004 (  0.2%)  AdaptTritonKernel
    0.0004 (  0.2%)  PullSliceIntoVectorFunction
    0.0004 (  0.2%)  PropagateConvertLayout
    0.0004 (  0.2%)  I1opSoftImpl
    0.0004 (  0.2%)  EnableMultiBuffer
    0.0004 (  0.2%)  AlignAllocSize
    0.0004 (  0.2%)  FuseTransposeIntoLoad
    0.0004 (  0.2%)  AnnotationLowering
    0.0004 (  0.2%)  AVENormalizeOps
    0.0004 (  0.2%)  LegalizeBF16Pass
    0.0004 (  0.2%)  OptimizeDpsOpWithYieldedInsertSlice
    0.0004 (  0.2%)  BindSyncBlockLockArg
    0.0004 (  0.2%)  SimplifyOps
    0.0003 (  0.2%)  LowerMemRefExt
    0.0003 (  0.2%)  ProcessVsstb
    0.0003 (  0.2%)  ReplaceWithVectorScalar
    0.0003 (  0.2%)  PropagateReshape
    0.0003 (  0.2%)  CombineOptimizedConvertLayout
    0.0003 (  0.2%)  MergeVecScope
    0.0003 (  0.2%)  ConvertTensorToHFusion
    0.0003 (  0.2%)  LegalizeLoopIterArgs
    0.0003 (  0.2%)  ConvertGPUToHFusion
    0.0003 (  0.2%)  AnalyzeVectorLayout
    0.0003 (  0.1%)  BindBuffer
    0.0003 (  0.1%)  OutlineAllocInVF
    0.0003 (  0.1%)  VectorizeOps
    0.0003 (  0.1%)  TileAndBindSubBlock
    0.0003 (  0.1%)  OptimizeReductionLoopHIVMAVE
    0.0003 (  0.1%)  OutlineScope
    0.0003 (  0.1%)  RemoveRedundantLoopInit
    0.0003 (  0.1%)  InsertAnchorsAndBackup
    0.0003 (  0.1%)  AllocToAlloca
    0.0003 (  0.1%)  RemoveRedundantWriteAndReadPair
    0.0003 (  0.1%)  OutlineCopyInVF
    0.0003 (  0.1%)  MapForToForall
    0.0003 (  0.1%)  HIVMInlineOTFLoadStore
    0.0003 (  0.1%)  NormalizeLastDimUnalignedTensorOp
    0.0003 (  0.1%)  LowerCreateSyncBlockLock
    0.0003 (  0.1%)  FinalizePTOASVMIBridge
    0.0003 (  0.1%)  InlineOTFBroadcast
    0.0003 (  0.1%)  CombineAVEOPs
    0.0003 (  0.1%)  BindWorkSpaceArg
    0.0003 (  0.1%)  ConstantizeBufferSize
    0.0003 (  0.1%)  InsertL12UBForDebug
    0.0003 (  0.1%)  HIVMComplexReductionIntermediateLowering
    0.0003 (  0.1%)  FoldAllocReshapeOp
    0.0003 (  0.1%)  AveLoopOptimize
    0.0003 (  0.1%)  HIVMMapForallToBlocks
    0.0003 (  0.1%)  ConvertLayoutToTranspose
    0.0003 (  0.1%)  InsertConvertLayout
    0.0003 (  0.1%)  HFusionGeneralizePass
    0.0003 (  0.1%)  LegalizeOptHIVMAVE
    0.0002 (  0.1%)  ProcessMembar
    0.0002 (  0.1%)  TileBatchMMIntoLoop
    0.0002 (  0.1%)  PLTToPGE
    0.0002 (  0.1%)  ScalarBroadcastToVLoad
    0.0002 (  0.1%)  CanonicalizeModule
    0.0002 (  0.1%)  PLTToPLTM
    0.0002 (  0.1%)  AppendTargetDeviceSpec
    0.0002 (  0.1%)  InferFuncCoreType
    0.0002 (  0.1%)  NormalizeArith
    0.0002 (  0.1%)  PrepareI1Nx1ForVectorization
    0.0002 (  0.1%)  AddFFTSAddr
    0.0002 (  0.1%)  ConvertAffineToStandard
    0.0002 (  0.1%)  HoistTensorEmpty
    0.0002 (  0.1%)  LiftArithIndexCast
    0.0002 (  0.1%)  TreeReduceV2
    0.0002 (  0.1%)  LegalizeFP8Pass
    0.0002 (  0.1%)  AutoVectorizeVerifier
    0.0002 (  0.1%)  MaterializeVectorWriteToDestination
    0.0002 (  0.1%)  ConvertTensorToHIVM
    0.0002 (  0.1%)  LoopInvariantPromotion
    0.0002 (  0.1%)  LoopInvariantCodeMotion
    0.0002 (  0.1%)  TritonGlobalKernelArgsToHIVMOp
    0.0002 (  0.1%)  InitEntryKernel
    0.0002 (  0.1%)  GenericUnroller
    0.0002 (  0.1%)  ConvertGenericToNamedOp
    0.0002 (  0.1%)  NormalizeSliceOps
    0.0002 (  0.1%)  RemoveVectorLayoutAttr
    0.0002 (  0.1%)  PreMarkStrideAlign
    0.0002 (  0.1%)  AutoBlockifyParallelLoop
    0.0002 (  0.1%)  ConvertLinalgToHFusion
    0.0002 (  0.1%)  ExposeMemrefWriteToTensor
    0.0002 (  0.1%)  InferHIVMDataLayout
    0.0002 (  0.1%)  InsertInferWorkSpaceSizeFunc
    0.0002 (  0.1%)  LegalizeScalarPass
    0.0001 (  0.1%)  DelayedCrossCoreGSS
    0.0001 (  0.1%)  RemoveMaskFromUnalignedReductionLoop
    0.0001 (  0.1%)  MarkStrideAlign
    0.0001 (  0.1%)  UpliftWhileToFor
    0.0001 (  0.1%)  AnalyzeArithVectorMask
    0.0001 (  0.1%)  InferVFMode
    0.0001 (  0.1%)  LowerMultiBufferCounter
    0.0001 (  0.1%)  MarkTightlyCoupledBuffer
    0.0001 (  0.1%)  SplitMixKernel
    0.0001 (  0.1%)  CrossCoreGSS
    0.0001 (  0.1%)  InjectIR
    0.0001 (  0.1%)  MarkDisableLoad
    0.0001 (  0.1%)  EraseSymbol
    0.0001 (  0.1%)  MarkSyncBlockLockWithSubblock
    0.0001 (  0.1%)  FixCallUnknownLoc
    0.0001 (  0.1%)  InsertInitAndFinishForDebug
    0.0001 (  0.1%)  InsertFreeLockVarBeforeReturn
    0.0001 (  0.1%)  AllocExtraBuffer
    0.0001 (  0.1%)  InsertInferSyncBlockLockNumAndInitFunc
    0.0001 (  0.1%)  AutoInferBufferSize
    0.0001 (  0.1%)  HoistTightlyCoupledAlloc
    0.0001 (  0.1%)  SetBufferSize
    0.0001 (  0.1%)  PeelLoopsContainingTranspose
    0.0000 (  0.0%)  (A) DominanceInfo
    0.0000 (  0.0%)  (A) CallGraph
    0.0000 (  0.0%)  Pipeline Collection : ['builtin.module']
    0.0036 (  1.8%)  Rest
    0.2030 (100.0%)  Total

real    0m0.235s
user    0m0.217s
sys     0m0.018s
```

This at the first look gives the impression that the same passes are running slower in the first flow, but that is not right because this is using `-mlir-timing-display=list` which aggregates the passes with the same name.
This means that in the first flow some passes are executed multiple times, after getting the output with `-mlir-timing-display=tree`, we found out that:

| Pass/stage               | Full compile | `--emit-ptoas-vmi` |
| ------------------------ | -----------: | -----------------: |
| `AutoVectorizeV2`        |       **5×** |             **1×** |
| `NormalizeVector`        |       **6×** |             **1×** |
| `GraphSyncSolver`        |       **6×** |             **1×** |
| `ConvertVectorToHIVMAVE` |       **6×** |             **1×** |
| `InlineScope`            |      **15×** |             **3×** |
| `MarkRealCoreType`       |      **23×** |             **4×** |
| `Canonicalizer`          |     **312×** |            **57×** |

Now the question is do all these repetitions occur after `hivm-ave-to-ptoas-vmi` pass?

also the total compilation time for npuir is 4.5s, when `-mlir-timing` is used (the results above), hivmc-a5 is not included which for the npuir flow gives 1.2s, so hivmc-a5 takes ~ 3.3s ?
