add_triton_library(TritonToTritonGPUCustom
  TritonToTritonGPUCustom.cpp

  DEPENDS
  TritonToTritonGPUCustomConversionPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonGPUIR
  TritonIR
)