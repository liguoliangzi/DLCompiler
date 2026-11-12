#ifndef TRITON_TO_TRITONGPU_CUSTOM_CONVERSION_PASSES
#define TRITON_TO_TRITONGPU_CUSTOM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToTritonGPUCustom
    : Pass<"triton-to-tritongpu-custom", "mlir::ModuleOp"> {
  let summary = "Convert DICP custom ops to TritonGPU custom ops";
  let description = [{
    Lower the custom op emitted by `dl.custom()` (hivm.custom at this stage)
    to a `triton_gpu.custom` op so that it carries tensor layouts and custom
    op attributes (symbol / bitcode / maca.*) through the TritonGPU passes.
  }];
  let constructor = "triton::createTritonToTritonGPUCustomPass()";
  let dependentDialects = ["triton::gpu::TritonGPUDialect"];
}

def LowerTritonGPUCustom
    : Pass<"lower-tritongpu-custom", "mlir::ModuleOp"> {
  let summary = "Lower TritonGPU custom ops to extern elementwise calls";
  let description = [{
    Expand `triton_gpu.custom` into `tt.extern_elementwise` calls of the
    user-provided bitcode symbol. The bitcode path is kept on the generated
    op so backends can collect it from metadata and link the library.
  }];
  let constructor = "triton::createLowerTritonGPUCustomPass()";
  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
  ];
}

#endif // TRITON_TO_TRITONGPU_CUSTOM_CONVERSION_PASSES