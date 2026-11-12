#ifndef DICP_TRITON_TO_TRITONGPU_CUSTOM_CONVERSION_PASSES_H
#define DICP_TRITON_TO_TRITONGPU_CUSTOM_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declarations.
class ModuleOp;

namespace triton {

/// Creates a pass that converts DICP custom ops (hivm.custom) emitted by
/// `dl.custom()` into `triton_gpu.custom` ops with layout-aware types.
std::unique_ptr<OperationPass<ModuleOp>> createTritonToTritonGPUCustomPass();

/// Creates a pass that lowers `triton_gpu.custom` ops to extern elementwise
/// calls of the user bitcode symbol.
std::unique_ptr<OperationPass<ModuleOp>> createLowerTritonGPUCustomPass();

#define GEN_PASS_REGISTRATION
#include "dicp/Conversion/TritonToTritonGPUCustom/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // DICP_TRITON_TO_TRITONGPU_CUSTOM_CONVERSION_PASSES_H