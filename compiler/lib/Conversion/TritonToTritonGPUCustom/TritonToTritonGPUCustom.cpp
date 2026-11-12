#include "dicp/Conversion/TritonToTritonGPUCustom/Passes.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOTRITONGPUCUSTOM
#define GEN_PASS_DEF_LOWERTRITONGPUCUSTOM
#include "dicp/Conversion/TritonToTritonGPUCustom/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {

// Attributes that belong to the hivm custom op itself (or its ascend
// scheduling model) and must not be copied verbatim onto the TritonGPU
// custom op.
static bool isHivmCustomOpLocalAttr(llvm::StringRef name) {
  return name == "name" || name == "operandSegmentSizes" ||
         name.starts_with("hivm.");
}

// Step 1: hivm.custom (created by the frontend `dl.custom()`) ->
// triton_gpu.custom. The op keeps its custom attributes (symbol, bitcode,
// maca.*) and receives the layout of its output operands, so it can flow
// through the regular TritonGPU layout and scheduling passes.
struct HIVMCustomOpToTritonGPUCustomOpConversion
    : OpRewritePattern<hivm::CustomOp> {
  using OpRewritePattern<hivm::CustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::CustomOp op,
                                PatternRewriter &rewriter) const final {
    TypeRange resultTypes = op.getOutputs().getTypes();
    auto newOp = rewriter.create<triton::gpu::CustomOp>(
        op.getLoc(), resultTypes, op.getName(), op.getInputs(),
        op.getOutputs());
    for (auto &attr : op->getAttrs()) {
      if (isHivmCustomOpLocalAttr(attr.getName().strref()))
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// Step 2: triton_gpu.custom -> tt.extern_elementwise calls of the user
// bitcode symbol. The lowering keeps the bitcode path attribute so the
// backend can collect it from metadata and link the library.
struct TritonGPUCustomOpToExternElementwiseConversion
    : OpRewritePattern<triton::gpu::CustomOp> {
  using OpRewritePattern<triton::gpu::CustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::CustomOp op,
                                PatternRewriter &rewriter) const final {
    auto symbolAttr = op->getAttrOfType<StringAttr>("symbol");
    if (!symbolAttr)
      return rewriter.notifyMatchFailure(op, "missing 'symbol' attribute");

    auto bitcodeAttr = op->getAttrOfType<StringAttr>("bitcode");
    if (!bitcodeAttr)
      return rewriter.notifyMatchFailure(op, "missing 'bitcode' attribute");

    auto results = op.getResults();
    if (results.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "only single-result elementwise custom ops are supported");

    auto resultTy = dyn_cast<RankedTensorType>(results[0].getType());
    if (!resultTy || !resultTy.getEncoding())
      return rewriter.notifyMatchFailure(
          op, "result must be a tensor with a TritonGPU layout");

    // Unify tensor input layouts with the result layout so that the
    // generated extern call reads matching per-thread elements.
    SmallVector<Value> srcs;
    for (Value input : op.getInputs()) {
      auto inputTy = dyn_cast<RankedTensorType>(input.getType());
      if (!inputTy) {
        // Scalars are passed through unchanged.
        srcs.push_back(input);
        continue;
      }
      if (inputTy.getShape() != resultTy.getShape())
        return rewriter.notifyMatchFailure(
            op, "custom op inputs must have the same shape as the result");
      if (inputTy.getEncoding() == resultTy.getEncoding()) {
        srcs.push_back(input);
        continue;
      }
      auto unifiedTy = RankedTensorType::get(
          inputTy.getShape(), inputTy.getElementType(), resultTy.getEncoding());
      srcs.push_back(rewriter.create<triton::gpu::ConvertLayoutOp>(
          op.getLoc(), unifiedTy, input));
    }

    auto externOp = rewriter.create<triton::ExternElementwiseOp>(
        op.getLoc(), resultTy, srcs,
        /*libname=*/symbolAttr,
        /*libpath=*/bitcodeAttr,
        /*symbol=*/symbolAttr,
        /*pure=*/rewriter.getBoolAttr(true));
    // Marker consumed by the backend to collect bitcode libraries.
    externOp->setAttr("bitcode", bitcodeAttr);
    rewriter.replaceOp(op, externOp.getResult());
    return success();
  }
};

} // end anonymous namespace

namespace {

struct TritonToTritonGPUCustomPass
    : public mlir::triton::impl::TritonToTritonGPUCustomBase<
          TritonToTritonGPUCustomPass> {
  void runOnOperation() override;
};

struct LowerTritonGPUCustomPass
    : public mlir::triton::impl::LowerTritonGPUCustomBase<
          LowerTritonGPUCustomPass> {
  void runOnOperation() override;
};

} // namespace

void TritonToTritonGPUCustomPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<HIVMCustomOpToTritonGPUCustomOpConversion>(
      patterns.getContext());
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
}

void LowerTritonGPUCustomPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(&getContext());
  patterns.add<TritonGPUCustomOpToExternElementwiseConversion>(
      patterns.getContext());
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToTritonGPUCustomPass() {
  return std::make_unique<TritonToTritonGPUCustomPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createLowerTritonGPUCustomPass() {
  return std::make_unique<LowerTritonGPUCustomPass>();
}