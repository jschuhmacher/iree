// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowerings for the StableHLO ops whose shapes are known only at runtime.

#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/LegalizeToLinalgUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {

// Reads element `i` of a 1-D shape operand as an index.
Value extractIndex(OpBuilder &b, Location loc, Value shapeTensor, int64_t i) {
  Value index = arith::ConstantIndexOp::create(b, loc, i);
  Value element = tensor::ExtractOp::create(b, loc, shapeTensor, index);
  if (element.getType().isIndex()) {
    return element;
  }
  return arith::IndexCastOp::create(b, loc, b.getIndexType(), element);
}

// tensor.reshape takes the shape tensor as-is, and Flow lowers it.
struct DynamicReshapeOpConversion final
    : OpConversionPattern<mlir::stablehlo::DynamicReshapeOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        op, resultType, adaptor.getOperand(), adaptor.getOutputShape());
    return success();
  }
};

} // namespace

void populateDynamicShapeConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet *patterns) {
  // Higher benefit than the upstream patterns, which decline these forms.
  patterns->add<DynamicReshapeOpConversion>(typeConverter, context,
                                            PatternBenefit{1000});
}

} // namespace mlir::iree_compiler::stablehlo
