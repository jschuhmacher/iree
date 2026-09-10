// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Drops the StableHLO annotations IREE has no representation for, so that a
// request IREE cannot meet is reported.

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_DROPIGNOREDANNOTATIONS
#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h.inc"

namespace {

bool isNonDefaultPrecision(ArrayAttr config) {
  return llvm::any_of(config.getAsRange<mlir::stablehlo::PrecisionAttr>(),
                      [](mlir::stablehlo::PrecisionAttr precision) {
                        return precision.getValue() !=
                               mlir::stablehlo::Precision::DEFAULT;
                      });
}

void dropAccuracyHints(Operation *op) {
  if (op->removeAttr("algorithm")) {
    op->emitWarning("dropping dot algorithm; IREE does not honour it");
  }

  if (auto config = op->getAttrOfType<ArrayAttr>("precision_config")) {
    if (isNonDefaultPrecision(config)) {
      op->emitWarning("dropping precision_config; IREE does not honour it");
    }
    op->removeAttr("precision_config");
  }

  if (auto accuracy = op->getAttrOfType<mlir::stablehlo::ResultAccuracyAttr>(
          "result_accuracy")) {
    if (accuracy.getMode().getValue() !=
        mlir::stablehlo::ResultAccuracyMode::DEFAULT) {
      op->emitWarning("dropping result_accuracy; IREE does not honour it");
      op->removeAttr("result_accuracy");
    }
  }
}

Type dropBounds(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !isa_and_present<mlir::stablehlo::TypeExtensionsAttr>(
                         tensorType.getEncoding())) {
    return type;
  }
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType());
}

bool dropBoundsInPlace(Value value) {
  Type dropped = dropBounds(value.getType());
  if (dropped == value.getType()) {
    return false;
  }
  value.setType(dropped);
  return true;
}

struct DropIgnoredAnnotations final
    : impl::DropIgnoredAnnotationsBase<DropIgnoredAnnotations> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    // getLoadedDialect can return null and make the equality below match
    // every unregistered op; force the dialect loaded instead.
    Dialect *stablehloDialect =
        getContext().getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
    bool droppedBounds = false;

    funcOp.walk([&](Operation *op) {
      if (op->getDialect() == stablehloDialect) {
        dropAccuracyHints(op);
      }
      for (Value result : op->getResults()) {
        droppedBounds |= dropBoundsInPlace(result);
      }
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            droppedBounds |= dropBoundsInPlace(arg);
          }
        }
      }
    });

    if (!droppedBounds) {
      return;
    }

    // The signature is not reached by the walk above.
    FunctionType oldType = funcOp.getFunctionType();
    funcOp.setType(FunctionType::get(
        &getContext(), llvm::map_to_vector(oldType.getInputs(), dropBounds),
        llvm::map_to_vector(oldType.getResults(), dropBounds)));
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
