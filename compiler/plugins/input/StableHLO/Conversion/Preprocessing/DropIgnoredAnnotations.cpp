// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Drops the StableHLO annotations IREE has no representation for, so that a
// request IREE cannot meet is reported.

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
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

struct DropIgnoredAnnotations final
    : impl::DropIgnoredAnnotationsBase<DropIgnoredAnnotations> {
  void runOnOperation() override {
    Dialect *stablehloDialect =
        getContext().getLoadedDialect<mlir::stablehlo::StablehloDialect>();

    getOperation().walk([&](Operation *op) {
      if (op->getDialect() == stablehloDialect) {
        dropAccuracyHints(op);
      }
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
