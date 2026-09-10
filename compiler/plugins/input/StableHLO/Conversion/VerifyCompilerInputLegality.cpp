// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/PassDetail.h"
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_VERIFYCOMPILERSTABLEHLOINPUTLEGALITY
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

struct VerifyCompilerStableHloInputLegality final
    : impl::VerifyCompilerStableHloInputLegalityBase<
          VerifyCompilerStableHloInputLegality> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget conversionTarget(*context);
    RewritePatternSet conversionPatterns(context);

    // Note that we would prefer allow-lists of what we positively support.
    // However, it is so common to sneak input-level ops into the pipeline
    // that we explicitly deny the dialects we know about.
    conversionTarget.addIllegalDialect<mlir::stablehlo::StablehloDialect>();
    conversionTarget.addIllegalDialect<mlir::chlo::ChloDialect>();
    conversionTarget.addIllegalDialect<mlir::vhlo::VhloDialect>();
    conversionTarget.addIllegalDialect<mlir::shape::ShapeDialect>();

    // Ops are not the only way an input dialect escapes: a type left in a
    // signature reaches the runtime ABI.
    auto isInputDialectType = [](Type type) {
      // `Type::getDialect` and `Attribute::getDialect` both return a
      // reference, which `isa` takes directly.
      if (isa<mlir::stablehlo::StablehloDialect, mlir::chlo::ChloDialect,
              mlir::vhlo::VhloDialect>(type.getDialect())) {
        return true;
      }
      auto tensorType = dyn_cast<RankedTensorType>(type);
      Attribute encoding = tensorType ? tensorType.getEncoding() : nullptr;
      return encoding &&
             isa<mlir::stablehlo::StablehloDialect, mlir::chlo::ChloDialect,
                 mlir::vhlo::VhloDialect>(encoding.getDialect());
    };

    conversionTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
      auto isLegal = [&](Type type) { return !isInputDialectType(type); };
      if (!llvm::all_of(op->getOperandTypes(), isLegal) ||
          !llvm::all_of(op->getResultTypes(), isLegal)) {
        return false;
      }
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        if (!llvm::all_of(funcOp.getArgumentTypes(), isLegal) ||
            !llvm::all_of(funcOp.getResultTypes(), isLegal)) {
          return false;
        }
      }
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          if (!llvm::all_of(block.getArgumentTypes(), isLegal)) {
            return false;
          }
        }
      }
      return true;
    });

    // NOTE: It is not fully illegal to tunnel input dialect ops through to
    // backends that expect them. When such situations arise, the container
    // op should be marked recursively legal here.
    SmallVector<Diagnostic> failures;
    {
      ScopedDiagnosticHandler diag(context,
                                   [&](Diagnostic &d) -> LogicalResult {
                                     failures.push_back(std::move(d));
                                     return success();
                                   });
      if (succeeded(applyPartialConversion(getOperation(), conversionTarget,
                                           std::move(conversionPatterns)))) {
        return;
      }
    }

    // Error fall-through. Attach all reported issues as notes.
    InFlightDiagnostic errorDiag =
        emitError(getOperation().getLoc())
        << "one or more illegal operations or types were found in the "
           "compiler input "
           "(are you missing an --iree-input-type= flag, or did you mean to "
           "pre-process through an IREE importer frontend?)";
    for (Diagnostic &failureDiag : failures) {
      Diagnostic &note = errorDiag.attachNote(failureDiag.getLocation());
      for (auto &arg : failureDiag.getArguments()) {
        note.append(arg);
      }
    }

    signalPassFailure();
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
