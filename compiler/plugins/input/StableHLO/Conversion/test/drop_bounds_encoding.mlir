// RUN: iree-opt --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.

// DropIgnoredAnnotations runs before the pass that stamps iree.abi.encoding,
// so a bounds encoding cannot survive into that attribute either.
// CHECK-LABEL: func.func @bounds_in_signature
func.func @bounds_in_signature(%arg0: tensor<?xf32, #stablehlo.bounds<8>>)
    -> tensor<?xf32, #stablehlo.bounds<8>> {
  %0 = stablehlo.abs %arg0 : tensor<?xf32, #stablehlo.bounds<8>>
  return %0 : tensor<?xf32, #stablehlo.bounds<8>>
}
