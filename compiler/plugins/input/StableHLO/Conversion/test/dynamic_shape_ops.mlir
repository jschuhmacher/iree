// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.

// CHECK-LABEL: @dynamic_reshape
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x2xf32>, %[[SHAPE:.+]]: tensor<1xi64>) -> tensor<?xf32>
// CHECK: %[[R:.+]] = tensor.reshape %[[ARG0]](%[[SHAPE]]) : (tensor<4x2xf32>, tensor<1xi64>) -> tensor<?xf32>
// CHECK: return %[[R]] : tensor<?xf32>
func.func @dynamic_reshape(%a: tensor<4x2xf32>, %s: tensor<1xi64>) -> tensor<?xf32> {
  %r = "stablehlo.dynamic_reshape"(%a, %s) : (tensor<4x2xf32>, tensor<1xi64>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}
