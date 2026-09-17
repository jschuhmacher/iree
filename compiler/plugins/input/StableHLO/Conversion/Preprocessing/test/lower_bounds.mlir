// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-lower-bounds))" \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: @bounds_in_signature
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?xf32>) -> tensor<?xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[ASSUMED:.+]] = util.assume.int %[[DIM]]<umax = 8> : index
// CHECK: %[[TIED:.+]] = flow.tensor.tie_shape %[[ARG0]] : tensor<?xf32>{%[[ASSUMED]]}
// CHECK: %[[ABS:.+]] = stablehlo.abs %[[TIED]] : tensor<?xf32>
// CHECK: %[[DIM1:.+]] = tensor.dim %[[ABS]], %[[C0]]
// CHECK: %[[ASSUMED1:.+]] = util.assume.int %[[DIM1]]<umax = 8> : index
// CHECK: %[[TIED1:.+]] = flow.tensor.tie_shape %[[ABS]] : tensor<?xf32>{%[[ASSUMED1]]}
// CHECK: return %[[TIED1]] : tensor<?xf32>
func.func @bounds_in_signature(%arg0: tensor<?xf32, #stablehlo.bounds<8>>)
    -> tensor<?xf32, #stablehlo.bounds<8>> {
  %0 = stablehlo.abs %arg0 : tensor<?xf32, #stablehlo.bounds<8>>
  return %0 : tensor<?xf32, #stablehlo.bounds<8>>
}

// -----

// A `?` bound leaves that dimension unassumed; static dims get nothing.
// CHECK-LABEL: @partial_bounds
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x8x?xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[A0:.+]] = util.assume.int %[[D0]]<umax = 16> : index
// CHECK: %[[D2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
// CHECK-NOT: util.assume.int %[[D2]]
// CHECK: flow.tensor.tie_shape %[[ARG0]] : tensor<?x8x?xf32>{%[[A0]], %[[D2]]}
func.func @partial_bounds(%arg0: tensor<?x8x?xf32, #stablehlo.bounds<16, ?, ?>>) -> tensor<?x8x?xf32> {
  %0 = stablehlo.abs %arg0 : (tensor<?x8x?xf32, #stablehlo.bounds<16, ?, ?>>) -> tensor<?x8x?xf32>
  return %0 : tensor<?x8x?xf32>
}
