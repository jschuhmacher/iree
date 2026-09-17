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

// -----

// CHECK-LABEL: @dynamic_conv_lhs_dilation
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK: return %{{.+}} : tensor<1x5x5x1xf32>
func.func @dynamic_conv_lhs_dilation(%a: tensor<1x4x4x1xf32>, %k: tensor<3x3x1x1xf32>, %p: tensor<2x2xi64>) -> tensor<1x5x5x1xf32> {
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>, lhs_dilation = array<i64: 2, 2>, rhs_dilation = array<i64: 1, 1>
  } : (tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>, tensor<2x2xi64>) -> tensor<1x5x5x1xf32>
  return %r : tensor<1x5x5x1xf32>
}

// -----

// The padding amounts are runtime values.
// CHECK-LABEL: @dynamic_pad
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4xf32>, %[[VAL:.+]]: tensor<f32>, %[[LO:.+]]: tensor<1xi64>, %[[HI:.+]]: tensor<1xi64>, %[[IN:.+]]: tensor<1xi64>)
// CHECK-DAG: %[[LOWX:.+]] = tensor.extract %[[LO]]
// CHECK-DAG: %[[LOWV:.+]] = arith.index_cast %[[LOWX]]
// CHECK-DAG: %[[HIGHX:.+]] = tensor.extract %[[HI]]
// CHECK-DAG: %[[HIGHV:.+]] = arith.index_cast %[[HIGHX]]
// CHECK-DAG: %[[INX:.+]] = tensor.extract %[[IN]]
// CHECK-DAG: %[[INV:.+]] = arith.index_cast %[[INX]]
// CHECK-DAG: %[[LOWPOS:.+]] = arith.maxsi %[[LOWV]], %c0
// CHECK-DAG: %[[HIGHPOS:.+]] = arith.maxsi %[[HIGHV]], %c0
// CHECK-DAG: %[[LOWNEG:.+]] = arith.maxsi %{{.+}}, %c0
// CHECK-DAG: %[[FILLDIM:.+]] = arith.addi %{{.+}}, %[[HIGHPOS]]
// CHECK-DAG: %[[RESULTDIM:.+]] = arith.addi %{{.+}}, %[[HIGHV]]
// CHECK-DAG: %[[STRIDE:.+]] = arith.addi %[[INV]], %c1
// CHECK: %[[EMPTY:.+]] = tensor.empty(%[[FILLDIM]]) : tensor<?xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[EMPTY]] : tensor<?xf32>)
// CHECK: %[[INS:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][%[[LOWPOS]]] [4] [%[[STRIDE]]] : tensor<4xf32> into tensor<?xf32>
// CHECK: %[[R:.+]] = tensor.extract_slice %[[INS]][%[[LOWNEG]]] [%[[RESULTDIM]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK: return %[[R]]
func.func @dynamic_pad(%a: tensor<4xf32>, %v: tensor<f32>, %lo: tensor<1xi64>, %hi: tensor<1xi64>, %in: tensor<1xi64>) -> tensor<?xf32> {
  %r = "stablehlo.dynamic_pad"(%a, %v, %lo, %hi, %in) : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// The padding amounts are runtime values but the result shape is static.
// CHECK-LABEL: @dynamic_pad_static_result
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x3xf32>, %[[VAL:.+]]: tensor<f32>, %[[LO:.+]]: tensor<2xi64>, %[[HI:.+]]: tensor<2xi64>, %[[IN:.+]]: tensor<2xi64>) -> tensor<5x9xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[EMPTY]] : tensor<?x?xf32>)
// CHECK: %[[INS:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][%{{.+}}, %{{.+}}] [2, 3] [%{{.+}}, %{{.+}}] : tensor<2x3xf32> into tensor<?x?xf32>
// CHECK: %[[R:.+]] = tensor.extract_slice %[[INS]][%{{.+}}, %{{.+}}] [5, 9] [1, 1] : tensor<?x?xf32> to tensor<5x9xf32>
// CHECK: return %[[R]] : tensor<5x9xf32>
func.func @dynamic_pad_static_result(%a: tensor<2x3xf32>, %v: tensor<f32>, %lo: tensor<2xi64>, %hi: tensor<2xi64>, %in: tensor<2xi64>) -> tensor<5x9xf32> {
  %r = "stablehlo.dynamic_pad"(%a, %v, %lo, %hi, %in) : (tensor<2x3xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x9xf32>
  return %r : tensor<5x9xf32>
}

// -----

// CHECK-LABEL: @dynamic_conv
// CHECK: tensor.insert_slice
// CHECK: linalg.conv_2d_nhwc_hwcf
// CHECK: return %{{.+}} : tensor<1x8x8x1xf32>
func.func @dynamic_conv(%a: tensor<1x8x8x1xf32>, %k: tensor<3x3x1x1xf32>, %p: tensor<2x2xi64>) -> tensor<1x8x8x1xf32> {
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>, lhs_dilation = array<i64: 1, 1>, rhs_dilation = array<i64: 1, 1>
  } : (tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>, tensor<2x2xi64>) -> tensor<1x8x8x1xf32>
  return %r : tensor<1x8x8x1xf32>
}
