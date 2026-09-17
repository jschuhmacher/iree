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

// -----

// CHECK-LABEL: @dynamic_gather
// CHECK: linalg.generic
// CHECK: tensor.extract %{{.+}} : tensor<3x4x2xi32>
// CHECK: return %{{.+}} : tensor<2x3x2x2xi32>
func.func @dynamic_gather(%a: tensor<3x4x2xi32>, %i: tensor<2x3x2xi64>, %s: tensor<3xi64>) -> tensor<2x3x2x2xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi32>
  return %r : tensor<2x3x2x2xi32>
}

// -----

// A dynamic result: the offset dims come from the slice_sizes operand.
// CHECK-LABEL: @dynamic_gather_dynamic_result
// CHECK: tensor.empty(%{{.+}}, %{{.+}}) : tensor<2x3x?x?xi32>
// CHECK: linalg.generic
// CHECK: return %{{.+}} : tensor<2x3x?x?xi32>
func.func @dynamic_gather_dynamic_result(%a: tensor<3x4x2xi32>, %i: tensor<2x3x2xi64>, %s: tensor<3xi64>) -> tensor<2x3x?x?xi32> {
  %r = "stablehlo.dynamic_gather"(%a, %i, %s) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [1, 0], index_vector_dim = 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x?x?xi32>
  return %r : tensor<2x3x?x?xi32>
}

// -----

// No annotation and a dynamic operand dim: whether it expands is decided
// per element at runtime.
// CHECK-LABEL: @dynamic_broadcast_undecidable
// CHECK: linalg.generic
// CHECK: arith.cmpi eq
// CHECK: arith.select
// CHECK: tensor.extract %{{.+}} : tensor<?xf32>
// CHECK: return %{{.+}} : tensor<?x?xf32>
func.func @dynamic_broadcast_undecidable(%a: tensor<?xf32>, %s: tensor<2xi32>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [1] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// The annotated form decides expansion statically, so upstream's own lowering applies.
// CHECK-LABEL: @dynamic_broadcast_annotated
// CHECK: linalg.generic
// CHECK-NOT: arith.select
// CHECK-NOT: tensor.extract
// CHECK: return %{{.+}} : tensor<?x?xf32>
func.func @dynamic_broadcast_annotated(%a: tensor<?xf32>, %s: tensor<2xi32>) -> tensor<?x?xf32> {
  %r = stablehlo.dynamic_broadcast_in_dim %a, %s, dims = [1] {known_nonexpanding_dimensions = array<i64: 0>} : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @dynamic_reduce_window
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x8xf32>, %[[INIT:.+]]: tensor<f32>)
// CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %c0
// CHECK: %[[EMPTY:.+]] = tensor.empty(%[[DIM]]) : tensor<?x4xf32>
// CHECK: linalg.fill ins(%{{.+}} : f32) outs(%[[EMPTY]] : tensor<?x4xf32>)
// CHECK: linalg.generic
// CHECK: arith.maximumf
// CHECK: return %{{.+}} : tensor<?x4xf32>
func.func @dynamic_reduce_window(%a: tensor<?x8xf32>, %i: tensor<f32>) -> tensor<?x4xf32> {
  %r = "stablehlo.reduce_window"(%a, %i) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %m = stablehlo.maximum %x, %y : tensor<f32>
    "stablehlo.return"(%m) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 2>, window_strides = array<i64: 1, 2>, padding = dense<0> : tensor<2x2xi64>} : (tensor<?x8xf32>, tensor<f32>) -> tensor<?x4xf32>
  return %r : tensor<?x4xf32>
}

// -----

// A pooling-shaped reduce_window with a dynamic batch dim: upstream's own
// pattern seeds the dynamic output dim itself, so it applies here too.
// CHECK-LABEL: @dynamic_reduce_window_pooling
// CHECK-NOT: linalg.generic
// CHECK: linalg.pooling_nhwc_max
// CHECK: return %{{.+}} : tensor<?x4x4x3xf32>
func.func @dynamic_reduce_window_pooling(%a: tensor<?x8x8x3xf32>, %i: tensor<f32>) -> tensor<?x4x4x3xf32> {
  %r = "stablehlo.reduce_window"(%a, %i) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %m = stablehlo.maximum %x, %y : tensor<f32>
    "stablehlo.return"(%m) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>, padding = dense<0> : tensor<4x2xi64>} : (tensor<?x8x8x3xf32>, tensor<f32>) -> tensor<?x4x4x3xf32>
  return %r : tensor<?x4x4x3xf32>
}

// -----

// A windowed dynamic dim: the result dim is computed from the input dim.
// CHECK-LABEL: @dynamic_reduce_window_strided
// CHECK: arith.subi
// CHECK: arith.divsi
// CHECK: arith.addi
// CHECK: tensor.empty(%{{.+}}) : tensor<?xf32>
// CHECK: linalg.generic
func.func @dynamic_reduce_window_strided(%a: tensor<?xf32>, %i: tensor<f32>) -> tensor<?xf32> {
  %r = "stablehlo.reduce_window"(%a, %i) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %m = stablehlo.add %x, %y : tensor<f32>
    "stablehlo.return"(%m) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 3>, window_strides = array<i64: 2>, padding = dense<[[1, 1]]> : tensor<1x2xi64>} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}
