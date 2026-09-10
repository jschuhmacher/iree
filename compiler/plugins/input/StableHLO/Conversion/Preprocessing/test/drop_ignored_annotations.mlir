// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-drop-ignored-annotations))" \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: @dot_algorithm
// CHECK: stablehlo.dot_general
// CHECK-NOT: algorithm
func.func @dot_algorithm(%arg0: tensor<4x8xf32>, %arg1: tensor<8x2xf32>) -> tensor<4x2xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0],
    algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32,
                 accumulation_type = f32, lhs_component_count = 1,
                 rhs_component_count = 1, num_primitive_operations = 1,
                 allow_imprecise_accumulation = false>
    : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: @precision_config
// CHECK: stablehlo.dot_general
// CHECK-NOT: precision_config
func.func @precision_config(%arg0: tensor<4x8xf32>, %arg1: tensor<8x2xf32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1],
                                            rhs_contracting_dimensions = [0]>,
    precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]
  } : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: @result_accuracy
// CHECK: stablehlo.exponential
// CHECK-NOT: result_accuracy
func.func @result_accuracy(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.exponential"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00,
      rtol = 0.000000e+00, ulps = 0,
      mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
