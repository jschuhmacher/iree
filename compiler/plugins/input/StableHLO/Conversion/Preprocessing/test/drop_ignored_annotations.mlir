// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-drop-ignored-annotations))" \
// RUN:   %s | FileCheck %s
// RUN: iree-opt --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-drop-ignored-annotations))" \
// RUN:   %s
// A default result_accuracy is verifier-constrained to atol=rtol=0, ulps=0,
// which StableHLO's pretty printer always elides; check with the generic
// form instead so the untouched attribute is actually visible to FileCheck.
// RUN: iree-opt --split-input-file --mlir-print-op-generic \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-drop-ignored-annotations))" \
// RUN:   %s | FileCheck --check-prefix=GENERIC %s

// CHECK-LABEL: @dot_algorithm
// CHECK: stablehlo.dot_general
// CHECK-NOT: algorithm
func.func @dot_algorithm(%arg0: tensor<4x8xf32>, %arg1: tensor<8x2xf32>) -> tensor<4x2xf32> {
  // expected-warning @+1 {{dropping dot algorithm; IREE does not honour it}}
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
  // expected-warning @+1 {{dropping precision_config; IREE does not honour it}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1],
                                            rhs_contracting_dimensions = [0]>,
    precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]
  } : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// A default precision_config carries no request IREE could fail to honour,
// so it is removed without a warning.
// CHECK-LABEL: @precision_config_default
// CHECK: stablehlo.dot_general
// CHECK-NOT: precision_config
func.func @precision_config_default(%arg0: tensor<4x8xf32>, %arg1: tensor<8x2xf32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1],
                                            rhs_contracting_dimensions = [0]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: @result_accuracy
// CHECK: stablehlo.exponential
// CHECK-NOT: result_accuracy
func.func @result_accuracy(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // expected-warning @+1 {{dropping result_accuracy; IREE does not honour it}}
  %0 = "stablehlo.exponential"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00,
      rtol = 0.000000e+00, ulps = 0,
      mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A default result_accuracy is not a request IREE fails to meet, so it is
// left untouched. The pretty printer elides it regardless of that, so
// presence is checked below under --mlir-print-op-generic (GENERIC-*).
// CHECK-LABEL: @result_accuracy_default
// CHECK: stablehlo.exponential
// GENERIC-LABEL: result_accuracy_default
// GENERIC: stablehlo.exponential
// GENERIC-SAME: result_accuracy
func.func @result_accuracy_default(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.exponential"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<atol = 0.000000e+00,
      rtol = 0.000000e+00, ulps = 0,
      mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
