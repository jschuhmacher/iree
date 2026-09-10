// RUN: iree-opt --split-input-file --iree-stablehlo-verify-compiler-input-legality \
// RUN:   --verify-diagnostics %s

// expected-error@+1 {{one or more illegal operations or types were found in the compiler input}}
module {
func.func @illegal_chlo(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-note@+1 {{failed to legalize operation 'chlo.broadcast_add' that was explicitly marked illegal}}
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
}

// -----
// expected-error@+1 {{one or more illegal operations or types were found in the compiler input}}
module {
func.func @illegal_stablehlo(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-note@+1 {{failed to legalize operation 'stablehlo.add' that was explicitly marked illegal}}
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
}

// -----
// expected-error@+1 {{one or more illegal operations or types were found in the compiler input}}
module {
func.func @illegal_shape(%arg0: tensor<*xf32>) -> index {
  // expected-note@+1 {{failed to legalize operation 'shape.shape_of' that was explicitly marked illegal}}
  %arg_shape = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %rank = shape.rank %arg_shape : tensor<?xindex> -> index
  return %rank : index
}
}

// -----
// A type from an input dialect reaches the runtime ABI if it is not caught
// here, so types are checked as well as operations.
// RUN: not iree-opt --iree-stablehlo-verify-compiler-input-legality \
// RUN:   -o /dev/null 2>&1 %s | FileCheck %s --check-prefix=TYPES
// TYPES: one or more illegal operations or types were found in the compiler input
// expected-error@+1 {{one or more illegal operations or types were found in the compiler input}}
module {
// expected-note@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @illegal_token_type(%arg0: !stablehlo.token) -> !stablehlo.token {
  return %arg0 : !stablehlo.token
}
}

// -----
// expected-error@+1 {{one or more illegal operations or types were found in the compiler input}}
module {
// expected-note@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @illegal_shape_type(%arg0: !shape.witness) -> !shape.witness {
  return %arg0 : !shape.witness
}
}

// -----
// expected-error@+1 {{one or more illegal operations or types were found in the compiler input}}
module {
// expected-note@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @illegal_element_type(%arg0: tensor<4x!stablehlo.token>) -> tensor<4x!stablehlo.token> {
  return %arg0 : tensor<4x!stablehlo.token>
}
}
