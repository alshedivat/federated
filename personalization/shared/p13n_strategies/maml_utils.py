# Copyright 2021, Maruan Al-Shedivat.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for model-agnostic meta-learning."""

from typing import Tuple

import tensorflow as tf


def _to_vec(tensors):
  return tf.concat(
      tf.nest.map_structure(lambda x: tf.reshape(x, [-1]), tensors), axis=0)


class LinearOperatorRegularizedHessian(tf.linalg.LinearOperator):
  """A linear operator that implements efficient matrix-vector product.

  The operator represents `alpha * I + H`, where `H` is a Hessian.
  """

  def __init__(self,
               grads: Tuple[tf.Tensor],
               vars: Tuple[tf.Tensor],
               tape: tf.GradientTape,
               alpha: float,
               name: str = "LinearOperatorRegularizedHessian"):
    """Initialize LinearOperatorHessian.

    Args:
      grads:
      vars:
      tape:
      alpha:
      name:
    """
    parameters = dict(alpha=alpha, grads=grads, vars=vars, tape=tape)

    self._alpha = alpha
    self._grads = grads
    self._vars = vars
    self._tape = tape

    super(LinearOperatorRegularizedHessian, self).__init__(
      dtype=tf.float32,
      is_non_singular=None,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True,
      name=name,
      parameters=parameters)

  def _shape(self):
    return tf.TensorShape((None, None))

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    """Implements efficient Hessian-vector product."""
    grads_vec = _to_vec(self._grads)
    grad_x_mul = tf.matmul(grads_vec[None, ...], tf.stop_gradient(x),
                           adjoint_a=adjoint, adjoint_b=adjoint_arg)
    with self._tape.stop_recording():
      hess_x_mul_vec = _to_vec(self._tape.gradient(grad_x_mul, self._vars))
    return self._alpha * x + hess_x_mul_vec[..., None]


def apply_implicit_maml_update(vars: Tuple[tf.Variable],
                               grads: Tuple[tf.Tensor],
                               tape: tf.GradientTape,
                               optimizer: tf.keras.optimizers.Optimizer,
                               prox_coeff: float,
                               cg_tol=1e-5,
                               cg_max_iter=20):
  """Computes the iMAML update using CG and applies to the model.

  The update has the following form: `(I + Hess / prox_coeff)^{-1} grads`,
  where:
    - `I` is the identity matrix, and
    - `Hess` is the Hessian (or gradient of the provided `grads`).
  Note that we do not explicitly compute or store the Hessian matrix; instead,
  we use `tf.linalg.LinearOperator` to only compute Hessian-vector products.

  Args:
    vars: A tuple of model weights.
    grads: A tuple of tensors that represent the gradients of the model weights.
    tape: The gradient tape that can be used to differentiate `grads` with
      respect to model parameters.
    optimizer: An instance of `tf.keras.optimizers.Optimizer`.
    prox_coeff: The prox coefficient used in the inner loop.
    cg_tol: Conjugate gradient (CG) tolerance coefficient.
    cg_max_iter: The limit on the number of CG iterations.

  References:
    - Meta-learing with implicit gradients.
        Aravind Rajeswaran, Chelsea Finn, Sham Kakade, Sergey Levine
        NeurIPS 2019 (https://arxiv.org/abs/1909.04630)
  """
  # Create a linear operator, initial solution x, and rhs.
  operator = LinearOperatorRegularizedHessian(
      grads=grads, vars=vars, tape=tape, alpha=prox_coeff)
  x = _to_vec(grads)
  rhs = prox_coeff * x

  # Solve the linear system using CG.
  cg_output = tf.linalg.experimental.conjugate_gradient(operator=operator,
                                                        rhs=rhs, x=x,
                                                        tol=cg_tol,
                                                        max_iter=cg_max_iter)

  # Convert the vector of updates to a tuple of the same structure as variables.
  vars_flat = tf.nest.map_structure(lambda x: tf.reshape(x, [-1]), vars)
  sizes = tf.nest.map_structure(lambda x: tf.size(x), vars_flat)
  updates = tf.nest.map_structure(lambda u, v: tf.reshape(u, tf.shape(v)),
                                  tuple(tf.split(cg_output.x, sizes)), vars)

  # Apply computed updates.
  updates_and_vars = zip(updates, vars)
  optimizer.apply_gradients(updates_and_vars)


def apply_first_order_maml_update(vars: Tuple[tf.Variable],
                                  grads: Tuple[tf.Tensor],
                                  optimizer: tf.keras.optimizers.Optimizer):
  """Computes the simple first order MAML update.

  Args:
    vars: A tuple of model weights.
    grads: A tuple of tensors that represent the gradients of the model weights.
    optimizer: An optimizer.
  """
  grads_and_vars = zip(grads, vars)
  optimizer.apply_gradients(grads_and_vars)
