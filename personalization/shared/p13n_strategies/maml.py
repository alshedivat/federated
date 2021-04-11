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
"""Model-agnostic meta-learning personalization strategy."""

import contextlib
from typing import Any, Callable, Dict, List, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from personalization.shared.p13n_strategies.finetuning import finetune_fn
from posterior_averaging.shared.fed_pa_schedule import DataPassOutput

# Types.
ModelFn = Callable[[], tff.learning.Model]
OptimizerFn = Callable[[], tf.keras.optimizers.Optimizer]
PersonalizeFn = Callable[
  [tff.learning.Model, tf.data.Dataset, tf.data.Dataset, Any],
  Dict[str, tf.Tensor]]
SingleDataPassFn = Callable[..., DataPassOutput]


@contextlib.contextmanager
def _maml_iteration_resets(model_weights_trainable: Tuple[tf.Variable],
                           model_weights_trainable_reset: Tuple[tf.Variable],
                           inner_optimizer: tf.keras.optimizers.Optimizer,
                           inner_optimizer_weights_reset: List[tf.Tensor]):
  """A context manager that does some bookkeeping for MAML.

  MAML requires the following resets:
    1. After the inner loop, MAML computes the outer loop updates using the
       fine tuned parameters of the model. However, these updates must be
       applied to the model parameters before the inner loop. This context
       manager takes care of correctly resetting model parameters.
    2. The inner loop optimizer may have weights (e.g., Adam optimizer has)
       which cannot be re-used from iteration to iteration. Here, we make sure
       these variables are correctly reset.

  Args:
    model_weights_trainable: A list of model trainable weights.
    model_weights_trainable_reset: A list of auxiliary variables used for
      resetting model trainable weights before applying the outer loop updates.
    inner_optimizer: The inner loop optimizer.
    inner_optimizer_weights_reset: A list tensors used to reset the inner loop
      optimizer at the end of each outer loop iteration.
  """
  # Prepare for running the inner loop.
  try:
    # Save the initial model weights before fine tuning.
    tf.nest.map_structure(lambda v, x: v.assign(x.value()),
                          model_weights_trainable_reset,
                          model_weights_trainable)
    # Force the inner optimizer to create weights, if necessary.
    # The values of the created initial weights are saved to the provided list.
    if not inner_optimizer_weights_reset:
      grads_and_vars_dummy = tf.nest.map_structure(
          lambda x: (tf.zeros_like(x), x), model_weights_trainable)
      inner_optimizer.apply_gradients(grads_and_vars_dummy)
      for w in inner_optimizer.weights:
        inner_optimizer_weights_reset.append(tf.identity(w))
    yield
  # Do necessary resets after the outer loop step is done.
  finally:
    # Reset model weights back to the pre-fine-tuned state.
    tf.nest.map_structure(lambda v, x: v.assign(x.value()),
                          model_weights_trainable,
                          model_weights_trainable_reset)
    # Reset the weights of the inner loop optimizer.
    tf.nest.map_structure(lambda v, x: v.assign(x),
                          inner_optimizer.weights,
                          inner_optimizer_weights_reset)


def create_client_single_data_pass_fn(model_fn: ModelFn,
                                      client_steps: int,
                                      inner_optimizer_fn: OptimizerFn,
                                      inner_prox_coeff: float = 0.0) -> SingleDataPassFn:
  """Returns a tf.function for taking a single pass over the client data.

  Args:
    model_fn: A function that builds a model.
    client_steps: The number of outer loops steps done by the client.
    inner_optimizer_fn: A function that builds the inner loop optimizer.
    inner_prox_coeff: A prox coefficient to use in the inner loop.
  """
  # Create the `optimizer` here instead of inside the `tf.function` below,
  # because a `tf.function` generally does not allow creating new variables.
  inner_optimizer = inner_optimizer_fn()
  inner_optimizer_weights_reset = []

  # We need an additional copy of model's trainable weights to be able to
  # reset the weights to the state they were in right before the inner loop
  # when applying gradients in the outer loop.
  model_weights_trainable_reset = tuple(model_fn().trainable_variables)

  @tf.function
  def _single_data_pass(model: tff.learning.Model,
                        dataset: tf.data.Dataset,
                        client_optimizer: tf.keras.optimizers.Optimizer) -> DataPassOutput:
    """Makes a single pass over the dataset and updates the model.

    Notes:
      - Each outer loop step essentially is computed on a randomly sampled data
        batch (the rest of the data is used in the inner loop) =>
        `client_steps = # of data batches` would effectively take a full pass
        over the data in the outer loop.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      client_optimizer: A `tf.keras.optimizer.Optimizer` object.

    Returns:
      A `DataPassOutput` structure.
    """
    loss_avg = tf.constant(0., dtype=tf.float32)
    num_batches = tf.constant(0., dtype=tf.float32)
    num_examples = tf.constant(0., dtype=tf.float32)
    model_weights_trainable = tuple(model.trainable_variables)
    model_weights_trainable_sum = tf.nest.map_structure(
        tf.zeros_like, model_weights_trainable)

    # Determine the number of outer loop steps, if necessary.
    if client_steps > 0:
      outer_steps = client_steps
    else:
      outer_steps = tf.data.experimental.cardinality(dataset)

    # Local outer loop.
    for _ in tf.range(outer_steps):
      # Create an iterator over the dataset.
      dataset_iterator = iter(dataset)
      # Hold out 1 batch for the outer loop step.
      held_out_batch = next(dataset_iterator)
      with _maml_iteration_resets(model_weights_trainable,
                                  model_weights_trainable_reset,
                                  inner_optimizer,
                                  inner_optimizer_weights_reset):
        # Run model fine-tuning in the inner loop.
        finetune_fn(model, inner_optimizer, dataset_iterator, inner_prox_coeff)
        # Do forward pass and compute gradients on the held out batch.
        with tf.GradientTape() as tape:
          output = model.forward_pass(held_out_batch)
        grads = tape.gradient(output.loss, model_weights_trainable)
      # Update model weights using the gradients computed on the held out batch.
      grads_and_vars = zip(grads, model_weights_trainable)
      client_optimizer.apply_gradients(grads_and_vars)
      # Accumulate weights.
      model_weights_trainable_sum = tf.nest.map_structure(
        lambda a, b: tf.add(a, tf.identity(b)),
        model_weights_trainable_sum,
        model_weights_trainable)
      # Accumulate losses.
      batch_size = tf.cast(tf.shape(output.predictions)[0], dtype=tf.float32)
      loss_avg += output.loss * batch_size
      num_examples += batch_size
      num_batches += 1.

    # Compute average loss and weights sample.
    loss_avg = tf.math.divide_no_nan(loss_avg, num_examples)
    model_weights_trainable_sample = tf.nest.map_structure(
      lambda x: tf.math.divide_no_nan(x, num_batches),
      model_weights_trainable_sum)

    outputs = DataPassOutput(
      loss=loss_avg,
      num_examples=num_examples,
      model_weights_trainable=model_weights_trainable,
      model_weights_trainable_sample=model_weights_trainable_sample)

    return outputs

  return _single_data_pass
