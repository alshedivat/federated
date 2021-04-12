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
from personalization.shared.p13n_strategies.maml_utils import apply_first_order_maml_update
from personalization.shared.p13n_strategies.maml_utils import apply_implicit_maml_update
from posterior_averaging.shared.fed_pa_schedule import DataPassOutput

# Types.
ModelFn = Callable[[], tff.learning.Model]
OptimizerFn = Callable[[float], tf.keras.optimizers.Optimizer]
LRSchedule = Callable[[tf.Tensor], tf.Tensor]
PersonalizeFn = Callable[
    [tff.learning.Model, tf.data.Dataset, tf.data.Dataset, Any],
    Dict[str, tf.Tensor]]
SingleDataPassFn = Callable[..., DataPassOutput]


def create_client_single_data_pass_fn(round_num: int,
                                      model_fn: ModelFn,
                                      client_steps: int,
                                      inner_optimizer_fn: OptimizerFn,
                                      inner_lr_schedule: LRSchedule,
                                      inner_prox_coeff: float = 0.0,
                                      maml_update_type: str = "first-order") -> SingleDataPassFn:
  """Returns a tf.function for taking a single pass over the client data.

  Args:
    round_num: An integer representing the round number.
    model_fn: A function that builds a model.
    client_steps: The number of outer loops steps done by the client.
    inner_optimizer_fn: A function that builds the inner loop optimizer.
    inner_lr_schedule: A function that maps round number to a learning rate,
      which is used by the inner loop optimizer.
    inner_prox_coeff: A prox coefficient to use in the inner loop.
    maml_update_type: A string that specifies the type of MAML updates.
  """
  # Create the `optimizer` here instead of inside the `tf.function` below,
  # because a `tf.function` generally does not allow creating new variables.
  inner_lr = inner_lr_schedule(round_num)
  inner_optimizer = inner_optimizer_fn(inner_lr)

  # We need an additional copy of model's trainable weights to be able to
  # reset the weights to the state they were in right before the inner loop
  # when applying gradients in the outer loop.
  model_weights_trainable_reset = tuple(model_fn().trainable_variables)

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
         which cannot be re-used if the inner loop is executed multiple times.
         Here, we make sure these variables are correctly reset.

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

    # Create reset tensors for the inner loop optimizer.
    grads_and_vars_dummy = tf.nest.map_structure(
        lambda x: (tf.zeros_like(x), x), model_weights_trainable)
    inner_optimizer.apply_gradients(grads_and_vars_dummy)
    inner_optimizer_weights_reset = tf.nest.map_structure(
        lambda x: tf.identity(x), inner_optimizer.weights)

    # Determine the number of outer loop steps, if necessary.
    if client_steps > 0:
      outer_steps = client_steps
    else:
      outer_steps = tf.data.experimental.cardinality(dataset)

    # Create gradient tape for second order derivatives, if necessary.
    if maml_update_type == 'implicit':
      second_order_tape = tf.GradientTape(persistent=True)

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
        # Enable second order gradient tape, if necessary.
        if maml_update_type == 'implicit':
          second_order_tape._push_tape()
        # Do forward pass and compute gradients on the held out batch.
        with tf.GradientTape() as first_order_tape:
          output = model.forward_pass(held_out_batch)
        grads = first_order_tape.gradient(output.loss, model_weights_trainable)
      # Update model weights using the gradients computed on the held out batch.
      if maml_update_type == "first-order":
        apply_first_order_maml_update(
            vars=model_weights_trainable,
            grads=grads,
            optimizer=client_optimizer)
      elif maml_update_type == "implicit":
        apply_implicit_maml_update(
            vars=model_weights_trainable,
            grads=grads,
            tape=second_order_tape,
            optimizer=client_optimizer,
            prox_coeff=inner_prox_coeff)
        second_order_tape._pop_tape()
      else:
        raise ValueError(f"Unsupported MAML update type: {maml_update_type}")
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
