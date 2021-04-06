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

from typing import Any, Callable, Dict

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


def create_client_single_data_pass_fn(model_fn: ModelFn,
                                      finetune_optimizer_fn: OptimizerFn):
  """Returns a tf.function for taking a single pass over the client data."""
  # Create the `optimizer` here instead of inside the `tf.function` below,
  # because a `tf.function` generally does not allow creating new variables.
  finetune_optimizer = finetune_optimizer_fn()

  # We need an additional copy of model's trainable weights to be able to
  # reset the weights to the state they were in right before the inner loop
  # fine-tuning when applying gradients in the outer loop.
  model_weights_trainable_init = tuple(model_fn().trainable_variables)

  @tf.function
  def _single_data_pass(model: tff.learning.Model,
                        dataset: Dict[str, tf.data.Dataset],
                        client_optimizer: tf.keras.optimizers.Optimizer):
    """Makes a single pass over the dataset and updates the model.

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

    # Make a pass over the dataset.
    for batch in dataset['test_data']:
      # Update the model's initial trainable weights.
      tf.nest.map_structure(lambda v, x: v.assign(x.value()),
                            model_weights_trainable_init,
                            model_weights_trainable)
      # Run model fine-tuning in the inner loop.
      finetune_fn(model, finetune_optimizer, dataset['train_data'])
      # Do forward pass and compute gradients (using post-fine-tuning weights!).
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model_weights_trainable)
      # Reset model weigths back to the pre-fine-tuned state.
      tf.nest.map_structure(lambda v, x: v.assign(x.value()),
                            model_weights_trainable,
                            model_weights_trainable_init)
      # Update model weights.
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

