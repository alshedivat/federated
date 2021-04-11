# Copyright 2020, The TensorFlow Federated Authors.
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
"""A simple fine-tuning personalization strategy."""

from typing import Any, Callable, Dict, Optional

import tensorflow as tf
import tensorflow_federated as tff

from personalization.shared import evaluation

# Types.
OptimizerFn = Callable[[], tf.keras.optimizers.Optimizer]
PersonalizeFn = Callable[
    [tff.learning.Model, tf.data.Dataset, tf.data.Dataset, Any],
    Dict[str, tf.Tensor]]


@tf.function
def finetune_fn(model: tff.learning.Model,
                optimizer: tf.keras.optimizers.Optimizer,
                dataset_iterator: tf.data.Iterator,
                prox_coeff: float = 0.0) -> tf.Tensor:
  """Runs the finetuning process of `model` on `dataset` using `optimizer`.

  Args:
    model: A `tff.learning.Model`.
    optimizer: A `tf.keras.optimizers.Optimizer` instance.
    dataset_iterator: A an iterator over `tf.data.Dataset` batches.
    prox_coeff: A float representing proximal regularization coefficient.

  Returns:
    A `tf.Tensor` that contains the number of examples seen during fine tuning.
  """
  init_trainable_variables = tf.nest.map_structure(lambda x: tf.identity(x),
                                                   model.trainable_variables)
  num_train_examples = 0
  for batch in dataset_iterator:
    with tf.GradientTape() as tape:
      output = model.forward_pass(batch)
      loss = output.loss
      if prox_coeff > 0:
        loss += prox_coeff * sum(tf.nest.map_structure(
            lambda x, y: 0.5 * tf.reduce_sum(tf.math.square(x - y)),
            model.trainable_variables, init_trainable_variables))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    num_train_examples += output.num_examples
  return num_train_examples


def build_personalize_fn(optimizer_fn: OptimizerFn,
                         prox_coeff: float = 0.0) -> PersonalizeFn:
  """Builds a `tf.function` that represents a personalization strategy.

  The returned `tf.function` fine-tunes a client model on the `train_data`
  subset of the client dataset and evaluates it on the `test_data` subset.

  Args:
    optimizer_fn: A no-argument function that returns a
      `tf.keras.optimizers.Optimizer`.
    prox_coeff: A float representing proximal regularization coefficient.

  Returns:
    A `tf.function` that trains a personalized model, evaluates the model at
    after that, and returns the evaluation metrics.
  """
  # Create the `optimizer` here instead of inside the `tf.function` below,
  # because a `tf.function` generally does not allow creating new variables.
  optimizer = optimizer_fn()

  @tf.function
  def personalize_fn(model: tff.learning.Model,
                     train_data: tf.data.Dataset,
                     test_data: tf.data.Dataset,
                     context: Optional[Any] = None) -> Dict[str, tf.Tensor]:
    """A personalization strategy that trains a model and returns the metrics.

    Args:
      model: A `tff.learning.Model`.
      train_data: A batched `tf.data.Dataset` used for training.
      test_data: A batched `tf.data.Dataset` used for evaluation.
      context: An optional object (e.g., extra dataset) used in personalization.

    Returns:
      An `OrderedDict` that maps a metric name to `tf.Tensor`s containing the
      evaluation metrics.
    """
    del context  # Fine-tuning strategy does not use extra context.

    # Fine-tune the model.
    num_train_examples = finetune_fn(model=model,
                                     optimizer=optimizer,
                                     dataset_iterator=iter(train_data),
                                     prox_coeff=prox_coeff)

    # Evaluate the model.
    metrics_dict = evaluation.evaluate_fn(model=model,
                                          dataset_iterator=iter(test_data))

    # Save the training statistics.
    metrics_dict['num_train_examples'] = num_train_examples

    return metrics_dict

  return personalize_fn
