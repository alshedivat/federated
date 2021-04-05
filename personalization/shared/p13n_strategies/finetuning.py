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

import collections
from typing import Any, Callable, Dict, Optional

import tensorflow as tf
import tensorflow_federated as tff

from personalization.shared import evaluation

# Types.
OptimizerFn = Callable[[], tf.keras.optimizers.Optimizer]
PersonalizeFn = Callable[
    [tff.learning.Model, tf.data.Dataset, tf.data.Dataset, Any],
    Dict[str, tf.Tensor]]


def build_personalize_fn(optimizer_fn: OptimizerFn) -> PersonalizeFn:
  """Builds a `tf.function` that represents a personalization strategy.

  The returned `tf.function` fine-tunes a client model on the `train_data`
  subset of the client dataset and evaluates it on the `test_data` subset.

  Args:
    optimizer_fn: A no-argument function that returns a
      `tf.keras.optimizers.Optimizer`.

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

    # Fine-tune the model on train_data.
    num_train_examples = 0
    for batch in train_data:
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch)
      grads = tape.gradient(output.loss, model.trainable_variables)
      optimizer.apply_gradients(
          zip(tf.nest.flatten(grads),
              tf.nest.flatten(model.trainable_variables)))
      num_train_examples += output.num_examples

    # Evaluate the model.
    metrics_dict = evaluation.evaluate_fn(model, test_data)

    # Save the training statistics.
    metrics_dict['num_train_examples'] = num_train_examples

    return metrics_dict

  return personalize_fn
