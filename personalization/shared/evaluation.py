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
"""Utility functions for p13n evaluation."""

import collections
from typing import Dict

import tensorflow as tf
import tensorflow_federated as tff

# Monkey-patch and disable batch dim removal, since it's unnecessary!
tff.learning.personalization_eval._remove_batch_dim = lambda x: x


@tf.function
def evaluate_fn(model: tff.learning.Model,
                dataset_iterator: tf.data.Iterator) -> Dict[str, tf.Tensor]:
  """Evaluates a model on the given dataset.

  The returned metrics include those given by `model.report_local_outputs`.
  These are specified by the `loss` and `metrics` arguments when the model is
  created by `tff.learning.from_keras_model`. The returned metrics also contain
  an integer metric with name 'num_test_examples'.

  Args:
    model: A `tff.learning.Model` created by `tff.learning.from_keras_model`.
    dataset_iterator: A `tf.data.Iterator` over data batches.

  Returns:
    An `OrderedDict` of metric names to scalar `tf.Tensor`s.
  """
  # Resets the model's local variables. This is necessary because
  # `model.report_local_outputs()` aggregates the metrics from *all* previous
  # calls to `forward_pass` (which include the metrics computed in training).
  # Resetting ensures that the returned metrics are computed on test data.
  # Similar to the `reset_states` method of `tf.keras.metrics.Metric`.
  for var in model.local_variables:
    if var.initial_value is not None:
      var.assign(var.initial_value)
    else:
      var.assign(tf.zeros_like(var))

  # Compute forward pass over the data.
  num_test_examples = 0
  for batch in dataset_iterator:
    output = model.forward_pass(batch, training=False)
    num_test_examples += output.num_examples

  # Collect metrics.
  eval_metrics = collections.OrderedDict()
  eval_metrics['num_test_examples'] = num_test_examples

  # Postprocesses the metric values. This is needed because the values returned
  # by `model.report_local_outputs()` are values of the state variables in each
  # `tf.keras.metrics.Metric`. These values should be processed in the same way
  # as the `result()` method of a `tf.keras.metrics.Metric`.
  local_outputs = model.report_local_outputs()
  for name, metric in local_outputs.items():
    if not isinstance(metric, list):
      raise TypeError(f'The metric value returned by `report_local_outputs` is '
                      f'expected to be a list, but found an instance of '
                      f'{type(metric)}. Please check that your TFF model is '
                      'built from a keras model.')
    if len(metric) == 2:
      eval_metrics[name] = metric[0] / metric[1]
    elif len(metric) == 1:
      eval_metrics[name] = metric[0]
    else:
      raise ValueError(f'The metric value returned by `report_local_outputs` '
                       f'is expected to be a list of length 1 or 2, but found '
                       f'one with length {len(metric)}.')
  return eval_metrics
