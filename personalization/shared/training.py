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
"""Utility functions for federated training."""

import collections
from typing import Any, Callable, Dict, Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared.fed_avg_schedule import build_server_init_fn
from optimization.shared.fed_avg_schedule import create_client_update_fn
from optimization.shared.fed_avg_schedule import server_update

# Convenience type aliases.
ModelBuilder = Callable[[], tff.learning.Model]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientWeightFn = Callable[..., float]
ClientDatasetFn = Callable[[Dict[str, tf.data.Dataset]], Any]
LRScheduleFn = Callable[[Union[int, tf.Tensor]], Union[tf.Tensor, float]]


def _initialize_optimizer_vars(model: tff.learning.Model,
                               optimizer: tf.keras.optimizers.Optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  assert optimizer.variables()


def _get_weights(model: tff.learning.Model) -> tff.learning.ModelWeights:
  return tff.learning.ModelWeights.from_model(model)


def build_fed_training_process(
    model_fn: ModelBuilder,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: Union[float, LRScheduleFn] = 0.1,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: Union[float, LRScheduleFn] = 1.0,
    client_dataset_fn: Optional[ClientDatasetFn] = None,
    client_weight_fn: Optional[ClientWeightFn] = None,
) -> tff.templates.IterativeProcess:
  """Builds the TFF computations for federated training.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    client_optimizer_fn: A function that accepts a `learning_rate` keyword
      argument and returns a `tf.keras.optimizers.Optimizer` instance.
    client_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    server_optimizer_fn: A function that accepts a `learning_rate` argument and
      returns a `tf.keras.optimizers.Optimizer` instance.
    server_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    client_dataset_fn: Optional function that preprocesses client dataset.
      The function must take as input an OrderedDict of datasets.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  client_lr_schedule = client_lr
  if not callable(client_lr_schedule):
    client_lr_schedule = lambda round_num: client_lr

  server_lr_schedule = server_lr
  if not callable(server_lr_schedule):
    server_lr_schedule = lambda round_num: server_lr

  if client_dataset_fn is None:
    client_dataset_fn = lambda x: x

  dummy_model = model_fn()

  server_init_tf = build_server_init_fn(
    model_fn,
    # Initialize with the learning rate for round zero.
    lambda: server_optimizer_fn(server_lr_schedule(0)))
  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  round_num_type = server_state_type.round_num

  # The dataset is actually a struct of train and test datasets.
  tf_dataset_type = tff.StructType([
      ('train_data', tff.SequenceType(dummy_model.input_spec)),
      ('test_data', tff.SequenceType(dummy_model.input_spec))])

  @tff.tf_computation(tf_dataset_type, model_weights_type, round_num_type)
  def client_update_fn(tf_dataset, initial_model_weights, round_num):
    client_lr = client_lr_schedule(round_num)
    client_optimizer = client_optimizer_fn(client_lr)
    client_update = create_client_update_fn()
    tf_dataset = client_dataset_fn(tf_dataset)
    return client_update(model_fn(), tf_dataset, initial_model_weights,
                         client_optimizer, client_weight_fn)

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta)

  @tff.federated_computation(
    tff.type_at_server(server_state_type),
    tff.type_at_clients(tf_dataset_type))
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation`.
    """
    client_model = tff.federated_broadcast(server_state.model)
    client_round_num = tff.federated_broadcast(server_state.round_num)

    client_outputs = tff.federated_map(
      client_update_fn,
      (federated_dataset, client_model, client_round_num))

    client_weight = client_outputs.client_weight
    model_delta = tff.federated_mean(
      client_outputs.weights_delta, weight=client_weight)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, model_delta))

    aggregated_outputs = dummy_model.federated_output_computation(
      client_outputs.model_output)
    if aggregated_outputs.type_signature.is_struct():
      aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  @tff.federated_computation
  def initialize_fn():
    return tff.federated_value(server_init_tf(), tff.SERVER)

  iterative_process = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn, next_fn=run_one_round)

  @tff.tf_computation(server_state_type)
  def get_model_weights(server_state):
    return server_state.model

  iterative_process.get_model_weights = get_model_weights
  return iterative_process
