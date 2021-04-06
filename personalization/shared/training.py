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

from typing import Any, Callable, Dict, Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from posterior_averaging.shared.fed_pa_schedule import _initialize_optimizer_vars
from posterior_averaging.shared.fed_pa_schedule import build_federated_mean_masked
from posterior_averaging.shared.fed_pa_schedule import build_server_init_fn
from posterior_averaging.shared.fed_pa_schedule import client_update
from posterior_averaging.shared.fed_pa_schedule import DataPassOutput
from posterior_averaging.shared.fed_pa_schedule import server_update

# Import type aliases.
from posterior_averaging.shared.fed_pa_schedule import (
  ModelBuilder,
  OptimizerBuilder,
  ClientMixedinScheduleFn,
  ClientUpdateDeltaFn,
  ClientWeightFn,
  LRScheduleFn,
)

# Define additional type aliases.
ClientDatasetFn = Callable[[Dict[str, tf.data.Dataset]], Any]
ClientSingleDataPassFn = Callable[
    [tff.learning.Model,
     Dict[str, tf.data.Dataset],
     tf.keras.optimizers.Optimizer],
    DataPassOutput]
CreateClientSingleDataPassFn = Callable[..., ClientSingleDataPassFn]


def build_fed_training_process(
    model_fn: ModelBuilder,
    client_update_epochs: int,
    create_client_single_data_pass_fn: CreateClientSingleDataPassFn,
    client_mixedin_schedule_fn: ClientMixedinScheduleFn,
    client_update_delta_fn: ClientUpdateDeltaFn,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: Union[float, LRScheduleFn] = 0.1,
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: Union[float, LRScheduleFn] = 1.0,
    client_dataset_fn: Optional[ClientDatasetFn] = None,
    client_weight_fn: Optional[ClientWeightFn] = None,
    mask_zeros_in_client_updates: bool = False,
) -> tff.templates.IterativeProcess:
  """Builds the TFF computations for federated training.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    client_update_epochs: An inteter that represents the number of local
      epochs to run on the clients.
    create_client_single_data_pass_fn: A function that returns a single data
      pass function used to compute loss and updated model parameters locally.
    client_mixedin_schedule_fn: A function that returns a client mixed in check
      function for given round; the latter determines whether the client has
      mixed-in based on the outputs of the previous two epochs; if mixed-in,the
      following epochs can be used to produce samples from the local posterior.
    client_update_delta_fn: A function that computes an updated weights delta
      based on the previous delta and a new posterior sample.
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
    mask_zeros_in_client_updates: A boolean indicating whether to average deltas
      with zero masking that affects the denominator in the average elementwise.

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
    client_mixedin_fn = client_mixedin_schedule_fn(round_num)
    client_single_data_pass_fn = create_client_single_data_pass_fn()
    tf_dataset = client_dataset_fn(tf_dataset)
    return client_update(
        model=model_fn(),
        dataset=tf_dataset,
        num_epochs=client_update_epochs,
        initial_weights=initial_model_weights,
        client_optimizer=client_optimizer,
        client_mixedin_fn=client_mixedin_fn,
        client_update_delta_fn=client_update_delta_fn,
        client_single_data_pass_fn=client_single_data_pass_fn,
        client_weight_fn=client_weight_fn)

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta)

  client_output_type = client_update_fn.type_signature.result
  client_model_delta_type = client_output_type.weights_delta
  client_weight_type = client_output_type.client_weight

  federated_mean_masked = build_federated_mean_masked(client_model_delta_type,
                                                      client_weight_type)

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

    # Aggregate model deltas.
    client_weight = client_outputs.client_weight
    if mask_zeros_in_client_updates:
      model_delta = federated_mean_masked(client_outputs.weights_delta,
                                          client_weight)
    else:
      model_delta = tff.federated_mean(client_outputs.weights_delta,
                                       client_weight)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, model_delta))

    # Aggregate model outputs that contain local metrics and various statistics.
    aggregated_outputs = dummy_model.federated_output_computation(
      client_outputs.model_output)
    additional_outputs = tff.federated_mean(
      client_outputs.additional_output, weight=client_weight)

    @tff.tf_computation(aggregated_outputs.type_signature.member,
                        additional_outputs.type_signature.member)
    def _update_aggregated_outputs(aggregated_outputs, additional_outputs):
      aggregated_outputs.update(additional_outputs)
      return aggregated_outputs

    aggregated_outputs = tff.federated_map(
      _update_aggregated_outputs, (aggregated_outputs, additional_outputs))
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
