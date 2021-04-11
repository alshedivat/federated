# Copyright 2019, Google LLC.
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
"""Federated EMNIST character recognition library using TFF."""

import collections
import functools
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from personalization.shared import evaluation
from personalization.shared import eval_specs
from personalization.shared import training_specs
from personalization.shared import utils
from personalization.shared.p13n_strategies import finetuning
from utils.datasets import emnist_dataset
from utils.models import emnist_models

EMNIST_MODELS = ['cnn', '2nn']


def configure_training(task_spec: training_specs.TaskSpec,
                       eval_spec: eval_specs.EvalSpec,
                       model: str = 'cnn',
                       seed: Optional[int] = None) -> training_specs.RunnerSpec:
  """Configures training for the EMNIST character recognition task.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `federated_research.utils.training_loop`.

  Args:
    task_spec: A `TaskSpec` object for creating federated training tasks.
    eval_spec: An `EvalSpec` object for creating federated p13n evaluation.
    model: A string specifying the model used for character recognition. Can be
      one of `cnn` and `2nn`, corresponding to a CNN model and a densely
      connected 2-layer model (respectively).
    seed: An optional int used to seed how to split clients into train and test.
      If `None`, no seed is used.

  Returns:
    A `RunnerSpec` containing attributes used for running the newly created
    federated task.
  """
  emnist_task = 'digit_recognition'

  (client_ids_train, client_ids_test,
   build_train_dataset_from_client_id, build_eval_dataset_from_client_id) = (
      emnist_dataset.get_federated_p13n_datasets(
          train_batch_size=task_spec.client_batch_size,
          train_epochs=task_spec.client_epochs_per_round,
          train_max_batches=task_spec.client_max_batches_per_round,
          eval_batch_size=eval_spec.client_batch_size,
          eval_inner_epochs=eval_spec.finetune_epochs,
          eval_inner_max_batches=eval_spec.finetune_max_batches,
          emnist_task=emnist_task,
          seed=seed))

  input_spec = build_train_dataset_from_client_id.type_signature.result.element

  if model == 'cnn':
    model_builder = functools.partial(
        emnist_models.create_conv_dropout_model, only_digits=False)
  elif model == '2nn':
    model_builder = functools.partial(
        emnist_models.create_two_hidden_layer_model, only_digits=False)
  else:
    raise ValueError(
        'Cannot handle model flag [{!s}], must be one of {!s}.'.format(
            model, EMNIST_MODELS))

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  iterative_process = task_spec.iterative_process_builder(tff_model_fn)
  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, iterative_process)
  training_process.get_model_weights = iterative_process.get_model_weights

  # Create a dictionary of personalization strategies.
  personalize_fn_dict = collections.OrderedDict()
  personalize_fn_dict[eval_spec.eval_strategy_name] = functools.partial(
      finetuning.build_personalize_fn,
      optimizer_fn=eval_spec.finetune_optimizer_fn)

  # Build the `tff.Computation` for evaluating the personalization strategies.
  # Here `p13n_eval` is a `tff.Computation` with the following type signature:
  # <model_weights@SERVER, datasets@CLIENTS> -> personalization_metrics@SERVER.
  evaluate_fn = tff.learning.build_personalization_eval(
      model_fn=tff_model_fn,
      personalize_fn_dict=personalize_fn_dict,
      baseline_evaluate_fn=evaluation.evaluate_fn)
  with utils.result_type_is_sequence_hack(build_eval_dataset_from_client_id):
    evaluate_fn = tff.simulation.compose_dataset_computation_with_computation(
        build_eval_dataset_from_client_id, evaluate_fn)

  # Define client sampling strategy at training time.
  _train_client_ids_fn = tff.simulation.build_uniform_sampling_fn(
      client_ids_train,
      size=task_spec.clients_per_round,
      replace=False,
      random_seed=task_spec.client_datasets_random_seed)
  # We convert the output to a list (instead of an np.ndarray) so that it can
  # be used as input to the iterative process.
  train_client_ids_fn = lambda x: list(_train_client_ids_fn(x))

  # Define client sampling strategy at test time.
  _test_client_ids_fn = tff.simulation.build_uniform_sampling_fn(
      client_ids_test,
      size=eval_spec.clients_per_eval,
      replace=False,
      random_seed=eval_spec.client_datasets_random_seed)
  # We convert the output to a list (instead of an np.ndarray) so that it can
  # be used as input to the iterative process.
  test_client_ids_fn = lambda x: list(_test_client_ids_fn(x))

  def test_fn(state):
    # Evaluate client in batches to avoid OOM issues.
    client_metrics = []
    num_clients_test = len(client_ids_test)
    progbar = tf.keras.utils.Progbar(num_clients_test,
                                     unit_name='clients evaluated')
    for i in range(0, num_clients_test, eval_spec.clients_per_eval):
      client_ids_batch = client_ids_test[i:i + eval_spec.clients_per_eval]
      client_metrics_batch = evaluate_fn(
          iterative_process.get_model_weights(state),
          client_ids_batch)
      client_metrics.append(tf.nest.map_structure(np.sum, client_metrics_batch))
      progbar.update(i)
    progbar.update(num_clients_test)
    return tf.nest.map_structure(lambda *x: np.sum(x) / num_clients_test,
                                 *client_metrics)

  def validation_fn(state, round_num):
    client_metrics = evaluate_fn(
        iterative_process.get_model_weights(state),
        test_client_ids_fn(round_num))
    return tf.nest.map_structure(np.mean, client_metrics)

  return training_specs.RunnerSpec(
      iterative_process=training_process,
      client_datasets_fn=train_client_ids_fn,
      validation_fn=validation_fn,
      test_fn=test_fn)
