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
"""Configuration classes for creating and running federated p13n evaluation."""

from typing import Any, Callable, Dict, List, Optional

import attr
import tensorflow as tf
import tensorflow_federated as tff

ModelFnType = Callable[[], tff.learning.Model]
ValidationFnType = Optional[Callable[[Any, int], Dict[str, float]]]
TestFnType = EvaluationFnType = Optional[Callable[[Any], Dict[str, float]]]


def _check_positive(instance, attribute, value):
  if value <= 0:
    raise ValueError(f'{attribute.name} must be positive. Found {value}.')


@attr.s(eq=False, order=False, frozen=True)
class EvalSpec(object):
  """Contains information for creating a federated p13n eval.

  Attributes:
    clients_per_eval: An integer representing the number of clients
      participating in each eval.
    client_outer_batch_size: An integer representing the outer batch size.
    client_inner_batch_size: An integer representing the inner batch size.
    client_inner_epochs: An integer representing the number of inner epochs.
    client_inner_steps: An integer representing the number of inner steps.
    finetune_optimizer_rn: A callable that returns fine-tuning optimizer.
    eval_strategy_name: A string that represents the eval strategy name.
    client_datasets_random_seed: An optional int used to seed which clients are
      sampled at each eval. If `None`, no seed is used.
  """
  clients_per_eval: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  client_outer_batch_size: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  client_inner_batch_size: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  client_inner_epochs: int = attr.ib(
      validator=attr.validators.instance_of(int),
      converter=int)
  client_inner_steps: int = attr.ib(
      validator=attr.validators.instance_of(int),
      converter=int)
  finetune_optimizer_fn: Callable[..., tf.keras.optimizers.Optimizer] = attr.ib(
      validator=attr.validators.is_callable())
  eval_strategy_name: str = attr.ib(
      validator=attr.validators.instance_of(str),
      converter=str)
  finetune_l2_regularizer: Optional[float] = attr.ib(
    default=0.,
    validator=attr.validators.instance_of(float),
    converter=float)
  client_datasets_random_seed: Optional[int] = attr.ib(
    default=None,
    validator=attr.validators.optional(attr.validators.instance_of(int)),
    converter=attr.converters.optional(int))
