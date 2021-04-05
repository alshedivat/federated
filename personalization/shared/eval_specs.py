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
"""Configuration classes for creating and running federated p13n tasks."""

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
    finetune_batch_size: An integer representing the fine-tuning batch size.
    finetune_epochs: An integer representing the number of fine-tuning epochs.
    finetune_optimizer_rn: A callable that returns fine-tuning optimizer.
    eval_random_seed: An optional int used to seed which clients are
      sampled each round. If `None`, no seed is used.
  """
  clients_per_eval: int = attr.ib(
    validator=[attr.validators.instance_of(int), _check_positive],
    converter=int)
  finetune_batch_size: int = attr.ib(
    validator=[attr.validators.instance_of(int), _check_positive],
    converter=int)
  finetune_epochs: int = attr.ib(
    validator=[attr.validators.instance_of(int), _check_positive],
    converter=int)
  finetune_optimizer_fn: Callable[..., tf.keras.optimizers.Optimizer] = attr.ib(
    validator=attr.validators.is_callable())
  eval_random_seed: Optional[int] = attr.ib(
    default=None,
    validator=attr.validators.optional(attr.validators.instance_of(int)),
    converter=attr.converters.optional(int))
