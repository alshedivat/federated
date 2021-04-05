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
"""Library for loading and preprocessing EMNIST training and testing data."""

import contextlib


@contextlib.contextmanager
def result_type_is_sequence_hack(tf_computation):
  # Monkey patch the result type of the dataset computation to avoid TypeError
  # being raised inside `tff.simultation.iterative_process_compositions`.
  # TODO: propose to relax the assumption about the type signature of the
  #       dataset computation being SequenceType in TFF.
  try:
    # Monkey-patch tf_computation's result type.
    tf_computation.type_signature.result.is_sequence = lambda: True
    yield
  finally:
    # Monkey-unpatch tf_computation's result type.
    tf_computation.type_signature.result.is_sequence = lambda: False
