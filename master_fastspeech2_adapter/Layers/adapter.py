# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
import six
import collections
import copy
import json
import math
import re







def feedforward_adapter(input_tensor, hidden_size=64, init_scale=1e-3):
  """A feedforward adapter layer with a bottleneck.

  Implements a bottleneck layer with a user-specified nonlinearity and an
  identity residual connection. All variables created are added to the
  "adapters" collection.

  Args:
    input_tensor: input Tensor of shape [batch size, hidden dimension]
    hidden_size: dimension of the bottleneck layer.
    init_scale: Scale of the initialization distribution used for weights.

  Returns:
    Tensor of the same shape as x.
  """
  with tf.variable_scope("adapters"):
    in_size = input_tensor.get_shape().as_list()[1]
    w1 = tf.get_variable(
        "weights1", [in_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=init_scale),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    b1 = tf.get_variable(
        "biases1", [1, hidden_size],
        initializer=tf.zeros_initializer(),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    net = tf.tensordot(input_tensor, w1, [[1], [0]]) + b1

    net = gelu(net)

    w2 = tf.get_variable(
        "weights2", [hidden_size, in_size],
        initializer=tf.truncated_normal_initializer(stddev=init_scale),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    b2 = tf.get_variable(
        "biases2", [1, in_size],
        initializer=tf.zeros_initializer(),
        collections=["adapters", tf.GraphKeys.GLOBAL_VARIABLES])
    net = tf.tensordot(net, w2, [[1], [0]]) + b2

  return net + input_tensor

def get_adapter(function_string):
  """Maps a string to a Python function.

  Args:
    function_string: String name of the adapter function.

  Returns:
    A Python function corresponding to the adatper function.
    `function_string` is None or empty, will return None.
    If `function_string` is not a string, it will return `function_string`.

  Raises:
    ValueError: The `function_string` does not correspond to a known
      adapter.
  """

  # We assume that anything that"s not a string is already an adapter
  # function, so we just return it.
  if not isinstance(function_string, six.string_types):
    return function_string

  if not function_string:
    return None

  fn = function_string.lower()
  if fn == "feedforward_adapter":
    return feedforward_adapter
  else:
    raise ValueError("Unsupported adapters: %s" % fn)