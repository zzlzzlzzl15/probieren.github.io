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

import torch
import numpy as np
import six
import collections
import copy
import json
import math
import re


class adapter(torch.nn.Module):
  def __init__(self):

    super(adapter, self).__init__()

  def gelu(self,x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + torch.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x * cdf

  def init_weights(self, m, init_scale=1e-3):
      if type(m) == torch.nn.Linear:
          # torch.nn.init.uniform_(tensor, a=0, b=1) 均匀分布
          torch.nn.init.normal_(m.weight, mean=0, std=init_scale)  # normal_ 初始化


  """
  def feedforward_adapter(input_tensor, hidden_size=64, init_scale=1e-3):
    """"""A feedforward adapter layer with a bottleneck.
  
    Implements a bottleneck layer with a user-specified nonlinearity and an
    identity residual connection. All variables created are added to the
    "adapters" collection.
  
    Args:
      input_tensor: input Tensor of shape [batch size, hidden dimension]
      hidden_size: dimension of the bottleneck layer.
      init_scale: Scale of the initialization distribution used for weights.
  
    Returns:
      Tensor of the same shape as x.
    """"""
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
  """


  def feedforward_adapter(self, input_tensor, hidden_size=64, init_scale=1e-3):
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

    in_size = input_tensor.shape[2]


    adapter_linear1 = torch.nn.Linear(in_features = in_size, out_features = hidden_size, bias = True).to(device=input_tensor.device)
    self.init_weights(adapter_linear1, init_scale)
    output_linear1 = adapter_linear1(input_tensor)

    out_gelu = self.gelu(output_linear1)

    adapter_linear2 = torch.nn.Linear(in_features=hidden_size, out_features=in_size, bias=True).to(device=input_tensor.device)
    self.init_weights(adapter_linear2, init_scale)
    net = adapter_linear2(out_gelu)



    return net + input_tensor

  def get_adapter(self, function_string):
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
      return self.feedforward_adapter
    else:
      raise ValueError("Unsupported adapters: %s" % fn)