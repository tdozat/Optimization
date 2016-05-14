#!/usr/bin/env python
 
# Copyright 2015 Google Inc and 2016 Timothy Dozat. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import math_ops

from base import BaseOptimizer

#***************************************************************
class AdamOptimizer(BaseOptimizer):
  """"""
  
  #=============================================================
  def __init__(self, lr=0.002, mu=.9, ups=.9, eps=1e-16,
               chi=0., clip=0., noise=None, use_locking=False, name='Adam'):
    """
    Implements Adam
    Inputs:
      lr: the learning rate (default is .002)
      mu: the decay constant for the first moment (originally beta1; default is
          .9)
      ups: the decay constant for the uncentered second moment (originall beta2;
           default is .9)
    """
    
    super(AdamOptimizer, self).__init__(lr=lr, eps=eps, chi=chi, clip=clip, noise=noise,
                                        use_locking=use_locking, name=name)
    self._mu = float(mu)
    self._ups = float(ups)
    
  #=============================================================
  @property
  def mu(self):
    """"""
    
    return self._mu
  
  #=============================================================
  @property
  def upsilon(self):
    """"""
    
    return self._ups
  
  #=============================================================
  def _create_slots(self, grads_and_vars):
    """"""
    
    super(AdamOptimizer, self)._create_slots(grads_and_vars)
    for g_t, x_tm1 in grads_and_vars:
      if self._mu > 0:
        self._zeros_slot(x_tm1, "m", self._name)
        if isinstance(g_t, ops.Tensor):
          self._zero_slot(x_tm1, "m/tm1", self._name)
        else:
          self._zeros_idx_slot(x_tm1, "m/tm1", self._name)
      if self._ups > 0:
        self._ones_slot(x_tm1, "v", self._name)
        if isinstance(g_t, ops.Tensor):
          self._zero_slot(x_tm1, "v/tm1", self._name)
        else:
          self._zeros_idx_slot(x_tm1, "v/tm1", self._name)
  
  #=============================================================
  def _apply_dense(self, g_t, x_tm1, prepare):
    """"""
    
    updates = []
    
    if self._mu > 0:
      m_and_t = self._dense_moving_average(x_tm1, g_t, 'm', self._mu)
      m_bar_t = m_and_t[0]
      updates.extend(m_and_t)
    else:
      m_bar_t = g_t
    
    if self._ups > 0:
      v_and_t = self._dense_moving_average(x_tm1, g_t**2, 'v', self._ups)
      eps_t = ops.convert_to_tensor(self._eps)
      v_bar_t = math_ops.sqrt(v_and_t[0] + eps_t)
      updates.extend(v_and_t)
    else:
      v_bar_t = 1.
    
    s_t = self._lr * m_bar_t / v_bar_t
    return [[s_t, x_tm1, g_t]] + updates
  
  #=============================================================
  def _apply_sparse(self, g_t, x_tm1, prepare):
    """"""
    
    idxs, idxs_ = array_ops.unique(g_t.indices)
    g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
    updates = []
    
    if self._mu > 0:
      m_and_t = self._sparse_moving_average(x_tm1, idxs, g_t_, 'm', self._mu)
      m_t_ = array_ops.gather(m_and_t[0], idxs)
      m_bar_t_ = m_t_
      updates.extend(m_and_t)
    else:
      m_bar_t_ = g_t_
    
    if self._ups > 0:
      v_and_t = self._sparse_moving_average(x_tm1, idxs, g_t_**2, 'v', self._ups)
      v_t_ = array_ops.gather(v_and_t[0], idxs)
      eps_t = ops.convert_to_tensor(self._eps)
      v_bar_t_ = math_ops.sqrt(v_t_ + eps_t)
      updates.extend(v_and_t)
    else:
      v_bar_t_ = 1.
    
    s_t_ = self._lr * m_bar_t_ / v_bar_t_
    return [[s_t_, x_tm1, idxs, g_t]] + updates
  
  #==============================================================
  def clear_slots(self, var_list=None):
    """"""
    
    updates = []
    if var_list is None:
      var_list = variables.trainable_variables()
    for var in var_list:
      if self._mu > 0:
        m = self.get_slot(var, 'm')
        updates.append(state_ops.assign(m, m*0, use_locking=self._use_locking))
        tm1_m = self.get_slot(var, 'm')
        updates.append(state_ops.assign(tm1_m, tm1_m*0, use_locking=self._use_locking))
      if self._ups > 0:
        v = self.get_slot(var, 'v')
        updates.append(state_ops.assign(v, v*0, use_locking=self._use_locking))
        tm1_v = self.get_slot(var, 'v/tm1')
        updates.append(state_ops.assign(tm1_v, tm1_v*0, use_locking=self._use_locking))
    return control_flow_ops.group(*updates)
 