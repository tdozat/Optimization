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

from adam import AdamOptimizer

#***************************************************************
class RadamOptimizer(AdamOptimizer):
  """
  Implements Reweighted Adam
      mu_t <- mu * (1-mu^(t-1)) / (1-mu^t)
      ups_t <- ups * (1-ups^(t-1)) / (1-ups^t)
      m_t <- mu_t*m_tm1 + (1-mu_t)*g_t
      mbar_t <- (1-gamma)*m_t + gamma*g_t (Reweighting)
      v_t <- ups_t*v_tm1 + (1-ups_t)*g_t**2
      vbar_t <- sqrt(v_t) + eps
      s_t <- lr * mbar_t / vbar_t
      x_t <- x_t - s_t
  """
  
  #=============================================================
  def __init__(self, lr=0.002, mu=.9, gamma=.05, ups=.9, eps=1e-7, chi=0., clip=0., noise=None, use_locking=False, name='Radam'):
    """"""
    
    super(RadamOptimizer, self).__init__(lr=lr, mu=mu, ups=ups, eps=eps,
                                         chi=chi, clip=clip, noise=noise,
                                         use_locking=use_locking, name=name)
    self._gamma = float(gamma)
  
  #=============================================================
  @property
  def gamma(self):
    """"""
    
    return self._gamma
  
  #=============================================================
  def _apply_dense(self, g_t, x_tm1, prepare):
    """"""
    
    updates = []
    
    if self._mu > 0:
      m_and_t = self._dense_moving_average(x_tm1, g_t, 'm', self._mu)
      gamma_t = ops.convert_to_tensor(self._gamma)
      m_bar_t = (1-gamma_t)*m_and_t[0] + gamma_t*g_t
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
    
    lr_t = ops.convert_to_tensor(self._lr)
    s_t = lr_t * m_bar_t / v_bar_t
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
      gamma_t = ops.convert_to_tensor(self._gamma)
      m_bar_t_ = (1-gamma_t)*m_t_ + gamma_t*g_t_
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
    
    lr_t = ops.convert_to_tensor(self._lr)
    s_t_ = lr_t * m_bar_t_ / v_bar_t_
    return [[s_t_, x_tm1, idxs, g_t]] + updates
