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

from optimizer import Optimizer

#***************************************************************
class BaseOptimizer(Optimizer):
  """
  The base optimizer that everything else here builds off of
  
  This class supports update clipping, update noising, and temporal averaging
  If you set the learning rate to None, it uses the Oren-Luerenburg scalar
  Hessian approximation as the learning rate (only seems to work for SGD, not 
  more complicated algorithms like Adam)
  """
  
  #=============================================================
  def __init__(self, lr=1., eps=1e-16, chi=0., clip=0., noise=None, save_step=False, save_grad=False, use_locking=False, name='Base'):
    """
    Inputs:
      lr: the global learning rate (default is 1; set to None for nifty second-
          order stuff
      eps: the stability constant sometimes needed (default is 1e-16)
      chi: the decay constant for temporal averaging (default is 0 for no
           averaging)
      clip: the maximum global norm for the updates (default is 0 for no
            clipping)
      noise: how much noise to add to the updates (default is None for no
             noise)
      save_step: whether to save the steps to a slot (default is False)
      save_grad: whether to save the grads to a slot (default is False) 
      use_locking: whether to use locking (default is False)
      name: name for the operator (default is 'Base')
    """
    
    super(BaseOptimizer, self).__init__(use_locking, name)
    self._lr = lr
    self._save_step = save_step
    self._save_grad = save_grad
    if lr is None:
      self._save_step = True
      self._save_grad = True
    self._eps = float(eps)
    self._chi = float(chi)
    self._clip = float(clip)
    self._noise = noise
  
  #=============================================================
  @property
  def learning_rate(self):
    """"""
    
    return self._lr
  
  #=============================================================
  @property
  def epsilon(self):
    """"""
    
    return self._eps
  
  #=============================================================
  @property
  def chi(self):
    """"""
    
    return self._chi
  
  #=============================================================
  @property
  def clip(self):
    """"""
    
    return self._clip
  
  #=============================================================
  @property
  def noise(self):
    """"""
    
    return self._noise
  
  #=============================================================
  def _create_slots(self, grads_and_vars):
    """"""
    
    for g_t, x_tm1 in grads_and_vars:
      if self._save_step:
        self._ones_slot(x_tm1, 's', self._name)
      if self._save_grad:
        self._ones_slot(x_tm1, 'g', self._name)
      if self._chi > 0:
        ops.add_to_collection(self._zeros_slot(x_tm1, 'x', self._name),
                              ops.GraphKeys.MOVING_AVERAGE_VARIABLES)
        if isinstance(g_t, ops.Tensor):
          self._zero_slot(x_tm1, 'x/tm1', self._name)
        else:
          self._zeros_idx_slot(x_tm1, 'x/tm1', self._name)
  
  #=============================================================
  def _prepare(self, grads_and_vars):
    """"""
    
    if self._lr is None:
      sTy = 0
      sTs = 0
      yTy = 0
      for g_t, x_tm1 in grads_and_vars:
        if g_t is None:
          continue
        with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
          if isinstance(g_t, ops.Tensor):
            g_tm1 = self.get_slot(x_tm1, 'g')
            s_tm1 = self.get_slot(x_tm1, 's')
            y_t = (g_t-g_tm1)
            sTy += math_ops.reduce_sum(s_tm1*y_t)
            sTs += math_ops.reduce_sum(s_tm1**2)
            yTy += math_ops.reduce_sum(y_t**2)
          else:
            idxs, idxs_ = array_ops.unique(g_t.indices)
            g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
            g_tm1 = self.get_slot(x_tm1, 'g')
            g_tm1_ = array_ops.gather(g_tm1, idxs)
            s_tm1 = self.get_slot(x_tm1, 's')
            s_tm1_ = array_ops.gather(s_tm1, idxs)
            y_t_ = (g_t_-g_tm1_)
            sTy += math_ops.reduce_sum(s_tm1_*y_t_)
            sTs += math_ops.reduce_sum(s_tm1_**2)
            yTy += math_ops.reduce_sum(y_t_**2)
      sTy = math_ops.abs(sTy)
      self._lr = sTs / (sTy + self._eps)
    
  #=============================================================
  def _apply_dense(self, g_t, x_tm1, prepare):
    """"""
    
    s_t = self._lr * g_t
    return [[s_t, x_tm1, g_t]]
  
  #=============================================================
  def _apply_sparse(self, g_t, x_tm1, prepare):
    """"""
    
    idxs, idxs_ = array_ops.unique(g_t.indices)
    g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
    
    s_t_ = self._lr * g_t_
    return [[s_t_, x_tm1, idxs, g_t_]]
    
  #=============================================================
  def _finish(self, update_ops, name_scope):
    """"""
    
    caches = [update_op[0] for update_op in update_ops]
    update_ops = [update_op[1:] for update_op in update_ops]
    if self._noise is not None:
      for cache in caches:
        s_t, x_tm1 = cache[:2]
        s_t += random_ops.random_normal(x_tm1.initialized_value().get_shape(), stddev=self._noise)
        cache[0] = s_t
    
    if self._clip > 0:
      S_t = [cache[0] for cache in caches]
      S_t, _ = clip_ops.clip_by_global_norm(S_t, self._clip)
      for cache, s_t in zip(caches, S_t):
        cache[0] = s_t
    
    new_update_ops = []
    for cache, update_op in zip(caches, update_ops):
      if len(cache) == 3:
        s_t, x_tm1 = cache[:2]
        with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
          x_t = state_ops.assign_sub(x_tm1, s_t, use_locking=self._use_locking)
          cache.append(x_t)
      else:
        s_t_, x_tm1, idxs = cache[:3]
        with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
          x_t = state_ops.scatter_sub(x_tm1, idxs, s_t_, use_locking=self._use_locking)
          cache.append(x_t)
      new_update_ops.append(control_flow_ops.group(*([x_t] + update_op)))
    
    with ops.control_dependencies(new_update_ops):
      more_update_ops = []
      if self._save_step:
        for cache in caches:
          if len(cache) == 4:
            s_t, x_tm1 = cache[:2]
            s_tm1 = self.get_slot(x_tm1, 's')
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              new_step_and_grads = []
              s_t = state_ops.assign(s_tm1, -s_t, use_locking=self._use_locking)
          else:
            s_t_, x_tm1, idxs = cache[:3]
            s_tm1 = self.get_slot(x_tm1, 's')
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              s_t = state_ops.scatter_update(s_tm1, idxs, -s_t_, use_locking=self._use_locking)
          more_update_ops.append(s_t)
      if self._save_grad:
        for cache in caches:
          if len(cache) == 4:
            x_tm1, g_t = cache[1:3]
            g_tm1 = self.get_slot(x_tm1, 'g')
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              new_step_and_grads = []
              g_t = state_ops.assign(g_tm1, g_t, use_locking=self._use_locking)
          else:
            x_tm1, idxs, g_t_ = cache[1:4]
            g_tm1 = self.get_slot(x_tm1, 'g')
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              g_t = state_ops.scatter_update(g_tm1, idxs, g_t_, use_locking=self._use_locking)
          more_update_ops.append(g_t)
      
      if self._chi > 0:
        for cache in caches:
          if len(cache) == 4:
            _, x_tm1, _, x_t = cache
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              x_and_t = self._dense_moving_average(x_tm1, x_t, 'x', self._chi)
              more_update_ops.append(control_flow_ops.group(*x_and_t))
          else:
            _, x_tm1, idxs, _, x_t = cache
            with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
              x_t_ = array_ops.gather(x_t, idxs)
              x_and_t = self._sparse_moving_average(x_tm1, idxs, x_t_, 'x', self._chi)
              more_update_ops.append(control_flow_ops.group(*x_and_t))
    
    return control_flow_ops.group(*(new_update_ops + more_update_ops), name=name_scope)
  
  #==============================================================
  def average(self, var):
    """"""
    
    return self._slot_dict('x').get(var, None)
  
  #==============================================================
  def average_name(self, var):
    """"""
    
    return var.op.name + '/' + self._name + '/' + 'x'
  
  #==============================================================
  def variables_to_restore(self, moving_avg_variables=None):
    """"""
    
    name_map = {}
    if moving_avg_variables is None:
      moving_avg_variables = variables.trainable_variables()
      moving_avg_variables += variables.moving_average_variables()
    # Remove duplicates
    moving_avg_variables = set(moving_avg_variables)
    # Collect all the variables with moving average,
    for v in moving_avg_variables:
      name_map[self.average_name(v)] = v
    # Make sure we restore variables without moving average as well.
    for v in list(set(variables.all_variables()) - moving_avg_variables):
      if v.op.name not in name_map:
        name_map[v.op.name] = v
    return name_map
