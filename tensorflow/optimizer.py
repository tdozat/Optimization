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

#***************************************************************
class Optimizer(object):
  """ Slightly modified version of the original Optimizer class """
  
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2
  
  #=============================================================
  def __init__(self, use_locking, name):
    """"""
    
    if not name:
      raise ValueError("Must specify the optimizer name")
    self._use_locking = use_locking
    self._name = name
    self._slots = {}
  
  #=============================================================
  def minimize(self, loss, global_step=None, var_list=None, gate_gradients=GATE_OP,
               aggregation_method=None, colocate_gradients_with_ops=False, name=None):
    """"""
    
    grads_and_vars = self.compute_gradients(
      loss, var_list=var_list,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)
    return self.apply_gradients(grads_and_vars, global_step=global_step, name=name)
  
  #=============================================================
  def compute_gradients(self, loss, var_list=None, gate_gradients=GATE_OP,
                        aggregation_method=None, colocate_gradients_with_ops=False):
    """"""
    
    # Error checking
    if gate_gradients not in [Optimizer.GATE_NONE, Optimizer.GATE_OP,
                              Optimizer.GATE_GRAPH]:
      raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, " +
        "Optimizer.GATE_OP, Optimizer.GATE_GRAPH. Not %s" % gate_gradients)
    self._assert_valid_dtypes([loss])
    if var_list is None:
      var_list = variables.trainable_variables()
    for x_tm1 in var_list:
      if not isinstance(x_tm1, variables.Variable):
        raise TypeError("Argument is not a tf.Variable: %s" % x_tm1)
    if not var_list:
      raise ValueError("No variables to optimize")
    
    # The actual stuff
    var_refs = [x_tm1.ref() for x_tm1 in var_list]
    grads = gradients.gradients(loss, var_refs,
                                gate_gradients=(gate_gradients == Optimizer.GATE_OP),
                                aggregation_method=aggregation_method,
                                colocate_gradients_with_ops=colocate_gradients_with_ops)
    if gate_gradients == Optimizer.GATE_GRAPH:
      grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, var_list))
    self._assert_valid_dtypes([x_tm1 for g_t, x_tm1 in grads_and_vars if g_t is not None])
    return grads_and_vars
  
  #=============================================================
  def approximate_hessian(self, grads_and_vars, name=None):
    """
    I haven't tested this yet so I have no idea if it works, but even if it
    does it's probably super slow, and either way nothing else has been modified
    to deal with it.
    """
    
    gv = 0
    var_refs = []
    for g_t, x_tm1 in grads_and_vars:
      var_refs.append(x_tm1.ref())
      if g_t is None:
        continue
      with ops.name_scope('update_' + x_tm1.op.name), ops.device(x_tm1.device):
        if isinstance(g_t, ops.Tensor):
          gv += math_ops.reduce_sum(g_t * random_ops.random_normal(g_t.get_shape()))
        else:
          idxs, idxs_ = array_ops.unique(g_t.indices)
          g_t_ = math_ops.unsorted_segment_sum(g_t.values, idxs_, array_ops.size(idxs))
          gv += math_ops.reduce_sum(g_t_ * random_ops.random_normal(g_t_.get_shape()))
    hesses = gradients.gradients(gv, var_refs,
                                 gate_gradients=(gate_gradients == Optimizer.GATE_OP),
                                 aggregation_method=aggregation_method,
                                 colocate_gradients_with_ops=colocate_gradients_with_ops)
    return zip([g_t for g_t, _ in grads_and_vars], [x_tm1 for _, x_tm1 in grads_and_vars], hesses)
  
  #=============================================================
  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """"""
    
    # Error checking
    grads_and_vars = tuple(grads_and_vars)
    for g_t, x_tm1 in grads_and_vars:
      if not isinstance(g_t, (ops.Tensor, ops.IndexedSlices, type(None))):
        raise TypeError(
            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g_t)
      if not isinstance(x_tm1, variables.Variable):
        raise TypeError(
            "Variable must be a tf.Variable: %s" % x_tm1)
      if g_t is not None:
        self._assert_valid_dtypes([g_t, x_tm1])
    var_list = [x_tm1 for g_t, x_tm1 in grads_and_vars if g_t is not None]
    if not var_list:
      raise ValueError("No gradients provided for any variable: %s" %
                       (grads_and_vars,))
    
    # The actual stuff
    with ops.control_dependencies(None):
      self._create_slots(grads_and_vars)
    update_ops = []
    with ops.op_scope([], name, self._name) as name:
      prepare = self._prepare(grads_and_vars)
      for g_t, x_tm1 in grads_and_vars:
        if g_t is None:
          continue
        with ops.name_scope("update_" + x_tm1.op.name), ops.device(x_tm1.device):
          if isinstance(g_t, ops.Tensor):
            update_ops.append(self._apply_dense(g_t, x_tm1, prepare))
          else:
            update_ops.append(self._apply_sparse(g_t, x_tm1, prepare))
      if global_step is None:
        return self._finish(update_ops, name)
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.device(global_step.device):
            return state_ops.assign_add(global_step, 1, name=name).op
  
  #=============================================================
  def get_slot(self, x_tm1, name):
    """"""
    
    named_slots = self._slots.get(name, None)
    if not named_slots:
      return None
    return named_slots.get(x_tm1, None)

  #=============================================================
  def get_slot_names(self):
    """"""
    
    return sorted(self._slots.keys())

  #=============================================================
  def _assert_valid_dtypes(self, tensors):
    """"""
    valid_dtypes = self._valid_dtypes()
    for t in tensors:
      dtype = t.dtype.base_dtype
      if dtype not in valid_dtypes:
        raise ValueError(
            "Invalid type %r for %s, expected: %s." % (
                dtype, t.name, [v for v in valid_dtypes]))

  #=============================================================
  def _valid_dtypes(self):
    """"""
    
    return set([dtypes.float32])

  #=============================================================
  def _create_slots(self, grads_and_vars):
    """"""
    
    pass

  #=============================================================
  def _prepare(self, grads_and_vars):
    """"""
    pass

  #=============================================================
  def _apply_dense(self, g_t, x_tm1, prepare):
    """"""
    
    raise NotImplementedError()

  #=============================================================
  def _apply_sparse(self, g_t, x_tm1, prepare):
    """"""
    
    raise NotImplementedError()

  #=============================================================
  def _dense_moving_average(self, x_tm1, b_t, name, beta=.9):
    """
    Creates a moving average for a dense variable.
    
    Inputs:
      x_tm1: the associated parameter (e.g. a weight matrix)
      b_t: the value to accumulate (e.g. the gradient)
      name: a string to use to retrieve it later (e.g. 'm')
      beta: the decay factor (defaults to .9)
    Outputs:
      a_t: the average after moving
      t: the internal timestep (used to correct initialization bias)
    """
    
    a_tm1 = self.get_slot(x_tm1, '%s' % name)
    tm1 = self.get_slot(x_tm1, '%s/tm1' % name)
    t = state_ops.assign_add(tm1, 1, use_locking = self._use_locking)
    if beta < 1:
      beta_t = ops.convert_to_tensor(beta, name='%s/decay' % name)
      beta_t = beta_t * (1-beta**tm1) / (1-beta**t)
    else:
      beta_t = tm1 / t
    a_t = state_ops.assign(a_tm1, beta_t*a_tm1, use_locking=self._use_locking)
    a_t = state_ops.assign_add(a_t, (1-beta_t)*b_t, use_locking=self._use_locking)
    return a_t, t
    
  #=============================================================
  def _sparse_moving_average(self, x_tm1, idxs, b_t_, name, beta=.9):
    """
    Creates a moving average for a sparse variable.
    Inputs:
      x_tm1: the associated parameter (e.g. a weight matrix)
      idxs: the tensor representing the indices used
      b_t_: the value to accumulate (e.g. slices of the gradient)
      name: a string to use to retrieve it later (e.g. 'm')
      beta: the decay factor (defaults to .9)
    Outputs:
      a_t: the average after moving (same shape as x_tm1, not b_t_)
      t: the internal timestep (used to correct initialization bias)
    """
    
    a_tm1 = self._zeros_slot(x_tm1, '%s' % name, self._name)
    a_tm1_ = array_ops.gather(a_tm1, idxs)
    tm1 = self._zeros_idx_slot(x_tm1, '%s/tm1' % name, self._name)
    tm1_ = array_ops.gather(tm1, idxs)
    t = state_ops.scatter_add(tm1, idxs, tm1_*0+1, use_locking=self._use_locking)
    t_ = array_ops.gather(t, idxs)
    if beta < 1:
      beta_t = ops.convert_to_tensor(beta, name='%s/decay' % name)
      beta_t_ = beta_t * (1-beta_t**tm1_) / (1-beta_t**t_)
    else:
      beta_t_ = tm1_/t_
    a_t = state_ops.scatter_update(a_tm1, idxs, beta_t_*a_tm1_, use_locking=self._use_locking)
    a_t = state_ops.scatter_add(a_t, idxs, (1-beta_t)*b_t_, use_locking=self._use_locking)
    return a_t, t
    
  #=============================================================
  def _finish(self, update_ops, steps_and_params, name_scope):
    """"""
    
    return control_flow_ops.group(*update_ops, name=name_scope)

  #=============================================================
  def _slot_dict(self, slot_name):
    """"""
    
    named_slots = self._slots.get(slot_name, None)
    if named_slots is None:
      named_slots = {}
      self._slots[slot_name] = named_slots
    return named_slots

  #=============================================================
  def _get_or_make_slot(self, x_tm1, val, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _zeros_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      val = array_ops.zeros_like(x_tm1.initialized_value())
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _ones_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      val = array_ops.ones_like(x_tm1.initialized_value())
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _zeros_idx_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      original_shape = x_tm1.initialized_value().get_shape().as_list()
      shape = [1] * len(original_shape)
      shape[0] = original_shape[0]
      val = array_ops.zeros(shape, dtype=x_tm1.dtype)
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _ones_idx_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      original_shape = x_tm1.initialized_value().get_shape().as_list()
      shape = [1] * len(original_shape)
      shape[0] = original_shape[0]
      val = array_ops.ones(shape, dtype=x_tm1.dtype)
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _zero_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      val = array_ops.zeros([], dtype=x_tm1.dtype)
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]

  #=============================================================
  def _one_slot(self, x_tm1, slot_name, op_name):
    """"""
    
    named_slots = self._slot_dict(slot_name)
    if x_tm1 not in named_slots:
      val = array_ops.ones([], dtype=x_tm1.dtype)
      named_slots[x_tm1] = Optimizer.create_slot(x_tm1, val, op_name+'/'+slot_name)
    return named_slots[x_tm1]
  
  #===============================================================
  @staticmethod
  def _create_slot_var(primary, val, scope):
    """"""
    
    slot = variables.Variable(val, name=scope, trainable=False)
    # pylint: disable=protected-access
    if isinstance(primary, variables.Variable) and primary._save_slice_info:
      # Primary is a partitioned x_tm1iable, so we need to also indicate that
      # the slot is a partitioned x_tm1iable.  Slots have the same partitioning
      # as their primaries.
      real_slot_name = scope[len(primary.op.name + "/"):-1]
      slice_info = primary._save_slice_info
      slot._set_save_slice_info(x_tm1iables.Variable.SaveSliceInfo(
          slice_info.full_name + "/" + real_slot_name,
          slice_info.full_shape[:],
          slice_info.var_offset[:],
          slice_info.var_shape[:]))
    # pylint: enable=protected-access
    return slot
  
  #===============================================================
  @staticmethod
  def create_slot(primary, val, name, colocate_with_primary=True):
    """"""
    
    # Scope the slot name in the namespace of the primary variable.
    with ops.name_scope(primary.op.name + "/" + name) as scope:
      if colocate_with_primary:
        with ops.device(primary.device):
          return Optimizer._create_slot_var(primary, val, scope)
      else:
        return Optimizer._create_slot_var(primary, val, scope)

