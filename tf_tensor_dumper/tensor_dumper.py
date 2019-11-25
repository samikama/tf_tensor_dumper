from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class _DumperRunHook(tf.train.SessionRunHook):
  def __init__(self, filename="tensors",global_step="global_step:0"):
    self._filename = filename
    self._fetches = []
    self._tensor_names=[]
    self._global_step=global_step
    self._count=0
    self._fetch_list=[]
    if self._global_step:
      self._fetch_list = [
          self._global_step,
      ]
    self._store_names=[]
  def after_create_session(self, session, coord):
    pass

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=self._fetch_list+[x[0] for x in self._fetches])

  def after_run(self, run_context, run_values):
    ret_vals = run_values.results
    if self._global_step:
      global_step=ret_vals[0]
      ret_vals=ret_vals[1:]
    else:
      global_step=self._count
    kwargs = dict(zip(self._store_names, ret_vals))
    np.savez_compressed(
        '%s.%s.npz' %(self._filename,global_step), **kwargs
    )
    self._count += 1

  def add_tensor(self, tensor, name=None):
    tf.logging.warn("Adding %s with name %s"%(tensor.name,name))
    if not isinstance(tensor,tf.Tensor):
      raise KeyError("tensor must be a tf.Tensor but is %s"%type(tensor))
    store_name = tensor.name if name is None else name
    if tensor.name in self._tensor_names:
      tf.logging.error("tensor %s is already added to the list %s"%(tensor.name,zip(self._tensor_names,self._store_names)))
      raise ValueError("tensor %s is already added to the list"%tensor.name)
    elif  name in self._store_names:
      raise ValueError("name %s is already used"%name)
    else:
      self._tensor_names.append(tensor.name)
      self._fetches.append((tensor,store_name))
      self._store_names.append(store_name)

_TFDumpers={None:_DumperRunHook()}

def add_dumper(name,filename="tensors",global_step=None):
  if name is None:
    raise KeyError("name must be non-none")
  for k,v in _TFDumpers:
    if v._filename == filename:
      raise KeyError("File name %s is already used in %s"%(filename,k if k is not None else "Default"))
  if name not in _TFDumpers:
    _TFDumpers[name]=_DumperRunHook(filename,global_step)

def get_dumper(name=None):
  return _TFDumpers.get(name,None)
