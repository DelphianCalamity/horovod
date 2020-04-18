# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
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
# ==============================================================================
# pylint: disable=g-short-docstring-punctuation
# horovod version: v0.18.1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.common.util import check_extension

check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')

from horovod.tensorflow.compression import Compression
from horovod.tensorflow.mpi_ops import allgather, broadcast, _allreduce
from horovod.tensorflow.mpi_ops import init, shutdown
from horovod.tensorflow.mpi_ops import size, local_size, rank, local_rank
from horovod.tensorflow.mpi_ops import mpi_threads_supported, mpi_enabled, mpi_built
from horovod.tensorflow.mpi_ops import gloo_enabled, gloo_built
from horovod.tensorflow.mpi_ops import nccl_built, ddl_built, mlsl_built
from horovod.tensorflow.util import _executing_eagerly, _make_subgraph, _cache

from distutils.util import strtobool

import tensorflow as tf
import math
import json
import os

def allreduce(tensor, average=True, device_dense='', device_sparse='',
                   compression=Compression.none,
                   params=None,
                   ):
    """Perform an allreduce on a tf.Tensor or tf.IndexedSlices.

    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
        The shape of the input must be identical across all ranks.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.
        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was build with HOROVOD_GPU_ALLREDUCE.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was build with HOROVOD_GPU_ALLGATHER.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.

    This function performs a bandwidth-optimal ring allreduce on the input
    tensor. If the input is an tf.IndexedSlices, the function instead does an
    allgather on the values and the indices, effectively doing an allreduce on
    the represented tensor.
    """

    comp_dict = {}
    comp_dict["none"] = Compression.none
    comp_dict["fp16"] = Compression.fp16
    comp_dict["randomk"] = Compression.randomk
    comp_dict["topk"] = Compression.topk
    comp_dict["threshold"] = Compression.threshold
    comp_dict["terngrad"] = Compression.terngrad
    comp_dict["qsgd"] = Compression.qsgd
    comp_dict["dgc"] = Compression.dgc
    comp_dict["adaq"] = Compression.adaq
    comp_dict["signsgd"] = Compression.signsgd
    comp_dict["efsignsgd"] = Compression.efsignsgd
    comp_dict["signum"] = Compression.signum
    comp_dict["adas"] = Compression.adas
    comp_dict["onebit"] = Compression.onebit
    comp_dict["powersgd"] = Compression.powersgd
    comp_dict["8bit"] = Compression.u8bit
    comp_dict["natural"] = Compression.natural
    comp_dict["sketch"] = Compression.sketch
    comp_dict["inceptionn"] = Compression.inceptionn
    comp_dict["bloom"] = Compression.bloom
    # testing
    if not params['compress_state']:
        for method in ['randomk', 'topk', 'threshold', 'terngrad', 'qsgd', 'dgc', 'adaq',
                       'signsgd', 'efsignsgd', 'signum', 'adas', 'onebit', 'powersgd', '8bit', 'natural', 'sketch',
                       'bloom']:
            comp_dict[method] = Compression.fake

    default_params = {}
    default_params["compress_method"] = 'none'
    default_params["comm_method"] = 'allgather'
    default_params["use_memory"] = False
    default_params["compress_ratio"] = 0.3
    default_params["threshold_val"] = 0.0001
    default_params["momentum"] = 0.9
    default_params["learning_rate"] = 0.1
    default_params["quantum_num"] = 256
    default_params["gradient_clipping"] = False
    default_params["horovod_size"] = size()
    default_params["compressor"] = comp_dict[default_params["compress_method"]]
    default_params['tensors_size_are_same'] = False
    default_params['debug'] = False
    # default_params['compression_device'] = device_dense
    default_params['average'] = average
    default_params['beta'] = 1.0
    default_params['gamma'] = 1.0
    default_params['compress_rank'] = 2
    default_params['error_bound'] = 2e-10
    default_params['mem_mode'] = 0
    default_params['suffix'] = 0

    if params is None:
        params={}
        params["compress_method"] = 'none'
        params["comm_method"] = 'allreduce'
    elif ("compress_method" in params) and ("comm_method" not in params):
        params["comm_method"] = 'allreduce' if params["compress_method"] == 'none' else 'allgather'

    for argument in default_params:
        if argument not in params:
            params[argument] = default_params[argument]

    params["compressor"] = comp_dict[params["compress_method"]]
    comm_method = params["comm_method"]
    horovod_size = tf.cast(params["horovod_size"], dtype=tensor.dtype)
    compression = params["compressor"]
    params['tensor_name'] = tensor.name


    # if params['compression_device'] =='':
    #     params['compression_device'] = device_dense

    # print("========================== print params ====================================")
    # print(params)
    if isinstance(tensor, tf.IndexedSlices):
        print("=====this model contains sparse gradient")
        with tf.device(device_sparse):
            # For IndexedSlices, do two allgathers intead of an allreduce.
            horovod_size = tf.cast(size(), tensor.values.dtype)
            values = allgather(tensor.values)
            indices = allgather(tensor.indices)

            # To make this operation into an average, divide all gathered values by
            # the Horovod size.
            new_values = (values / horovod_size) if average else values
        return tf.IndexedSlices(new_values, indices,
                                dense_shape=tensor.dense_shape)
    else:
        with tf.device(device_dense):
            print("=====this model contains dense gradient")
            params['tensor_dims'] = len(tensor.get_shape().as_list())
            def Allreduce(tensors):
                if tensors is None:
                    tensors = []
                elif type(tensors) not in [list, tuple]:
                    tensors = [tensors]
                summed_tensor_compressed = []

                for i in range(len(tensors)):
                    # if params['debug']:
                    #     tensors[i] = tf.Print(tensors[i], [tf.size(tensors[i])],
                    #                           message="==Debug== tensor %d/%d on rank %d %s size:"
                    #                                   % (i, len(tensors), rank(), tensors[i].dtype))
                    summed_tensor_compressed.append(_allreduce(tensors[i]))
                return summed_tensor_compressed

            def Allgather(tensors):
                from collections import defaultdict
                if tensors is None:
                    tensors = []
                elif type(tensors) not in [list, tuple]:
                    tensors = [tensors]
                tensors_size = []
                tensors_shape = {}
                tensors_1d = {}
                tensors_ag = {}
                new_tensors = defaultdict(list)
                num = len(tensors)
                for i in range(len(tensors)):
                    # tensors_size.append(tf.reshape(tf.size(tensors[i]), [-1]))
                    # tensors_shape[i] = tf.shape(tensors[i])
                    # tensors_1d[i] = tf.reshape(tensors[i], [-1])
                    tensors_size.append(tf.reshape(tf.shape(tensors[i])[0], [-1]))
                    # tensors_shape[i] = tf.shape(tensors[i])
                    tensors_1d[i] = tensors[i]
                    #print tensor size
                    # if params['debug']:
                    #     tensors_1d[i] = tf.Print(tensors_1d[i], [tf.size(tensors_1d[i])],
                    #                              message="==Debug== tensor %d/%d on rank %d %s size:"
                    #                                      % (i, len(tensors), rank(), tensors_1d[i].dtype))
                    tensors_ag[i] = allgather(tensors_1d[i])
                tensors_size = tf.concat(tensors_size, 0)

                tensors_size_list = []
                for ranki in range(size()):
                    tensors_size_list.append(tensors_size)

                if params['tensors_size_are_same']:
                    tensors_size_ag = tf.concat(tensors_size_list, 0)
                else:
                    tensors_size_ag = allgather(tensors_size)

                index_a = defaultdict(int)
                index_b = {}
                for ranki in range(size()):
                    tensors_size = tensors_size_ag[num * ranki:num * (ranki+1)]
                    for i in range(len(tensors)):
                        index_b[i] = index_a[i] + tensors_size[i]
                        # new_tensors[ranki].append(tf.reshape(tensors_ag[i][index_a[i]:index_b[i]], tensors_shape[i]))
                        new_tensors[ranki].append(tensors_ag[i][index_a[i]:index_b[i]])
                        index_a[i] = index_b[i]
                return new_tensors

            def Broadcast(tensors):
                from collections import defaultdict
                if tensors is None:
                    tensors = []
                elif type(tensors) not in [list, tuple]:
                    tensors = [tensors]
                new_tensors = defaultdict(list)
                for ranki in range(size()):
                    for i in range(len(tensors)):
                        # if params['debug']:
                        #     tensors[i] = tf.Print(tensors[i], [tf.size(tensors[i])],
                        #                           message="==Debug== tensor %d/%d on rank %d %s size:"
                        #                                   % (i, len(tensors), rank(), tensors[i].dtype))
                        new_tensors[ranki].append(broadcast(tensors[i], root_rank=ranki, name=None))
                return new_tensors

            communicate = {}
            communicate['allreduce'] = Allreduce
            communicate['allgather'] = Allgather
            communicate['broadcast'] = Broadcast
            tensor_compensate = compression.memory_compensate(tensor, params)
            # if params['memory_debug']:
            #     tensor_compensate = tf.cond(tf.train.get_global_step() < 3,
            #                                 lambda: tf.Print(tensor_compensate, [tf.train.get_global_step(),
            #                                         tf.reduce_sum(tensor_compensate - tensor)],
            #                                         message="==Debug== tensor_compensate - tensor:")
            #                                 , lambda: tensor_compensate)
            tensor_compressed, ctx = compression.compress(tensor_compensate, params)

            memory_update_op = compression.memory_update(tensor, tensor_compensate, tensor_compressed, ctx, params)

            if comm_method == 'allreduce':
                summed_tensor_compressed = Allreduce(tensor_compressed)
                if len(summed_tensor_compressed) == 1:
                    summed_tensor_compressed = summed_tensor_compressed[0]
                summed_tensor = compression.decompress(summed_tensor_compressed, ctx, params)
                new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif comm_method in ['broadcast', 'allgather']:
                list_tensor_compressed = communicate[comm_method](tensor_compressed)
                list_tensor_decompressed = []
                for ranki in range(size()):
                    if len(list_tensor_compressed[ranki]) == 1:
                        temp = list_tensor_compressed[ranki][0]
                    else:
                        temp = list_tensor_compressed[ranki]
                    list_tensor_decompressed.append(
                        compression.decompress(temp, ctx, params))

                new_tensor = compression.aggregate(list_tensor_decompressed, params)

            for op in memory_update_op:
                new_tensor = new_tensor + op - op
        return new_tensor


@_cache
def _make_broadcast_group_fn():
    if _executing_eagerly():
        # Eager mode will parallelize independent control flow
        def broadcast_group(variables, root_rank):
            for var in variables:
                var.assign(broadcast(var, root_rank))

        return _make_subgraph(broadcast_group)
    else:
        # Graph mode requires an Op
        def broadcast_group(variables, root_rank):
            return tf.group(*[var.assign(broadcast(var, root_rank))
                              for var in variables])

        return broadcast_group


def broadcast_variables(variables, root_rank):
    """Broadcasts variables from root rank to all other processes.
    Arguments:
        variables: variables for broadcast
        root_rank: rank of the process from which global variables will be broadcasted
                   to all other processes.
    """
    broadcast_group = _make_broadcast_group_fn()
    return broadcast_group(variables, root_rank)


try:
    _global_variables = tf.global_variables
except AttributeError:
    try:
        _global_variables = tf.compat.v1.global_variables
    except AttributeError:
        _global_variables = None

if _global_variables is not None:
    def broadcast_global_variables(root_rank):
        """Broadcasts all global variables from root rank to all other processes.
        **NOTE:** deprecated in TensorFlow 2.0.
        Arguments:
            root_rank: rank of the process from which global variables will be broadcasted
                       to all other processes.
        """
        if _executing_eagerly():
            raise RuntimeError(
                "hvd.broadcast_global_variables() does not support eager execution. "
                "Please use `hvd.broadcast_variables(<model/optimizer variables>)` instead."
            )

        return broadcast_variables(_global_variables(), root_rank)

try:
    _get_default_graph = tf.get_default_graph
except AttributeError:
    try:
        _get_default_graph = tf.compat.v1.get_default_graph
    except AttributeError:
        _get_default_graph = None

try:
    _SessionRunHook = tf.estimator.SessionRunHook
except AttributeError:
    try:
        _SessionRunHook = tf.train.SessionRunHook
    except AttributeError:
        _SessionRunHook = None

if _SessionRunHook is not None and _get_default_graph is not None:
    class BroadcastGlobalVariablesHook(_SessionRunHook):
        """
        SessionRunHook that will broadcast all global variables from root rank
        to all other processes during initialization.
        This is necessary to ensure consistent initialization of all workers when
        training is started with random weights or restored from a checkpoint.
        **NOTE:** deprecated in TensorFlow 2.0.
        """

        def __init__(self, root_rank, device=''):
            """Construct a new BroadcastGlobalVariablesHook that will broadcast all
            global variables from root rank to all other processes during initialization.
            Args:
              root_rank:
                Rank that will send data, other ranks will receive data.
              device:
                Device to be used for broadcasting. Uses GPU by default
                if Horovod was built with HOROVOD_GPU_BROADCAST.
            """
            super(BroadcastGlobalVariablesHook, self).__init__()
            self.root_rank = root_rank
            self.bcast_op = None
            self.device = device

        def begin(self):
            if not self.bcast_op or self.bcast_op.graph != _get_default_graph():
                with tf.device(self.device):
                    self.bcast_op = broadcast_global_variables(self.root_rank)

        def after_create_session(self, session, coord):
            session.run(self.bcast_op)


@_cache
def _make_allreduce_grads_fn(name, device_dense, device_sparse,
                             compression, sparse_as_dense, params):
    if type(params)==str:
        params = json.loads(params)
    def allreduce_grads(grads):
        params_size = 0
        params_size_ls = []
        for grad in grads:
            params_size += tf.size(grad)
            params_size_ls.append(tf.zeros([tf.size(grad)]).get_shape().as_list())

        print('The model has ', len(grads), ' gradients')
        print("The model has ", tf.zeros([params_size]).get_shape().as_list(), ' parameters')
        print("=======Parameters size print BEGIN========")
        print(params_size_ls)
        print("=======Parameters size print END========")

        with tf.name_scope(name + "_Allreduce"):
            if sparse_as_dense:
                grads = [tf.convert_to_tensor(grad)
                         if grad is not None and isinstance(grad, tf.IndexedSlices)
                         else grad for grad in grads]

            all_reduce_list = []
            for i, grad in enumerate(grads):

                params['logfile_suffix'] = i

                if grad is not None:
                    all_reduce_list.append(allreduce(grad,
                              device_dense=device_dense,
                              device_sparse=device_sparse,
                              compression=compression,
                              params=params))
                else:
                    all_reduce_list.append(grad)

            return all_reduce_list

    if _executing_eagerly():
        return _make_subgraph(allreduce_grads)
    else:
        return allreduce_grads


try:
    # TensorFlow 2.x
    _LegacyOptimizer = tf.compat.v1.train.Optimizer
except AttributeError:
    try:
        # TensorFlow 1.x
        _LegacyOptimizer = tf.train.Optimizer
    except AttributeError:
        # Future TensorFlow versions
        _LegacyOptimizer = None

if _LegacyOptimizer is not None:
    class _DistributedOptimizer(_LegacyOptimizer):
        """An optimizer that wraps another tf.Optimizer, using an allreduce to
        average gradient values before applying gradients to model weights."""

        def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                    device_sparse='', compression=Compression.none,
                    sparse_as_dense=False, params=None):
            if name is None:
                name = "Distributed{}".format(type(optimizer).__name__)
            super(_DistributedOptimizer, self).__init__(name=name, use_locking=use_locking)

            self._optimizer = optimizer
            self._allreduce_grads = _make_allreduce_grads_fn(
                name, device_dense, device_sparse, compression, sparse_as_dense, params)

        def compute_gradients(self, *args, **kwargs):
            """Compute gradients of all trainable variables.
            See Optimizer.compute_gradients() for more info.
            In DistributedOptimizer, compute_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = self._optimizer.compute_gradients(*args, **kwargs)
            # if os.environ.get('HOROVOD_DEBUG', False):
            # print(f"==Debug== The model has {len(gradients)} gradient tensors")
            if size() > 1:
                grads, vars = zip(*gradients)
                avg_grads = self._allreduce_grads(grads)
                return list(zip(avg_grads, vars))
            else:
                return gradients

        def apply_gradients(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.apply_gradients(*args, **kwargs)

        def get_slot(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot(*args, **kwargs)

        def get_slot_names(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot_names(*args, **kwargs)

        def variables(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.variables(*args, **kwargs)


def DistributedOptimizer(optimizer, name=None, use_locking=False, device_dense='',
                         device_sparse='', compression=Compression.none,
                         sparse_as_dense=False, params=None):
    """Construct a new DistributedOptimizer, which uses another optimizer
    under the hood for computing single-process gradient values and
    applying gradient updates after the gradient values have been averaged
    across all the Horovod ranks.
    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "Distributed" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.
      device_dense:
        Device to be used for dense tensors. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_ALLREDUCE.
      device_sparse:
        Device to be used for sparse tensors. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_ALLGATHER.
      compression:
        Compression algorithm used during allreduce to reduce the amount
        of data sent during each parameter update step.  Defaults to
        not using compression.
      sparse_as_dense:
        Treat all sparse gradients as dense tensors.  This can help improve
        performance and memory utilization if the original sparse gradient
        has high density.  Defaults to false.
    """
    if type(params) == dict:
        params = json.dumps(params)
    if isinstance(optimizer, _LegacyOptimizer):
        return _DistributedOptimizer(optimizer, name, use_locking, device_dense,
                                     device_sparse, compression, sparse_as_dense, params)
    elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
        import horovod.tensorflow.keras as hvd_k
        return hvd_k.DistributedOptimizer(optimizer, name, device_dense, device_sparse,
                                          compression, sparse_as_dense, params)
    else:
        raise ValueError('Provided optimizer doesn\'t inherit from either legacy '
                         'TensorFlow or Keras optimizer: %s' % optimizer)


if hasattr(tf, 'GradientTape'):
    class _DistributedGradientTape(tf.GradientTape):
        def __init__(self, tape, device_dense, device_sparse, compression, sparse_as_dense, params=None,
                     persistent=False, watch_accessed_variables=True,):
            if hasattr(tape, '_watch_accessed_variables'):
                super(self.__class__, self).__init__(persistent, watch_accessed_variables)
            else:
                super(self.__class__, self).__init__(persistent)

            self._tape = tape
            self._allreduce_grads = _make_allreduce_grads_fn(
                'DistributedGradientTape', device_dense, device_sparse, compression,
                sparse_as_dense, params)

        def gradient(self, target, sources, output_gradients=None):
            gradients = super(self.__class__, self).gradient(target, sources, output_gradients)
            if size() > 1:
                return self._allreduce_grads(gradients)
            else:
                return gradients


    def DistributedGradientTape(gradtape, device_dense='', device_sparse='',
                                compression=Compression.none, sparse_as_dense=False, params=None):
        """A tape that wraps another tf.GradientTape, using an allreduce to
        average gradient values before applying gradients to model weights.
        Args:
          gradtape:
            GradientTape to use for computing gradients and applying updates.
          device_dense:
            Device to be used for dense tensors. Uses GPU by default
            if Horovod was built with HOROVOD_GPU_ALLREDUCE.
          device_sparse:
            Device to be used for sparse tensors. Uses GPU by default
            if Horovod was built with HOROVOD_GPU_ALLGATHER.
          compression:
            Compression algorithm used during allreduce to reduce the amount
            of data sent during each parameter update step.  Defaults to
            not using compression.
          sparse_as_dense:
            Treat all sparse gradients as dense tensors.  This can help improve
            performance and memory utilization if the original sparse gradient
            has high density.  Defaults to false.
        """
        cls = type(gradtape.__class__.__name__, (gradtape.__class__,),
                   dict(_DistributedGradientTape.__dict__))
        if hasattr(gradtape, '_watch_accessed_variables'):
            return cls(gradtape._tape, device_dense, device_sparse, compression,
                       sparse_as_dense, params, gradtape._persistent,
                       gradtape._watch_accessed_variables)
        else:
            return cls(gradtape._tape, device_dense, device_sparse, compression,
                       sparse_as_dense, params, gradtape._persistent)
