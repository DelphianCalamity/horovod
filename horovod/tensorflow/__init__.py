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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from horovod.common.util import check_extension

check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')

from horovod.tensorflow.compression import Compression
from horovod.tensorflow.mpi_ops import allgather, broadcast, _allreduce
from horovod.tensorflow.mpi_ops import init, shutdown
from horovod.tensorflow.mpi_ops import size, local_size, rank, local_rank
from horovod.tensorflow.mpi_ops import mpi_threads_supported
from horovod.tensorflow.util import _executing_eagerly

import tensorflow as tf

def allreduce(tensor, average=True, device_dense='', device_sparse='',
               compress_ratio=None,
               comm_method=None,
               compress_method=None,
               use_memory=None
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

    if isinstance(tensor, tf.IndexedSlices):
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
        with tf.device(device_dense):  #with tf.device('/device:GPU:0'):#
            if compress_method == 'none':
                #new_tensor = hvd.allreduce(tensor, average=average, device_dense=device_dense)
                compression = Compression.none
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                tensor_compressed, ctx = compression.compress(tensor)
                summed_tensor_compressed = _allreduce(tensor_compressed)
                summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
                new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'fp16':
                compression = Compression.fp16
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                tensor_compressed, ctx = compression.compress(tensor)
                summed_tensor_compressed = _allreduce(tensor_compressed)
                summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
                new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'randomk':
                compression = Compression.randomk
                if comm_method == 'allreduce':
                    print("======================allreduce method called==============================")
                    horovod_size = tf.cast(size(), dtype=tensor.dtype)
                    tensor_compressed, ctx = compression.compress(tensor,compress_ratio)
                    summed_tensor_compressed = _allreduce(tensor_compressed)
                    summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor


                elif comm_method == 'broadcast':
                    print("======================broadcast method called==============================")
                    horovod_size = tf.cast(size(), dtype=tensor.dtype)
                    tensor_compressed, ctx = compression.compress(tensor,compress_ratio)
                    bd_tensor_sparsed = {}

                    for ranki in range(hvd.size()):
                        bd_tensor_sparsed[str(ranki)] = broadcast(tensor_compressed, root_rank=ranki, name=None)

                    summed_tensor_compressed = tf.math.add_n(list(bd_tensor_sparsed.values()))
                    summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor


                elif comm_method == 'allgather':
                    print("======================allgather method called==============================")
                    horovod_size = tf.cast(size(), dtype=tensor.dtype)
                    tensor_compressed, ctx = compression.compress(tensor,compress_ratio)
                    shape, indices, elemnum = ctx
                    tensor_compressed_1d = allgather(tensor_compressed)

                    nnz = max(1, int(elemnum * compress_ratio))
                    summed_tensor_compressed = tf.Variable(tf.zeros([nnz], dtype=tf.float32))

                    for i in range(hvd.size()):
                        summed_tensor_compressed = tf.math.add(summed_tensor_compressed, tensor_compressed_1d[i * nnz:(i + 1) * nnz])

                    summed_tensor_compressed = _allreduce(tensor_compressed)
                    summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor


            elif compress_method == 'topk':
                compression = Compression.topk
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                elemnum = tf.size(tensor)
                shape = tf.shape(tensor)
                tensor = tf.reshape(tensor, [-1])
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                    elemnum = elemnum.eval()
                    print('tensor size is ', str(elemnum))
                # num_arr = tensor.eval()
                # compress_ratio = 30

                if comm_method == 'broadcast':
                    print("======================broadcast method called==============================")
                    tensor_sparsed, indices = compression.compress(tensor, elemnum, compress_ratio)
                    bd_indices = {}
                    bd_tensor_sparsed = {}
                    # bd_indices[str(hvd.local_rank())] = broadcast(indices, root_rank=hvd.local_rank(), name=None)
                    # bd_tensor_sparsed[str(hvd.local_rank())] = broadcast(tensor_sparsed, root_rank=hvd.local_rank(), name=None)

                    bd_tensors = {}

                    for ranki in range(hvd.size()):
                        bd_indices[str(ranki)] = broadcast(indices, root_rank=ranki, name=None)
                        bd_tensor_sparsed[str(ranki)] = broadcast(tensor_sparsed, root_rank=ranki, name=None)
                        zero_tensor = tf.Variable(tf.zeros([elemnum], dtype=tf.float32))
                        bd_tensors[str(ranki)] = tf.scatter_update(zero_tensor, bd_indices[str(ranki)],
                                                                   bd_tensor_sparsed[str(ranki)])

                    summed_tensor = tf.math.add_n(list(bd_tensors.values()))
                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)


                elif comm_method == 'allgather':
                    print("======================allgather method called==============================")
                    tensor_sparsed, indices = compression.compress(tensor, elemnum, compress_ratio)

                    tensor_sparsed_1d = allgather(tensor_sparsed)
                    indices_1d = allgather(indices)

                    # print('tensor_sparsed_1d.shape',tensor_sparsed_1d.get_shape())
                    # print('indices_1d.shape', indices_1d.get_shape())

                    nnz = max(1, int(elemnum * compress_ratio))
                    summed_tensor = tf.zeros_like(tensor)

                    for i in range(hvd.size()):
                        # if i is not local_rank:
                        zero_tensor = tf.Variable(tf.zeros_like(tensor))  # tf.Variable(tf.zeros_like(tf.reshape(tensor, [-1])))
                        index = indices_1d[i * nnz:(i + 1) * nnz]
                        summed_tensor = tf.math.add(summed_tensor, tf.scatter_update(zero_tensor, index,tensor_sparsed_1d[i * nnz:(i + 1) * nnz]))

                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

            elif compress_method == 'threshold':
                compression = Compression.threshold
                if comm_method == 'allgather':
                    print("======================thershold_allgather method called==============================")
                    tensor_sparsed, indices = compression.compress(tensor, elemnum, compress_ratio)

                    # indices_size = tf.reshape(tf.size(indices), [-1])
                    # indices_size_1d = allgather(indices_size)

                    tensor_sparsed_1d = allgather(tensor_sparsed)
                    indices_1d = allgather(indices)
                    # indices_size_1d = allgather(index_size)

                    with tf.Session():
                        indices_size_1d = indices_1d.eval()
                        # indices_size_1d = [0 for x in range(hvd.size())]
                        # indices_siee_1d[local_rank] = tf.size(indices).eval()

                    # print('tensor_sparsed_1d.shape',tensor_sparsed_1d.get_shape())
                    # print('indices_1d.shape', indices_1d.get_shape())

                    # nz = max(1,int(elemnum * compress_ratio))
                    summed_tensor = tf.zeros_like(tensor)
                    a = 0
                    for i in indices_size_1d:
                        b = a + i
                        zero_tensor = tf.Variable(tf.zeros_like(tensor))
                        index = indices_1d[a:b]
                        summed_tensor = tf.math.add(summed_tensor,
                                                    tf.scatter_update(zero_tensor, index, tensor_sparsed_1d[a:b]))
                        a = b

                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)


                elif comm_method == 'broadcast':
                    print("======================threshold broadcast method called==============================")
                    tensor_sparsed, indices = compression.compress(tensor, elemnum, compress_ratio)
                    bd_indices = []
                    bd_tensor_sparsed = []
                    bd_tensors = []
                    shape_arr = {}
                    indices_arr = {}
                    tensor_sparsed_arr = {}
                    with tf.Session():
                        indices_size = tf.size(indices).eval()
                    # indices_size = indices.get_shape().as_list()[-1]
                    print("===========debug========\nindices_size:", indices_size)
                    for i in range(hvd.size()):
                        if i == hvd.local_rank():
                            shape_arr[str(i)] = tf.Variable(tf.constant(indices_size), dtype=tf.int32)
                        else:
                            shape_arr[str(i)] = tf.zeros(1, dtype=tf.int32)
                    for i in range(hvd.size()):
                        if i == hvd.local_rank():
                            shape_arr[str(i)] = broadcast(shape_arr[str(i)], i, name=None)
                    for i in range(hvd.size()):
                        with tf.Session as sess:
                            shape_arr[str(i)] = shape_arr[str(i)].eval()
                    for i in range(hvd.size()):
                        if i == hvd.local_rank():
                            indices_arr[str(i)] = indices
                            tensor_sparsed_arr[str(i)] = tensor_sparsed
                        else:
                            indices_arr[str(i)] = tf.Variable(tf.zeros(shape_arr[str(i)]), dtype=indices.dtype)
                            tensor_sparsed_arr[str(i)] = tf.Variable(tf.zeros(shape_arr[str(i)]),
                                                                     dtype=tensor_sparsed.dtype)

                    zero_tensor = tf.Variable(tf.zeros([elemnum], dtype=tf.float32))

                    # bd_indices[str(hvd.local_rank())] = broadcast(indices, hvd.local_rank(), name=None)
                    # bd_tensor_sparsed[str(hvd.local_rank())] = broadcast(tensor_sparsed, hvd.local_rank(), name=None)

                    # bd_tensors[str(hvd.local_rank())] = tf.scatter_update(zero_tensor, bd_indices[str(hvd.local_rank())], bd_tensor_sparsed[str(hvd.local_rank())])

                    # for ranki in range(hvd.size()):
                    # for ranki in range(1):
                    for ranki in range(hvd.size()):
                        bd_indices = broadcast(indices_arr[str(ranki)], root_rank=ranki, name=None)
                        bd_tensor_sparsed = broadcast(tensor_sparsed_arr[str(ranki)], root_rank=ranki, name=None)
                        zero_tensor = tf.Variable(tf.zeros([elemnum], dtype=tf.float32))
                        bd_tensors.append(tf.scatter_update(zero_tensor, bd_indices, bd_tensor_sparsed))

                    summed_tensor = tf.math.add_n(list(bd_tensors.values()))
                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

        return new_tensor


def broadcast_global_variables(root_rank):
    """Broadcasts all global variables from root rank to all other processes.

    Arguments:
        root_rank: rank of the process from which global variables will be broadcasted
        to all other processes.
    """
    return broadcast_variables(tf.global_variables(), root_rank)


def broadcast_variables(variables, root_rank):
    """Broadcasts variables from root rank to all other processes.

    Arguments:
        variables: variables for broadcast
        root_rank: rank of the process from which global variables will be broadcasted
                   to all other processes.
    """
    return tf.group(*[tf.assign(var, broadcast(var, root_rank))
                      for var in variables])


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    """
    SessionRunHook that will broadcast all global variables from root rank
    to all other processes during initialization.

    This is necessary to ensure consistent initialization of all workers when
    training is started with random weights or restored from a checkpoint.
    """

    def __init__(self, root_rank, device=''):
        """Construct a new BroadcastGlobalVariablesHook that will broadcast all
        global variables from root rank to all other processes during initialization.

        Args:
          root_rank:
            Rank that will send data, other ranks will receive data.
          device:
            Device to be used for broadcasting. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_BROADCAST.
        """
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.bcast_op = None
        self.device = device

    def begin(self):
        if not self.bcast_op or self.bcast_op.graph != tf.get_default_graph():
            with tf.device(self.device):
                self.bcast_op = broadcast_global_variables(self.root_rank)

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)


class DistributedOptimizer(tf.train.Optimizer):
    """An optimizer that wraps another tf.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights."""

    def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                 device_sparse='', compression=Compression.none,
                 sparse_as_dense=False):
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
            if Horovod was build with HOROVOD_GPU_ALLREDUCE.
          device_sparse:
            Device to be used for sparse tensors. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_ALLGATHER.
          compression:
            Compression algorithm used during allreduce to reduce the amount
            of data sent during the each parameter update step.  Defaults to
            not using compression.
          sparse_as_dense:
            Treat all sparse gradients as dense tensors.  This can help improve
            performance and memory utilization if the original sparse gradient
            has high density.  Defaults to false.
        """
        if name is None:
            name = "Distributed{}".format(type(optimizer).__name__)

        self._optimizer = optimizer
        self._device_dense = device_dense
        self._device_sparse = device_sparse
        self._compression = compression
        self._sparse_as_dense = sparse_as_dense

        def allreduce_grads(grads):
            with tf.name_scope(self._name + "_Allreduce"):
                if self._sparse_as_dense:
                    grads = [tf.convert_to_tensor(grad)
                             if grad is not None and isinstance(grad, tf.IndexedSlices)
                             else grad for grad in grads]

                return [allreduce(grad,
                                  device_dense=self._device_dense,
                                  device_sparse=self._device_sparse,
                                  compression=self._compression)
                        if grad is not None else grad
                        for grad in grads]

        if _executing_eagerly():
            self._allreduce_grads = tf.contrib.eager.defun(allreduce_grads)
        else:
            self._allreduce_grads = allreduce_grads

        super(DistributedOptimizer, self).__init__(
            name=name, use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.

        See Optimizer.compute_gradients() for more info.

        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
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


if hasattr(tf, 'GradientTape'):
    class _DistributedGradientTape(tf.GradientTape):

        def __init__(self, tape, device_dense, device_sparse,
                     compression, sparse_as_dense, persistent=False, watch_accessed_variables=True):
            if hasattr(tape, '_watch_accessed_variables'):
                super(self.__class__, self).__init__(persistent, watch_accessed_variables)
            else:
                super(self.__class__, self).__init__(persistent)
            self._tape = tape
            self._persistent = persistent
            self._watch_accessed_variables = watch_accessed_variables
            self._name = "Distributed"
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense

            def allreduce_grads(grads):
                with tf.name_scope(self._name + "_Allreduce"):
                    if self._sparse_as_dense:
                        grads = [tf.convert_to_tensor(grad)
                                 if grad is not None and isinstance(grad, tf.IndexedSlices)
                                 else grad for grad in grads]
                    return [allreduce(grad,
                                      device_dense=self._device_dense,
                                      device_sparse=self._device_sparse,
                                      compression=self._compression)
                            if grad is not None else grad
                            for grad in grads]

            self._allreduce_grads = tf.contrib.eager.defun(allreduce_grads)

        def gradient(self, target, sources, output_gradients=None):
            gradients = super(self.__class__, self).gradient(target, sources, output_gradients)
            if size() > 1:
                avg_grads = self._allreduce_grads(gradients)
                return avg_grads
            else:
                return gradients


def DistributedGradientTape(gradtape, device_dense='', device_sparse='',
                            compression=Compression.none, sparse_as_dense=False):
    """An tape that wraps another tf.GradientTape, using an allreduce to
    average gradient values before applying gradients to model weights.

    Args:
      gradtape:
        GradientTape to use for computing gradients and applying updates.
      device_dense:
        Device to be used for dense tensors. Uses GPU by default
        if Horovod was build with HOROVOD_GPU_ALLREDUCE.
      device_sparse:
        Device to be used for sparse tensors. Uses GPU by default
        if Horovod was build with HOROVOD_GPU_ALLGATHER.
      compression:
        Compression algorithm used during allreduce to reduce the amount
        of data sent during the each parameter update step.  Defaults to
        not using compression.
      sparse_as_dense:
        Treat all sparse gradients as dense tensors.  This can help improve
        performance and memory utilization if the original sparse gradient
        has high density.  Defaults to false.
    """
    cls = type(gradtape.__class__.__name__, (gradtape.__class__,),
               dict(_DistributedGradientTape.__dict__))
    if hasattr(gradtape, '_watch_accessed_variables'):
        return cls(gradtape._tape, device_dense, device_sparse,
                   compression, sparse_as_dense,
                   gradtape._persistent, gradtape._watch_accessed_variables)
    else:
        return cls(gradtape._tape, device_dense, device_sparse,
                   compression, sparse_as_dense,
                   gradtape._persistent)
