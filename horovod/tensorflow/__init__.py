# Modifications copyright (C) 2017 Uber Technologies, Inc.
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
#w limitations under the License.
# ==============================================================================
# pylint: disable=g-short-docstring-punctuation
"""## Communicating Between Processes with MPI

TensorFlow natively provides inter-device communication through send and
receive ops and inter-node communication through Distributed TensorFlow, based
on the same send and receive abstractions. On HPC clusters where Infiniband or
other high-speed node interconnects are available, these can end up being
insufficient for synchronous data-parallel training (without asynchronous
gradient descent). This module implements a variety of MPI ops which can take
advantage of hardware-specific MPI libraries for efficient communication.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.common import check_extension

check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')

from horovod.tensorflow.compression import Compression
from horovod.tensorflow.mpi_ops import allgather, broadcast, _allreduce
from horovod.tensorflow.mpi_ops import init, shutdown
from horovod.tensorflow.mpi_ops import size, local_size, rank, local_rank
from horovod.tensorflow.mpi_ops import mpi_threads_supported

import tensorflow as tf
import math


def allreduce(tensor, average=True, device_dense='', device_sparse='',
                   compression=Compression.none, params = None,
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
    comp_dict["signum"] = Compression.signum
    comp_dict["adas"] = Compression.adas
    comp_dict["onebit"] = Compression.onebit
    comp_dict["powersgd"] = Compression.powersgd
    comp_dict["8bit"] = Compression.u8bit
    comp_dict["natural"] = Compression.natural

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

    for argument in default_params:
        if argument not in params:
            params[argument] = default_params[argument]

    params["compressor"] = comp_dict[params["compress_method"]]
    compress_method = params["compress_method"]
    comm_method = params["comm_method"]
    use_memory = params["use_memory"]
    compress_ratio = params["compress_ratio"]
    threshold_val = params["threshold_val"]
    momentum = params["momentum"]
    learning_rate = params["learning_rate"]
    quantum_num = params["quantum_num"]
    gradient_clipping = params["gradient_clipping"]
    horovod_size = tf.cast(params["horovod_size"], dtype=tensor.dtype)
    compression = params["compressor"]

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
        with tf.device(device_dense):
            tensor_flatten = tf.reshape(tensor, [-1])
            elemnum = tensor_flatten.get_shape().as_list()[0]
            tensor_rank = len(tensor.get_shape().as_list())

            if (compress_method is None) or (compress_method == 'none'):
                compression = Compression.none
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                tensor_compressed, ctx = compression.compress(tensor, params)
                summed_tensor_compressed = _allreduce(tensor_compressed)
                summed_tensor = compression.decompress(summed_tensor_compressed, ctx, params)
                new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'fp16':
                compression = Compression.fp16
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                tensor_compressed, ctx = compression.compress(tensor, params)
                summed_tensor_compressed = _allreduce(tensor_compressed)
                summed_tensor = compression.decompress(summed_tensor_compressed, ctx, params)
                new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'randomk':
                tensor_compressed, ctx = compression.compress(tensor, params)
                if comm_method == 'allreduce':
                    summed_tensor_compressed = _allreduce(tensor_compressed)
                    summed_tensor = compression.decompress(summed_tensor_compressed, ctx, params)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

                elif comm_method == 'broadcast':
                    bd_tensor_compressed = {}
                    for ranki in range(size()):
                        bd_tensor_compressed[ranki] = broadcast(tensor_compressed, root_rank=ranki, name=None)
                    summed_tensor_compressed = tf.math.add_n(list(bd_tensor_compressed.values()))
                    summed_tensor = compression.decompress(summed_tensor_compressed, ctx, params)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

                elif comm_method == 'allgather':
                    tensor_compressed_1d = allgather(tensor_compressed)
                    tensor_len = max(1, int(elemnum * compress_ratio))
                    summed_tensor_compressed = tf.Variable(tf.zeros([tensor_len], dtype=tf.float32))
                    for i in range(size()):
                        summed_tensor_compressed += tensor_compressed_1d[i * tensor_len:(i + 1) * tensor_len]
                    summed_tensor = compression.decompress(summed_tensor_compressed, ctx, params)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'topk':
                tensor_compressed, ctx = compression.compress(tensor, params)
                indices, tensor_shape = ctx
                if comm_method == 'broadcast':
                    bd_indices = {}
                    bd_tensor_compressed = {}
                    bd_tensors = {}
                    for ranki in range(size()):
                        bd_indices[ranki] = broadcast(indices, root_rank=ranki, name=None)
                        bd_tensor_compressed[ranki] = broadcast(tensor_compressed, root_rank=ranki, name=None)
                        tensor_compressed_i = bd_tensor_compressed[ranki]
                        ctx_i = bd_indices[ranki], tensor_shape
                        bd_tensors[ranki] = compression.decompress(tensor_compressed_i, ctx_i, params)

                    summed_tensor = tf.math.add_n(list(bd_tensors.values()))
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

                elif comm_method == 'allgather':
                    tensor_compressed_1d = allgather(tensor_compressed)
                    indices_1d = allgather(indices)
                    tensor_len = max(1, int(elemnum * compress_ratio))
                    summed_tensor = tf.zeros_like(tensor)
                    for i in range(size()):
                        indices_i = indices_1d[i * tensor_len:(i + 1) * tensor_len]
                        ctx_i = indices_i, tensor_shape
                        tensor_compressed_i = tensor_compressed_1d[i * tensor_len:(i + 1) * tensor_len]
                        summed_tensor += compression.decompress(tensor_compressed_i, ctx_i, params)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'threshold':
                tensor_compressed, ctx = compression.compress(tensor, params)
                indices, tensor_shape = ctx
                if comm_method == 'allgather':
                    index_size = tf.reshape(tf.size(indices), [-1])
                    tensor_compressed_1d = allgather(tensor_compressed)
                    indices_1d = allgather(indices)
                    indices_size_1d = allgather(index_size)

                    summed_tensor = tf.zeros_like(tensor)
                    a = 0
                    for i in range(size()):
                        b = a + indices_size_1d[i]
                        indices_i = indices_1d[a:b]
                        tensor_compressed_i = tensor_compressed_1d[a:b]
                        ctx_i = indices_i, tensor_shape
                        summed_tensor += compression.decompress(tensor_compressed_i, ctx_i, params)
                        a = b
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method in ['signsgd', 'signum']:
                tensor_compressed, ctx = compression.compress(tensor, params)

                if comm_method == 'allreduce':
                    tensor_decompress = compression.decompress(tensor_compressed, ctx, params)
                    tensor_decompress = tf.cast(tensor_decompress, dtype=tf.float16)
                    summed_tensor = _allreduce(tensor_decompress)
                    summed_tensor = tf.cast(summed_tensor, dtype=tf.float32)
                    if use_memory:
                        # do average and learning rate adjustment
                        new_tensor = (summed_tensor / horovod_size) if average else summed_tensor
                        new_tensor = new_tensor / learning_rate
                    else:
                        # do majority vote
                        summed_tensor_sign = tf.cast(tf.math.greater_equal(summed_tensor, 0), dtype=tf.float32)
                        new_tensor = summed_tensor_sign * 2.0 - 1.0

                elif comm_method == 'allgather':
                    sign = tensor_compressed
                    mean, tensor_shape = ctx
                    mean = tf.reshape(mean, [-1])
                    mean_1d = allgather(mean) if use_memory else None
                    sign_1d = allgather(sign)
                    tensor_len = elemnum
                    summed_tensor = tf.zeros_like(tensor)
                    for i in range(size()):
                        tensor_compressed_i = sign_1d[i * tensor_len:(i + 1) * tensor_len]
                        mean_i = mean_1d[i] if use_memory else None
                        ctx_i = mean_i, tensor_shape
                        tensor_decompress = compression.decompress(tensor_compressed_i, ctx_i, params)
                        summed_tensor += tensor_decompress
                    if use_memory:
                        # do average and learning rate adjustment
                        new_tensor = (summed_tensor / horovod_size) if average else summed_tensor
                        new_tensor = new_tensor / learning_rate
                    else:
                        # do majority vote
                        summed_tensor_sign = tf.cast(tf.math.greater_equal(summed_tensor, 0), dtype=tf.float32)
                        new_tensor = summed_tensor_sign * 2.0 - 1.0

            elif compress_method == 'qsgd':
                tensor_compressed, ctx = compression.compress(tensor, params)
                norm, tensor_shape = ctx
                if comm_method == 'broadcast':
                    bd_tensor_compressed = {}
                    bd_tensors = {}
                    bd_norm = {}
                    for ranki in range(size()):
                        bd_norm[ranki] = broadcast(norm, root_rank=ranki, name=None)
                        bd_tensor_compressed[ranki] = broadcast(tensor_compressed, root_rank=ranki, name=None)
                        tensor_compressed_i = bd_tensor_compressed[ranki]
                        ctx_i = bd_norm[ranki], tensor_shape
                        bd_tensors[ranki] = compression.decompress(tensor_compressed_i, ctx_i, params)
                    summed_tensor = tf.math.add_n(list(bd_tensors.values()))
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

                elif comm_method == 'allgather':
                    tensor_compressed_1d = allgather(tensor_compressed)
                    norm_1d = allgather(norm)
                    bits = int(math.log(quantum_num, 2) + 1)
                    tensor_len = bits + 1
                    summed_tensor = tf.zeros_like(tensor)
                    for i in range(size()):
                        norm_i = norm_1d[i]
                        tensor_compressed_i = tensor_compressed_1d[i * tensor_len : (i + 1) * tensor_len]
                        ctx_i = norm_i, tensor_shape
                        summed_tensor += compression.decompress(tensor_compressed_i, ctx_i, params)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'onebit':
                tensor_compressed, ctx = compression.compress(tensor, params)
                quantum_mid, lower, tensor_shape = ctx
                if comm_method == 'allgather':
                    tensor_compressed_1d = allgather(tensor_compressed)
                    quantum_mid_1d = allgather(quantum_mid)
                    lower_1d = allgather(lower)
                    tensor_len = elemnum
                    summed_tensor = tf.zeros_like(tensor)
                    for i in range(size()):
                        tensor_compressed_i = tensor_compressed_1d[i * tensor_len:(i + 1) * tensor_len]
                        ctx_i = quantum_mid_1d[i], lower_1d[i], tensor_shape
                        tensor_decompress = compression.decompress(tensor_compressed_i, ctx_i, params)
                        summed_tensor +=  tensor_decompress
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

                elif comm_method == 'allreduce':
                    tensor_decompress = compression.decompress(tensor_compressed, ctx, params)
                    tensor_decompress = tf.cast(tensor_decompress, dtype=tf.float16)
                    summed_tensor = _allreduce(tensor_decompress)
                    summed_tensor = tf.cast(summed_tensor, dtype=tf.float32)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'terngrad':
                tensor_compressed, ctx = compression.compress(tensor, params)
                scaler, tensor_shape = ctx
                if comm_method == 'broadcast':
                    bd_scalers = {}
                    bd_tensor_compressed = {}
                    bd_tensors = {}
                    for ranki in range(size()):
                        bd_scalers[ranki] = broadcast(scaler, root_rank=ranki, name=None)
                        bd_tensor_compressed[ranki] = broadcast(tensor_compressed, root_rank=ranki, name=None)
                        tensor_compressed_i = bd_tensor_compressed[ranki]
                        ctx_i = bd_scalers[ranki], tensor_shape
                        bd_tensors[ranki] = compression.decompress(tensor_compressed_i, ctx_i, params)
                    summed_tensor = tf.math.add_n(list(bd_tensors.values()))
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

                elif comm_method == 'allgather':
                    tensor_compressed_1d = allgather(tensor_compressed)
                    scalers_1d = allgather(tf.reshape(scaler, [-1]))
                    tensor_len = elemnum // 4 + 1
                    summed_tensor = tf.zeros_like(tensor)
                    for i in range(size()):
                        ctx_i = scalers_1d[i], tensor_shape
                        tensor_compressed_i = tensor_compressed_1d[i * tensor_len:(i + 1) * tensor_len]
                        summed_tensor += compression.decompress(tensor_compressed_i, ctx_i, params)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'dgc':  # dgc is deep gradient compression
                tensor_compressed, ctx = compression.compress(tensor, params)
                indices, tensor_shape = ctx
                if comm_method == 'allgather':
                    index_size = tf.reshape(tf.size(indices), [-1])
                    tensor_compressed_1d = allgather(tensor_compressed)
                    indices_1d = allgather(indices)
                    indices_size_1d = allgather(index_size)

                    summed_tensor = tf.zeros_like(tensor)
                    a = 0
                    for i in range(size()):
                        b = a + indices_size_1d[i]
                        indices_i = indices_1d[a:b]
                        tensor_compressed_i = tensor_compressed_1d[a:b]
                        ctx_i = indices_i, tensor_shape
                        summed_tensor += compression.decompress(tensor_compressed_i, ctx_i, params)
                        a = b
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'adaq':  # adaq is adaptive quantization
                _, ctx = compression.compress(tensor, params)
                plus_indices, plus_mean, minus_indices, minus_mean = ctx
                if comm_method == 'allgather':

                    plus_indices_size = tf.reshape(tf.size(plus_indices), [-1])
                    minus_indices_size = tf.reshape(tf.size(minus_indices), [-1])

                    plus_indices_1d = allgather(plus_indices)
                    plus_mean_1d = allgather(plus_mean)
                    plus_indices_size_1d = allgather(plus_indices_size)

                    minus_indices_1d = allgather(minus_indices)
                    minus_mean_1d = allgather(minus_mean)
                    minus_indices_size_1d = allgather(minus_indices_size)

                    summed_tensor = tf.zeros_like(tensor)
                    ap, am = 0, 0
                    for i in range(size()):
                        bp = ap + plus_indices_size_1d[i]
                        bm = am + minus_indices_size_1d[i]

                        plus_mean_i = plus_mean_1d[i]
                        minus_mean_i = minus_mean_1d[i]
                        plus_indices_i = plus_indices_1d[ap:bp]
                        minus_indices_i = minus_indices_1d[am:bm]

                        ctx_i = plus_indices_i, plus_mean_i, minus_indices_i, minus_mean_i
                        tensor_decompress = compression.decompress(tensor, ctx_i, params)
                        summed_tensor += tensor_decompress
                        ap, am = bp, bm
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'adas':
                tensor_compressed, ctx = compression.compress(tensor, params)
                indices, tensor_shape = ctx
                if comm_method == 'allgather':
                    index_size = tf.reshape(tf.size(indices), [-1])
                    tensor_compressed_1d = allgather(tensor_compressed)
                    indices_1d = allgather(indices)
                    indices_size_1d = allgather(index_size)

                    summed_tensor = tf.zeros_like(tensor)
                    a = 0
                    for i in range(size()):
                        b = a + indices_size_1d[i]
                        indices_i = indices_1d[a:b]
                        tensor_compressed_i = tensor_compressed_1d[a:b]
                        ctx_i = indices_i, tensor_shape
                        summed_tensor += compression.decompress(tensor_compressed_i, ctx_i, params)
                        a = b
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'powersgd':
                if tensor_rank == 1:
                    new_tensor = _allreduce(tensor) / horovod_size
                elif tensor_rank > 1:
                    _, ctx = compression.compress(tensor, params)
                    new_tensor = compression.decompress(tensor, ctx, params)

            elif compress_method == '8bit':
                tensor_compressed, ctx = compression.compress(tensor, params)
                scaler, tensor_shape = ctx
                if comm_method == 'broadcast':
                    bd_scalers = {}
                    bd_tensor_compressed = {}
                    bd_tensors = {}
                    for ranki in range(size()):
                        bd_scalers[ranki] = broadcast(scaler, root_rank=ranki, name=None)
                        bd_tensor_compressed[ranki] = broadcast(tensor_compressed, root_rank=ranki, name=None)
                        tensor_compressed_i = bd_tensor_compressed[ranki]
                        ctx_i = bd_scalers[ranki], tensor_shape
                        bd_tensors[ranki] = compression.decompress(tensor_compressed_i, ctx_i, params)
                    summed_tensor = tf.math.add_n(list(bd_tensors.values()))
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

                elif comm_method == 'allgather':
                    tensor_compressed_1d = allgather(tensor_compressed)
                    scalers_1d = allgather(scaler)
                    tensor_len = elemnum
                    summed_tensor = tf.zeros_like(tensor)
                    for i in range(size()):
                        ctx_i = scalers_1d[i], tensor_shape
                        tensor_compressed_i = tensor_compressed_1d[i * tensor_len:(i + 1) * tensor_len]
                        summed_tensor += compression.decompress(tensor_compressed_i, ctx_i, params)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'natural':
                tensor_compressed, ctx = compression.compress(tensor, params)
                sign, tensor_shape = ctx
                if comm_method == 'broadcast':
                    bd_sign = {}
                    bd_tensor_compressed = {}
                    bd_tensors = {}
                    for ranki in range(size()):
                        bd_sign[ranki] = broadcast(sign, root_rank=ranki, name=None)
                        bd_tensor_compressed[ranki] = broadcast(tensor_compressed, root_rank=ranki, name=None)
                        tensor_compressed_i = bd_tensor_compressed[ranki]
                        ctx_i = bd_sign[ranki], tensor_shape
                        bd_tensors[ranki] = compression.decompress(tensor_compressed_i, ctx_i, params)
                    summed_tensor = tf.math.add_n(list(bd_tensors.values()))
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

                elif comm_method == 'allgather':
                    tensor_compressed_1d = allgather(tensor_compressed)
                    sign_1d = allgather(tf.reshape(sign, [-1]))
                    tensor_len = elemnum
                    summed_tensor = tf.zeros_like(tensor)
                    for i in range(size()):
                        sign_i = sign_1d[i * tensor_len:(i + 1) * tensor_len]
                        ctx_i = sign_i, tensor_shape
                        tensor_compressed_i = tensor_compressed_1d[i * tensor_len:(i + 1) * tensor_len]
                        summed_tensor += compression.decompress(tensor_compressed_i, ctx_i, params)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor
        return new_tensor


def broadcast_global_variables(root_rank):
    """Broadcasts all global variables from root rank to all other processes.

    Arguments:
        root_rank: rank of the process from which global variables will be broadcasted
        to all other processes.
    """
    return tf.group(*[tf.assign(var, broadcast(var, root_rank))
                      for var in tf.global_variables()])


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
                 device_sparse='', compression=Compression.none, sparse_as_dense=False,
                   params=None,
                   ):
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
        self._params = params
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
            averaged_gradients = []
            with tf.name_scope(self._name + "_Allreduce"):
                for grad, var in gradients:
                    if grad is not None:
                        if self._sparse_as_dense and \
                                isinstance(grad, tf.IndexedSlices):
                            grad = tf.convert_to_tensor(grad)
                        avg_grad = allreduce(grad,
                                             device_dense=self._device_dense,
                                             device_sparse=self._device_sparse,
                                             compression=self._compression,
                                             params = self._params,)
                        averaged_gradients.append((avg_grad, var))
                    else:
                        averaged_gradients.append((None, var))
            return averaged_gradients
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
