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


def allreduce(tensor, average=True, device_dense='', device_sparse='', compression=Compression.none,
              compress_ratio=None,
              threshold_val=None,
              comm_method=None,
              compress_method=None,
              use_memory=None,
              momentum=None,
              gradient_clipping=False,
              quantum_num=256,
              learning_rate=0.1,
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
            if (compress_method is None) or (compress_method == 'none'):
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
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                elemnum = tf.size(tensor)
                shape = tf.shape(tensor)
                tensor = tf.reshape(tensor, [-1])
                with tf.Session():
                    elemnum = elemnum.eval()
                tensor_compressed, indices = compression.compress(tensor, elemnum, shape, compress_ratio, use_memory)
                if comm_method == 'allreduce':

                    summed_tensor_compressed = _allreduce(tensor_compressed)
                    summed_tensor = compression.decompress(summed_tensor_compressed, elemnum, shape, indices)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor


                elif comm_method == 'broadcast':

                    bd_tensor_sparsed = {}

                    for ranki in range(size()):
                        bd_tensor_sparsed[str(ranki)] = broadcast(tensor_compressed, root_rank=ranki, name=None)

                    summed_tensor_compressed = tf.math.add_n(list(bd_tensor_sparsed.values()))
                    summed_tensor = compression.decompress(summed_tensor_compressed, elemnum, shape, indices)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor


                elif comm_method == 'allgather':

                    tensor_compressed_1d = allgather(tensor_compressed)

                    tensor_len = max(1, int(elemnum * compress_ratio))
                    summed_tensor_compressed = tf.Variable(tf.zeros([tensor_len], dtype=tf.float32))

                    for i in range(size()):
                        summed_tensor_compressed += tensor_compressed_1d[i * tensor_len:(i + 1) * tensor_len]

                    summed_tensor = compression.decompress(summed_tensor_compressed, elemnum, shape, indices)
                    new_tensor = (summed_tensor / horovod_size) if average else summed_tensor

            elif compress_method == 'topk':
                compression = comp_dict[compress_method]
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                elemnum = tf.size(tensor)
                shape = tf.shape(tensor)
                tensor = tf.reshape(tensor, [-1])
                with tf.Session():
                    elemnum = elemnum.eval()

                tensor_sparsed, indices = compression.compress(tensor, elemnum, compress_ratio, use_memory)
                if comm_method == 'broadcast':

                    bd_indices = {}
                    bd_tensor_sparsed = {}
                    bd_tensors = {}

                    for ranki in range(size()):
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

                    tensor_sparsed_1d = allgather(tensor_sparsed)
                    indices_1d = allgather(indices)
                    tensor_len = max(1, int(elemnum * compress_ratio))
                    summed_tensor = tf.zeros_like(tensor)

                    for i in range(size()):
                        # if i is not local_rank:
                        zero_tensor = tf.Variable(
                            tf.zeros_like(tensor))  # tf.Variable(tf.zeros_like(tf.reshape(tensor, [-1])))
                        index = indices_1d[i * tensor_len:(i + 1) * tensor_len]
                        summed_tensor = tf.math.add(summed_tensor, tf.scatter_update(zero_tensor, index,
                                                        tensor_sparsed_1d[i * tensor_len:(i + 1) * tensor_len]))

                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

            elif compress_method in ['signsgd', 'signum']:
                compression = comp_dict[compress_method]
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                elemnum = tf.size(tensor)
                shape = tf.shape(tensor)
                with tf.Session():
                    elemnum = elemnum.eval()

                tensor = tf.reshape(tensor, [-1])


                if comm_method == 'allgather':
                    if use_memory:
                        sign, mean = compression.compress(tensor, shape, use_memory, momentum, learning_rate)
                        mean = tf.reshape(mean, [-1])
                        mean_1d = allgather(mean)
                    else:
                        sign = compression.compress(tensor, shape, use_memory, momentum, learning_rate)
                    sign_1d = allgather(sign)
                    tensor_len = elemnum
                    summed_tensor = tf.Variable(tf.zeros_like(tensor))

                    if use_memory:
                        for i in range(size()):
                            tensor_encoded = sign_1d[i * tensor_len:(i + 1) * tensor_len]
                            mean = mean_1d[i]
                            tensor_decoded = tf.cast(tensor_encoded, dtype=tf.float32) * 2.0 - 1.0
                            delta = mean * tensor_decoded
                            summed_tensor = tf.math.add(summed_tensor, delta)

                        new_tensor = (tf.div(summed_tensor, horovod_size)
                                      if average else summed_tensor)
                        new_tensor = new_tensor / learning_rate
                        new_tensor = tf.reshape(new_tensor, shape)
                    else:
                        for i in range(size()):
                            tensor_encoded = sign_1d[i * tensor_len:(i + 1) * tensor_len]
                            tensor_decoded = tf.cast(tensor_encoded, dtype=tf.float32) * 2.0 - 1.0
                            summed_tensor = tf.math.add(summed_tensor, tensor_decoded)

                        summed_tensor_sign = tf.cast(tf.math.greater_equal(summed_tensor, 0), dtype=tensor.dtype)

                        new_tensor = summed_tensor_sign * 2.0 - 1.0
                        new_tensor = tf.reshape(new_tensor, shape)

            elif compress_method == 'onebit':
                compression = Compression.onebit
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                shape = tf.shape(tensor)

                tensor = tf.reshape(tensor, [-1])
                quant_tensor, ctx = compression.compress(tensor, use_memory, horovod_size)
                quantum_mid, lower = ctx

                if comm_method == 'allgather':

                    quant_tensor_1d = allgather(quant_tensor)
                    quantum_mid_1d = allgather(quantum_mid)
                    lower_1d = allgather(lower)
                    tensor_len = tf.size(tensor)
                    summed_tensor = tf.Variable(tf.zeros_like(tensor))

                    for i in range(size()):
                        quant_tensorx = quant_tensor_1d[i * tensor_len:(i + 1) * tensor_len]
                        quantum_midx = quantum_mid_1d[i]
                        lowerx = lower_1d[i]
                        tensor_decompress = compression.decompress(quant_tensorx, [quantum_midx,lowerx])
                        summed_tensor = tf.math.add(summed_tensor, tensor_decompress)

                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

            elif compress_method == 'adas':
                compression = comp_dict[compress_method]
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                elemnum = tf.size(tensor)
                shape = tf.shape(tensor)
                tensor = tf.reshape(tensor, [-1])
                with tf.Session():
                    elemnum = elemnum.eval()

                tensor_sparsed, indices = compression.compress(tensor, elemnum, compress_ratio, use_memory)
                if comm_method == 'allgather':

                    index_size = tf.reshape(tf.size(indices), [-1])

                    tensor_sparsed_1d = allgather(tensor_sparsed)
                    indices_1d = allgather(indices)
                    indices_size_1d = allgather(index_size)

                    summed_tensor = tf.Variable(tf.zeros_like(tensor))
                    a = 0
                    for i in range(size()):
                        b = a + indices_size_1d[i]
                        zero_tensor = tf.Variable(tf.zeros_like(tensor))
                        index = indices_1d[a:b]
                        summed_tensor = tf.math.add(summed_tensor,
                                                    tf.scatter_update(zero_tensor, index, tensor_sparsed_1d[a:b]))
                        a = b

                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

            elif compress_method == 'terngrad':
                compression = Compression.terngrad
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                elemnum = tf.size(tensor)
                shape = tf.shape(tensor)
                with tf.Session():
                    elemnum = elemnum.eval()

                tensor_encoded, scaler = compression.compress(tensor, shape, use_memory)

                if comm_method == 'broadcast':

                    bd_scalers = {}
                    bd_tensor_encoded = {}
                    bd_tensor_decoded = {}
                    for ranki in range(size()):
                        bd_scalers[str(ranki)] = broadcast(scaler, root_rank=ranki, name=None)
                        bd_tensor_encoded[str(ranki)] = broadcast(tensor_encoded, root_rank=ranki, name=None)
                        bd_tensor_decoded[str(ranki)] = compression.decompress(bd_tensor_encoded[str(ranki)],
                                                                               bd_scalers[str(ranki)], shape)
                    summed_tensor = tf.math.add_n(list(bd_tensor_decoded.values()))
                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)


                elif comm_method == 'allgather':

                    # scaler = tf.convert_to_tensor(1.0) #for testing signSGD
                    scaler_shape = tf.shape(scaler)
                    tensor_encoded_1d = allgather(tensor_encoded)
                    scalers_1d = allgather(tf.reshape(scaler, [-1]))
                    tensor_len = elemnum // 4 + 1
                    summed_tensor = tf.zeros_like(tensor)

                    for i in range(size()):
                        scaler = tf.reshape(scalers_1d[i:(i + 1)], scaler_shape)
                        tensor_encoded = tensor_encoded_1d[i * tensor_len:(i + 1) * tensor_len]
                        summed_tensor = tf.math.add(summed_tensor,
                                                    compression.decompress(tensor_encoded, scaler, shape))

                    # summed_tensor = tf.sign(summed_tensor) #for testing signSGD
                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

            elif compress_method == 'qsgd':
                compression = Compression.qsgd
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                shape = tf.shape(tensor)
                tensor = tf.reshape(tensor, [-1])
                encode_tensor = compression.compress(tensor, shape, quantum_num, use_memory)
                if comm_method == 'broadcast':
                    bd_encode_tensor = {}
                    bd_tensor_decoded = {}
                    for ranki in range(size()):
                        bd_encode_tensor[str(ranki)] = broadcast(encode_tensor, root_rank=ranki, name=None)
                        bd_tensor_decoded[str(ranki)] = compression.decompress(tensor, bd_encode_tensor[str(ranki)], quantum_num)
                    summed_tensor = tf.math.add_n(list(bd_tensor_decoded.values()))
                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

                elif comm_method == 'allgather':

                    encode_tensor_1d = allgather(encode_tensor)
                    bits = int(math.log(quantum_num, 2) + 1)
                    length = bits + 1
                    summed_tensor = tf.zeros_like(tensor)
                    for i in range(size()):
                        encode_tensorx = encode_tensor_1d[i * length:(i + 1) * length]
                        summed_tensor = summed_tensor + compression.decompress(tensor, encode_tensorx, quantum_num)
                    new_tensor = (tf.div(summed_tensor, horovod_size) if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

            elif compress_method == 'threshold':
                compression = Compression.threshold
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                elemnum = tf.size(tensor)
                shape = tf.shape(tensor)
                tensor = tf.reshape(tensor, [-1])
                with tf.Session():
                    elemnum = elemnum.eval()

                tensor_sparsed, indices = compression.compress(tensor, elemnum, threshold_val, use_memory)
                if comm_method == 'allgather':

                    index_size = tf.reshape(tf.size(indices), [-1])

                    tensor_sparsed_1d = allgather(tensor_sparsed)
                    indices_1d = allgather(indices)
                    indices_size_1d = allgather(index_size)

                    summed_tensor = tf.Variable(tf.zeros_like(tensor))
                    a = 0
                    for i in range(size()):
                        b = a + indices_size_1d[i]
                        zero_tensor = tf.Variable(tf.zeros_like(tensor))
                        index = indices_1d[a:b]
                        summed_tensor = tf.math.add(summed_tensor,
                                                    tf.scatter_update(zero_tensor, index, tensor_sparsed_1d[a:b]))
                        a = b

                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

            elif compress_method == 'dgc':  # dgc is deep gradient compression
                compression = Compression.dgc
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                elemnum = tf.size(tensor)
                shape = tf.shape(tensor)
                tensor = tf.reshape(tensor, [-1])
                with tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                    elemnum = elemnum.eval()

                tensor_sparsed, indices = compression.compress(tensor, elemnum, compress_ratio, use_memory, momentum,
                                                               horovod_size, gradient_clipping)
                if comm_method == 'allgather':

                    index_size = tf.reshape(tf.size(indices), [-1])

                    tensor_sparsed_1d = allgather(tensor_sparsed)
                    indices_1d = allgather(indices)
                    indices_size_1d = allgather(index_size)

                    summed_tensor = tf.Variable(tf.zeros_like(tensor))
                    a = 0
                    for i in range(size()):
                        b = a + indices_size_1d[i]
                        zero_tensor = tf.Variable(tf.zeros_like(tensor))
                        index = indices_1d[a:b]
                        summed_tensor = tf.math.add(summed_tensor,
                                                    tf.scatter_update(zero_tensor, index, tensor_sparsed_1d[a:b]))
                        a = b

                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)

            elif compress_method == 'adaq':  # adaq is adaptive quantization
                compression = Compression.adaq
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                shape = tf.shape(tensor)
                tensor = tf.reshape(tensor, [-1])
                plus_indices, plus_mean, minus_indices, minus_mean = compression.compress(tensor, compress_ratio,
                                                                                          use_memory)
                if comm_method == 'allgather':

                    plus_indices_size = tf.reshape(tf.size(plus_indices), [-1])
                    minus_indices_size = tf.reshape(tf.size(minus_indices), [-1])

                    plus_indices_1d = allgather(plus_indices)
                    plus_mean_1d = allgather(plus_mean)
                    plus_indices_size_1d = allgather(plus_indices_size)

                    minus_indices_1d = allgather(minus_indices)
                    minus_mean_1d = allgather(minus_mean)
                    minus_indices_size_1d = allgather(minus_indices_size)

                    summed_tensor = tf.Variable(tf.zeros_like(tensor))
                    ap = 0
                    am = 0
                    for i in range(size()):
                        bp = ap + plus_indices_size_1d[i]
                        bm = am + minus_indices_size_1d[i]

                        plus_meanx = plus_mean_1d[i]
                        minus_meanx = minus_mean_1d[i]

                        plus_indicesx = plus_indices_1d[ap:bp]
                        minus_indicesx = minus_indices_1d[am:bm]

                        tensor_decompress = compression.decompress(tensor, shape, plus_indicesx, plus_meanx,
                                                                   minus_indicesx, minus_meanx)
                        summed_tensor = tf.math.add(summed_tensor, tensor_decompress)

                        ap = bp
                        am = bm

                    ## concatenate tensors, then allgather
                    # plus_indices, plus_mean, minus_indices, minus_mean = compression.compress(tensor, compress_ratio)
                    # plus_indices_size = tf.cast(tf.reshape(tf.size(plus_indices), [-1]),dtype=tf.float32)
                    # minus_indices_size = tf.cast(tf.reshape(tf.size(minus_indices), [-1]),dtype=tf.float32)
                    # indices_concat = tf.concat([plus_indices, minus_indices],axis=0)
                    # others_concat = tf.concat([plus_mean, minus_mean, plus_indices_size, minus_indices_size],axis=0)
                    # indices_concat_1d = allgather(indices_concat)
                    # others_concat_1d = allgather(others_concat)
                    # summed_tensor = tf.Variable(tf.zeros_like(tensor))
                    # a = 0
                    # for i in range(size()):
                    #
                    #     others_concatx = others_concat_1d[4*i:4*(i+1)]
                    #     psize = tf.cast(others_concatx[2],dtype=tf.int32)
                    #     msize = tf.cast(others_concatx[3],dtype=tf.int32)
                    #     b = a + psize + msize
                    #     indices_concatx = indices_concat_1d[a:b]
                    #     plus_indicesx = indices_concatx[:psize]
                    #     minus_indicesx = indices_concatx[psize:]
                    #     plus_meanx = others_concatx[0]
                    #     minus_meanx = others_concatx[1]
                    #
                    #     tensor_decompress = compression.decompress(tensor, shape, plus_indicesx, plus_meanx,
                    #                                                minus_indicesx, minus_meanx)
                    #     summed_tensor = tf.math.add(summed_tensor, tensor_decompress)
                    #
                    #     a = b

                    new_tensor = (tf.div(summed_tensor, horovod_size)
                                  if average else summed_tensor)
                    new_tensor = tf.reshape(new_tensor, shape)
            elif compress_method == 'powersgd':
                compression = Compression.powersgd
                horovod_size = tf.cast(size(), dtype=tensor.dtype)
                with tf.Session():
                    tensor_rank = tf.rank(tensor).eval()
                if tensor_rank == 1:
                    new_tensor = _allreduce(tensor) / horovod_size
                elif tensor_rank > 1:
                    tensor_compressed = compression.compress(tensor, use_memory, horovod_size)
                    new_tensor = tensor_compressed
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
                   compress_ratio=None,
                   threshold_val=None,
                   comm_method=None,
                   compress_method=None,
                   use_memory=None,
                   momentum=None,
                   gradient_clipping=False,
                   quantum_num=256,
                   learning_rate=0.1,
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
        self._compress_ratio = compress_ratio
        self._threshold_val = threshold_val
        self._comm_method = comm_method
        self._compress_method = compress_method
        self._use_memory = use_memory
        self._momentum = momentum
        self._gradient_clipping = gradient_clipping
        self._quantum_num = quantum_num
        self._learning_rate = learning_rate
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
                                             compress_ratio = self._compress_ratio,
                                             threshold_val = self._threshold_val,
                                             comm_method = self._comm_method,
                                             compress_method = self._compress_method,
                                             use_memory = self._use_memory,
                                             momentum = self._momentum,
                                             gradient_clipping = self._gradient_clipping,
                                             quantum_num=self._quantum_num,
                                             learning_rate=self._learning_rate)
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