"""Gradient compression algorithms."""

from __future__ import division
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import random, math
from horovod.tensorflow.mpi_ops import _allreduce
from horovod.tensorflow.mpi_ops import rank
import numpy as np
import wandb


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    residuals = {}

    @staticmethod
    def compress(tensor, params):
        """Compresses a tensor and returns a list of compressed tensors with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensors, ctx, params):
        """Decompress a list of compressed tensors with the given context."""
        pass

    @classmethod
    def memory_compensate(cls, tensor, params):
        """Update the tensor with the residuals."""
        use_memory = params['use_memory']
        beta = params['beta']
        gamma = params['gamma']
        if use_memory:
            name = tensor.name
            cls.residuals[tensor.name] = tf.Variable(tf.zeros_like(tensor), trainable=False)
            tensor = beta * cls.residuals[name] + gamma * tensor
        return tensor

    @classmethod
    def memory_update(cls, tensor, tensor_compensate, tensor_compressed, ctx, params):
        """Update the residuals."""
        use_memory = params['use_memory']
        if use_memory:
            name = tensor.name
            params['mem_mode'] = 1
            tensor_decompressed = cls.decompress(tensor_compressed, ctx, params)
            params['mem_mode'] = 0
            delta = tensor_compensate - tensor_decompressed
            memory_update_op = cls.residuals[name].assign(delta)
        return [memory_update_op] if use_memory else []

    @staticmethod
    def aggregate(tensors, params):
        """Aggregate a list of tensors."""
        average = params['average']
        agged_tensor = tf.math.add_n(tensors)
        horovod_size = tf.cast(params["horovod_size"], dtype=agged_tensor.dtype)
        agged_tensor = (agged_tensor / horovod_size) if average else agged_tensor
        return agged_tensor

class NoneCompressor(Compressor):
    """Default no-op compression."""

    @staticmethod
    def compress(tensor, params):
        """Returns the tensor unmodified."""
        params['tensors_size_are_same'] = True
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx, params):
        """Returns the tensor unmodified."""
        return tensor

class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    @staticmethod
    def compress(tensor, params):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor

        if tensor.dtype.is_floating:
            # Only allow compression from other floating point types
            tensor_compressed = tf.cast(tensor, dtype=tf.float16)
        params['tensors_size_are_same'] = True
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx, params):
        """Upcasts the tensor to the initialization dtype."""

        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating:
            tensor_decompressed = tf.cast(tensor, dtype=dtype)
        return tensor_decompressed

class SparsifierCompressor(Compressor):

    global_step = 0

    @staticmethod
    def compress(tensor, params):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]

        k = max(1, int(elemnum * compress_ratio))
        params['K'] = k
        params['values_size'] = k

        ################################################################################
        if params["compress_method"] == "topk":
            _, indices = tf.math.top_k(tf.math.abs(tensor_flatten), k, sorted=False)
            indices = tf.sort(indices, axis=0, direction='ASCENDING')
            values = tf.gather(tensor_flatten, indices)

        elif params["compress_method"] == "randomk":
            # all_indices = tf.range(elemnum, dtype=tf.int32)
            # indices = tf.random.shuffle(all_indices, seed=None)[:k]
            # indices = tf.sort(indices, axis=0, direction='ASCENDING')

            all_indices = tf.range(elemnum, dtype=tf.int32)
            h = hash(tensor.name) + RandomkCompressor.global_step
            tf.compat.v1.set_random_seed(1)
            indices = tf.random.shuffle(all_indices, seed=h)[:k]
            indices = tf.sort(indices, axis=0, direction='ASCENDING')
            RandomkCompressor.global_step += 1
            values = tf.gather(tensor_flatten, indices)

        # elif params["compress_method"] == "threshold":
        #     threshold_val = params["threshold_val"]
        #     thr_mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), threshold_val)
        #     values = tf.boolean_mask(tensor_flatten, thr_mask)
        #     indices = tf.reshape(tf.where(thr_mask), [-1])
        #     params['K'] = k
        #     params['values_size'] =  k
        ################################################################################

        if params['encoding'] is not None:
            filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            library = load_library.load_op_library(filename)

            if params['encoding'] == "integer_compression":
                integer_compressor = library.integer_compressor

                values = tf.bitcast(values, tf.int32)
                values_shape = tf.shape(values)
                indices = tf.bitcast(indices, tf.uint32)
                compressed_indices = integer_compressor(indices,
                                                        tf.train.get_or_create_global_step(),
                                                        logs_path=params['bloom_logs_path'],
                                                        gradient_id=params['gradient_id'],
                                                        verbosity_frequency=params['bloom_verbosity_frequency'],
                                                        verbosity=params['bloom_verbosity'],
                                                        rank=rank(),
                                                        code=params['code'])
                compressed_indices = tf.bitcast(compressed_indices, tf.int32)

            elif params['encoding'] == "rle_compression":
                if params['code'] == "0_8":
                    rle_compressor = library.rle_compressor_v0_code8
                    values = tf.bitcast(values, tf.uint8)
                    values_shape = tf.shape(values)
                    values = tf.reshape(values, [-1])
                    params['values_size'] = k*4

                # elif params['code'] == "1_8":
                #     rle_compressor = library.rle_compressor_v1_code8
                #     values = tf.bitcast(values, tf.uint8)
                #     values_shape = tf.shape(values)
                #     values = tf.reshape(values, [-1])
                #     params['values_size'] = k*4
                #
                # elif params['code'] == "0_32":
                #     rle_compressor = library.rle_compressor_v0_code32
                #     values = tf.bitcast(values, tf.int32)
                #     params['values_size'] = k
                #
                # elif params['code'] == "1_32":
                #     rle_compressor = library.rle_compressor_v1_code32
                #     values = tf.bitcast(values, tf.int32)
                #     params['values_size'] = k

                compressed_indices = rle_compressor(indices, elemnum,
                                                    tf.train.get_or_create_global_step(),
                                                    logs_path=params['bloom_logs_path'],
                                                    gradient_id=params['gradient_id'],
                                                    verbosity_frequency=params['bloom_verbosity_frequency'],
                                                    verbosity=params['bloom_verbosity'],
                                                    rank=rank())
            params['tensors_size_are_same'] = False

        else:
            compressed_indices = indices
            values = tf.bitcast(values, tf.int32)
            params['values_size'] = k
            values_shape = tf.shape(values)
            params['tensors_size_are_same'] = True

        tensor_compressed = tf.concat([values, compressed_indices], 0)
        ctx = [tensor_shape, values_shape]
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        compressed_tensor_size = tf.math.reduce_prod(tf.shape(tensor_compressed))
        values, indices = tf.split(tensor_compressed, [params['values_size'], compressed_tensor_size-params['values_size']])
        values = tf.reshape(values, ctx[1])
        values = tf.bitcast(values, tf.float32)
        tensor_shape = ctx[0]
        tensor_size = tf.math.reduce_prod(tensor_shape)

        if params['encoding'] is not None:
            filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            library = load_library.load_op_library(filename)

            if params['encoding'] == "integer_compression":
                integer_decompressor = library.integer_decompressor
                indices = tf.bitcast(indices, tf.uint32)
                decompressed_indices = integer_decompressor(indices, params['K'],
                                                            tf.train.get_or_create_global_step(),
                                                            logs_path=params['bloom_logs_path'],
                                                            gradient_id=params['gradient_id'],
                                                            verbosity_frequency=params['bloom_verbosity_frequency'],
                                                            verbosity=params['bloom_verbosity'],
                                                            suffix=params['suffix'],
                                                            rank=rank(),
                                                            code=params['code'])

                decompressed_indices = tf.bitcast(decompressed_indices, tf.int32)

            elif params['encoding'] == "rle_compression":
                if params['code'] == "0_8":
                    rle_decompressor = library.rle_decompressor_v0_code8
                # elif params['code'] == "1_8":
                #     rle_decompressor = library.rle_decompressor_v1_code8
                # elif params['code'] == "0_32":
                #     rle_decompressor = library.rle_decompressor_v0_code32
                # elif params['code'] == "1_32":
                #     rle_decompressor = library.rle_decompressor_v1_code32

                decompressed_indices = rle_decompressor(indices, params['K'],
                                                        tf.train.get_or_create_global_step(),
                                                        logs_path=params['bloom_logs_path'],
                                                        gradient_id=params['gradient_id'],
                                                        verbosity_frequency=params['bloom_verbosity_frequency'],
                                                        verbosity=params['bloom_verbosity'],
                                                        suffix=params['suffix'],
                                                        rank=rank())
        else:
            decompressed_indices = indices

        decompressed_indices = tf.expand_dims(decompressed_indices, 1)
        tensor_decompressed = tf.scatter_nd(decompressed_indices, values, [tensor_size])
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)

        # zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32), trainable=False)
        # op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.float32))
        # with tf.control_dependencies([op]):
        #     tensor_decompressed = tf.scatter_update(zero_tensor, decompressed_indices, values)
        # tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed



class RandomkCompressor(Compressor):
    global_step = 0

    @staticmethod
    def compress(tensor, params):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]
        all_indices = tf.range(elemnum, dtype=tf.int32)
        h = hash(tensor.name) + RandomkCompressor.global_step
        tf.compat.v1.set_random_seed(1)
        indices = tf.random.shuffle(all_indices, seed=h)[:max(1, int(elemnum * compress_ratio))]
        RandomkCompressor.global_step += 1
        values = tf.gather(tensor_flatten, indices)
        ctx = indices, tensor_shape, tensor_flatten
        tensor_compressed = values
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, tensor_shape, tensor_flatten = ctx
        values = tensor_compressed
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32), trainable=False)
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.float32))
        with tf.control_dependencies([op]):
            tensor_decompressed = tf.scatter_update(zero_tensor, indices, values)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed

class TopKCompressor(Compressor):

    @staticmethod
    def compress(tensor, params):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]

        k = max(1, int(elemnum * compress_ratio))
        params['topk_k'] = k
        params['values_size'] = k

        _, indices = tf.math.top_k(tf.math.abs(tensor_flatten), k, sorted=False)
        indices = tf.sort(indices, axis=0, direction='ASCENDING')
        values = tf.gather(tensor_flatten, indices)

        if params['encoding'] is not None:
            filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            library = load_library.load_op_library(filename)

            if params['encoding'] == "integer_compression":
                integer_compressor = library.integer_compressor

                values = tf.bitcast(values, tf.int32)
                values_shape = tf.shape(values)
                indices = tf.bitcast(indices, tf.uint32)
                compressed_indices = integer_compressor(indices,
                                                        tf.train.get_or_create_global_step(),
                                                        logs_path=params['bloom_logs_path'],
                                                        gradient_id=params['gradient_id'],
                                                        verbosity_frequency=params['bloom_verbosity_frequency'],
                                                        verbosity=params['bloom_verbosity'],
                                                        rank=rank(),
                                                        code=params['code'])
                compressed_indices = tf.bitcast(compressed_indices, tf.int32)

            elif params['encoding'] == "rle_compression":
                if params['code'] == "0_8":
                    rle_compressor = library.rle_compressor_v0_code8
                    values = tf.bitcast(values, tf.uint8)
                    values_shape = tf.shape(values)
                    values = tf.reshape(values, [-1])
                    params['values_size'] = k*4

                # elif params['code'] == "1_8":
                #     rle_compressor = library.rle_compressor_v1_code8
                #     values = tf.bitcast(values, tf.uint8)
                #     values_shape = tf.shape(values)
                #     values = tf.reshape(values, [-1])
                #     params['values_size'] = k*4
                #
                # elif params['code'] == "0_32":
                #     rle_compressor = library.rle_compressor_v0_code32
                #     values = tf.bitcast(values, tf.int32)
                #     params['values_size'] = k
                #
                # elif params['code'] == "1_32":
                #     rle_compressor = library.rle_compressor_v1_code32
                #     values = tf.bitcast(values, tf.int32)
                #     params['values_size'] = k

                compressed_indices = rle_compressor(indices, elemnum,
                                                    tf.train.get_or_create_global_step(),
                                                    logs_path=params['bloom_logs_path'],
                                                    gradient_id=params['gradient_id'],
                                                    verbosity_frequency=params['bloom_verbosity_frequency'],
                                                    verbosity=params['bloom_verbosity'],
                                                    rank=rank())
            params['tensors_size_are_same'] = False

        else:
            compressed_indices = indices
            values = tf.bitcast(values, tf.int32)
            params['values_size'] = k
            values_shape = tf.shape(values)
            params['tensors_size_are_same'] = True

        tensor_compressed = tf.concat([values, compressed_indices], 0)
        ctx = [tensor_shape, values_shape]
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        compressed_tensor_size = tf.math.reduce_prod(tf.shape(tensor_compressed))
        values, indices = tf.split(tensor_compressed, [params['values_size'], compressed_tensor_size-params['values_size']])
        values = tf.reshape(values, ctx[1])
        values = tf.bitcast(values, tf.float32)
        tensor_shape = ctx[0]
        tensor_size = tf.math.reduce_prod(tensor_shape)

        if params['encoding'] is not None:
            filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            library = load_library.load_op_library(filename)

            if params['encoding'] == "integer_compression":
                integer_decompressor = library.integer_decompressor
                indices = tf.bitcast(indices, tf.uint32)
                decompressed_indices = integer_decompressor(indices, params['topk_k'],
                                                            tf.train.get_or_create_global_step(),
                                                            logs_path=params['bloom_logs_path'],
                                                            gradient_id=params['gradient_id'],
                                                            verbosity_frequency=params['bloom_verbosity_frequency'],
                                                            verbosity=params['bloom_verbosity'],
                                                            suffix=params['suffix'],
                                                            rank=rank(),
                                                            code=params['code'])

                decompressed_indices = tf.bitcast(decompressed_indices, tf.int32)

            elif params['encoding'] == "rle_compression":
                if params['code'] == "0_8":
                    rle_decompressor = library.rle_decompressor_v0_code8
                elif params['code'] == "1_8":
                    rle_decompressor = library.rle_decompressor_v1_code8
                elif params['code'] == "0_32":
                    rle_decompressor = library.rle_decompressor_v0_code32
                elif params['code'] == "1_32":
                    rle_decompressor = library.rle_decompressor_v1_code32

                decompressed_indices = rle_decompressor(indices, params['topk_k'],
                                                        tf.train.get_or_create_global_step(),
                                                        logs_path=params['bloom_logs_path'],
                                                        gradient_id=params['gradient_id'],
                                                        verbosity_frequency=params['bloom_verbosity_frequency'],
                                                        verbosity=params['bloom_verbosity'],
                                                        suffix=params['suffix'],
                                                        rank=rank())
        else:
            decompressed_indices = indices

        # decompressed_indices = tf.expand_dims(decompressed_indices, 1)
        # tensor_decompressed = tf.scatter_nd(decompressed_indices, values, [tensor_size])
        # tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)

        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32), trainable=False)
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.float32))
        with tf.control_dependencies([op]):
            tensor_decompressed = tf.scatter_update(zero_tensor, decompressed_indices, values)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class Bloom_Filter_Compressor(Compressor):
    """"""
    @staticmethod
    def bloom_configuration(k, fpr):
        # Given FPR compute M and H
        m = (k*abs(math.log(fpr))) / (math.pow(math.log(2), 2))
        # Give bloom size in number of bytes ; bloom size must be a multiple of 8
        m = int(m/8)
        rem = m % 8
        if rem != 0 or m == 0:
            m += 1
        h = (m*8 / k)*math.log(2)
        h = int(math.ceil(h))
        return m, h

    @staticmethod
    def topk_indices(tensor, K):
        _, indices = tf.math.top_k(tf.math.abs(tensor), K, sorted=False)
        indices = tf.sort(indices, axis=0, direction='ASCENDING')
        return indices

    @staticmethod
    def randomk_indices(tensor_name, N, K):
        all_indices = tf.range(N, dtype=tf.int32)
        h = hash(tensor_name) + RandomkCompressor.global_step
        tf.compat.v1.set_random_seed(1)
        indices = tf.random.shuffle(all_indices, seed=h)[:K]
        indices = tf.sort(indices, axis=0, direction='ASCENDING')
        RandomkCompressor.global_step += 1
        return indices

    @staticmethod
    def compress(tensor, params):

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]

        compress_ratio = params["compress_ratio"]
        k = max(1, int(elemnum * compress_ratio))

        # Configure bloom filter's m, k values
        assert params["bloom_fpr"] is not None, "False Positive Rate is None"
        params['m'], params['k'] = Bloom_Filter_Compressor.bloom_configuration(k, params["bloom_fpr"])
        assert params['k'] < 256, "Number of hash functions too big"

        params["bloom_config"].add_data(k, params['m']*8, params['k'], params["bloom_fpr"])
        params["throughput_info"].add_data(elemnum, elemnum/8,  params['m']*8, (params['m']*8)/8, elemnum-params['m']*8, (elemnum-params['m']*8)/8)

        if params['bloom_on'] == "topk":            # Topk Sparsification Method
            indices = Bloom_Filter_Compressor.topk_indices(tensor_flatten, k)
        elif params['bloom_on'] == "randomk":       # Randomk Sparsification Method
            indices = Bloom_Filter_Compressor.randomk_indices(tensor.name, elemnum, k)

        values = tf.gather(tensor_flatten, indices)
        values = tf.bitcast(values, tf.int32)

        filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
        library = load_library.load_op_library(filename)
        bloom_compressor = library.bloom_compressor

        log_initial_tensor = tf.bitcast(tensor_flatten, tf.int32)
        compressed_tensor = bloom_compressor(values, indices, log_initial_tensor, tf.train.get_or_create_global_step(),
                                             false_positives_aware=params['bloom_false_positives_aware'],
                                             policy=params['bloom_policy'],
                                             hash_num=params['k'],
                                             bloom_size=params['m'],
                                             bloom_logs_path=params['bloom_logs_path'],
                                             gradient_id=params['gradient_id'],
                                             verbosity_frequency=params['bloom_verbosity_frequency'],
                                             verbosity=params['bloom_verbosity'],
                                             rank=rank())
        ctx = tensor_shape
        params['tensors_size_are_same'] = True
        return compressed_tensor, ctx

    @staticmethod
    def decompress(compressed_tensor, ctx, params):

        tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)

        filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
        library = load_library.load_op_library(filename)
        bloom_decompressor = library.bloom_decompressor

        decompressed_tensor = bloom_decompressor(compressed_tensor, tensor_size,
                                                 tf.train.get_or_create_global_step(),
                                                 policy=params['bloom_policy'],
                                                 mem_mode=params['mem_mode'],
                                                 hash_num=params['k'],
                                                 bloom_size=params['m'],
                                                 bloom_logs_path=params['bloom_logs_path'],
                                                 gradient_id=params['gradient_id'],
                                                 verbosity_frequency=params['bloom_verbosity_frequency'],
                                                 verbosity=params['bloom_verbosity'],
                                                 suffix=params['suffix'],
                                                 rank=rank())

        decompressed_tensor = tf.bitcast(decompressed_tensor, tf.float32)
        decompressed_tensor = tf.reshape(decompressed_tensor, tensor_shape)
        return decompressed_tensor


class Values_Approximation_Helper(Compressor):

    @staticmethod
    def double_exponential_fit(X_, Y_, K):

        # S, SS initialization
        Ysum = Y_ + tf.roll(Y_, shift=-1, axis=0)
        Xsum = tf.roll(X_, shift=-1, axis=0) - X_
        S = tf.tensor_scatter_nd_update(tf.roll(0.5 * Ysum * Xsum, shift=1, axis=0), [[0]], tf.zeros(1, tf.float64))
        S = tf.math.cumsum(S)
        Ssum = S + tf.roll(S, shift=-1, axis=0)
        SS = tf.tensor_scatter_nd_update(tf.roll(0.5 * Ssum * Xsum, shift=1, axis=0), [[0]], tf.zeros(1, tf.float64))
        SS = tf.math.cumsum(SS)

        sum_SSk_squared = tf.math.reduce_sum(tf.math.pow(SS, 2))
        sum_SSk_Sk = tf.math.reduce_sum(S * SS)
        sum_SSk_xk = tf.math.reduce_sum(SS * X_)
        sum_SSk = tf.math.reduce_sum(SS)
        sum_Sk_squared = tf.math.reduce_sum(tf.math.pow(S, 2))
        sum_Sk_xk = tf.math.reduce_sum(S * X_)
        sum_Sk = tf.math.reduce_sum(S)
        sum_data_x = tf.cast(K * (K + 1) / 2, tf.float64)
        sum_data_x_squared = tf.cast(K * (K + 1) * (2 * K + 1) / 6, tf.float64)
        K = tf.cast(K, tf.float64)

        # Form the first system
        values = tf.stack([sum_SSk_squared, sum_Sk_squared, sum_data_x_squared, K,
                            sum_SSk_Sk, sum_SSk_xk, sum_SSk, sum_Sk_xk, sum_Sk, sum_data_x], axis=0)

        A_LS_1 = tf.scatter_nd([[0, 0], [1, 1], [2, 2], [3, 3],
                                [0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3],
                                [2, 3]],
                               values, [4, 4])
        A_LS_1 = tf.tensor_scatter_nd_update(A_LS_1,
                                             [[0, 0], [1, 1], [2, 2], [3, 3],
                                              [1, 0], [2, 0], [3, 0],
                                              [2, 1], [3, 1],
                                              [3, 2]],
                                             values)

        a = tf.math.reduce_sum(tf.transpose(SS) * Y_)
        b = tf.math.reduce_sum(tf.transpose(S) * Y_)
        c = tf.math.reduce_sum(tf.transpose(X_) * Y_)
        d = tf.math.reduce_sum(Y_)

        b_vector_1 = tf.stack([a, b, c, d], axis=0)
        b_vector_1 = tf.reshape(b_vector_1, [4, 1])

        # Solve the first system
        Coefficient_vector_1 = tf.linalg.solve(A_LS_1, b_vector_1)

        # Calculate p1 and q1
        p1 = 0.5 * (Coefficient_vector_1[1] + tf.math.sqrt(
            tf.math.pow(Coefficient_vector_1[1], 2) + 4 * Coefficient_vector_1[0]))
        q1 = 0.5 * (Coefficient_vector_1[1] - tf.math.sqrt(
            tf.math.pow(Coefficient_vector_1[1], 2) + 4 * Coefficient_vector_1[0]))

        beta_k = tf.math.exp(p1 * X_)
        eta_k = tf.math.exp(q1 * X_)

        sum_betak_square = tf.math.reduce_sum(tf.math.pow(beta_k, 2))
        sum_etak_square = tf.math.reduce_sum(tf.math.pow(eta_k, 2))
        sum_betak_etak = tf.math.reduce_sum(beta_k * eta_k)

        # Form the second system
        A_LS_2 = tf.stack([sum_betak_square, sum_betak_etak, sum_betak_etak, sum_etak_square], axis=0)
        A_LS_2 = tf.reshape(A_LS_2, [2, 2])
        a = tf.reshape(tf.math.reduce_sum(tf.transpose(beta_k) * Y_), [1, ])
        b = tf.reshape(tf.math.reduce_sum(tf.transpose(eta_k) * Y_), [1, ])
        b_vector_2 = tf.stack([a, b], axis=0)
        b_vector_2 = tf.reshape(b_vector_2, [2, 1])

        # Solve the second system
        Coefficient_vector_2 = tf.linalg.solve(A_LS_2, b_vector_2)

        # print("Coefficient_vector_1: \n", Coefficient_vector_1)
        # print("p1:\n", p1)
        # print("Coefficient_vector_2:\n", Coefficient_vector_2)
        # print("q1:\n", q1)
        return Coefficient_vector_2[0], Coefficient_vector_2[1], p1, q1

    @staticmethod
    def logit_basis(X, a, N):  # log(p/(1-p))
        return tf.cast(a * tf.math.log(X / ((N + 1) - X)), dtype=tf.float64)

    @staticmethod
    def exp_basis(X, b, c):
        return tf.cast(b * tf.math.exp(c * X), dtype=tf.float64)

    @staticmethod
    def polynomial_basis(X, a):
        return tf.cast(tf.pow(X, a), dtype=tf.float64)

    @staticmethod
    def GetInputMatrix_Polynomial(xcol, x):
        N = tf.size(x)
        Xtrans = tf.ones([1, N], tf.float64)
        for i in range(1, xcol):
            basis = Values_Approximation_Helper.polynomial_basis(x, i)
            Xtrans = tf.concat([Xtrans, basis], axis=0)
        return tf.transpose(Xtrans)

    @staticmethod
    def find_breaks(y_train, num_of_segments, N):
        b = tf.constant(0, shape=(1,), dtype=tf.int32)
        N = tf.constant(N, shape=(1,), dtype=tf.int32)
        y = y_train
        break_points = tf.zeros(num_of_segments + 1, tf.int32)
        for i in range(num_of_segments - 1):
            a = tf.math.argmax(tf.abs(tf.linspace(y[0], y[-1], tf.size(y)) - y))
            b = b + tf.cast(a, tf.int32)
            break_points = tf.tensor_scatter_nd_update(break_points, [[i + 1]], b)
            y = tf.gather(y_train, tf.range(b[0], N[0]))
        break_points = tf.tensor_scatter_nd_update(break_points, [[num_of_segments]], N)
        sizes = [break_points[i + 1] - break_points[i] for i in range(num_of_segments)]
        return break_points, sizes

    @staticmethod
    def get_breaks(model, N):
        if model == "resnet20_v2":
            breaks = {
                432 : [0, 353, 432],
                2304 : [0, 1847, 2229, 2304],
                4608 : [0, 4073, 4544, 4608],
                9216 : [0, 8164, 9012, 9216],
                18432 : [0, 16094, 18060, 18432],
                36864 : [0, 33742, 36595, 36864]}
        elif model == "vgg16":
            breaks = {
                1728 : [0, 1443, 1663, 1728],
                36864 : [0, 34097, 36467, 36815, 36864],
                73728 : [0, 67595, 73032, 73630, 73728],
                147456 : [0, 132193, 145286, 147125, 147456],
                294912 : [0, 272485, 292623, 294580, 294844, 294912],
                589824 : [0, 553577, 586620, 589431, 589764, 589824],
                1179648 : [0, 1099105, 1172811, 1179005, 1179543, 1179648],
                2359296 : [0, 2195844, 2343594, 2357633, 2359102, 2359296]}
        elif model == "resnet50":
            breaks = {
                4096 : [0, 3656, 4018, 4096],
                9408 : [0, 8476, 9165, 9408],
                16384 : [0, 14406, 16145, 16327, 16384],
                36864 : [0, 32238, 36292, 36726, 36864],
                131072 : [0, 121069, 130381, 130989, 131072],
                32768 : [0, 29429, 32320, 32692, 32768],
                147456 : [0, 133258, 145944, 147255, 147456],
                65536 : [0, 58690, 64507, 65371, 65536],
                524288 : [0, 494762, 522078, 524067, 524238, 524288],
                589824 : [0, 539407, 584654, 589214, 589738, 589824],
                262144 : [0, 237433, 259437, 261782, 262062, 262144],
                2097152 : [0, 1990620, 2088919, 2096322, 2097036, 2097152],
                2359296 : [0, 2188168, 2341896, 2356580, 2358793, 2359296],
                1048576 : [0, 981145, 1041707, 1047784, 1048461, 1048576],
                2050048 : [0, 1980923, 2044274, 2049225, 2049929, 2050048]}
        return breaks[N]

    @staticmethod
    def GetInputMatrix(x, p0, N):
        Xtrans = tf.ones([1, N], tf.float64)  # [np.ones(N)] #{1}
        for [a, b, c] in p0:
            basis = Values_Approximation_Logit_Compressor.logit_basis(x, a, N)
            Xtrans = tf.concat([Xtrans, basis], axis=0)
            basis = Values_Approximation_Logit_Compressor.exp_basis(x, b, c)
            Xtrans = tf.concat([Xtrans, basis], axis=0)
        return tf.transpose(Xtrans)

    @staticmethod
    def LeastSquares(X, y):  # returns (X'X)^-1 X'y
        Xtrans = tf.transpose(X)
        tmp = tf.matmul(Xtrans, X)
        inverse = tf.linalg.inv(tmp)
        theta_estimates = tf.matmul(tf.matmul(inverse, Xtrans), y)
        return theta_estimates

    @staticmethod
    def is_convolutional(model, N):
        print(model)
        if model == "resnet20_v2":
            conv_sizes = [432, 2304, 4608, 9216, 18432, 36864]
        elif model == "vgg16":
            conv_sizes = [1728, 36864, 73728, 147456, 294912, 589824, 1179648, 2359296]
        elif model == "resnet50":
            conv_sizes = [4096, 9408, 16384, 36864, 131072, 32768, 147456, 65536, 524288,
                          589824, 262144, 2097152, 2359296, 1048576, 2050048]
        if N in conv_sizes:
            return True
        return False

    @staticmethod
    def get_num_of_segments(model, N):
        if model == "resnet20_v2":
            segments = {432:2, 2304:3, 4608:3, 9216:3, 18432:3, 36864:3}
        elif model == "vgg16":
            segments = {1728:3, 36864:4, 73728:4, 147456:4, 294912:5, 589824:5, 1179648:5, 2359296:5}
        elif model == "resnet50":
            segments = {  4096:3, 9408:3, 16384:4, 36864:4, 131072:4, 32768:4, 147456:4, 65536:4, 524288:5,
                          589824:5, 262144:5, 2097152:5, 2359296:5, 1048576:5, 2050048:5}
        return segments[N]

class Values_Approximation_Compressor(Compressor):

    @staticmethod
    def compress(tensor, params):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        N = tensor_flatten.get_shape().as_list()[0]
        params['N'] = int(N)
        print("Tensor", tensor, "size:", params['N'])

        # Values Approximation
        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):
            abs_values = tf.math.abs(tensor_flatten)
            mapping = tf.argsort(abs_values, axis=0, direction='ASCENDING')
            values = tf.gather(abs_values, mapping)

            #  Indices have a negative sign if they correspond to negative values and positive otherwise
            negative_indices = tf.where(tf.less(tf.gather(tensor_flatten, mapping), 0))
            X = tf.cast(tf.range(1, N+1), tf.float64)
            Nneg = tf.size(negative_indices)
            mask = tf.tensor_scatter_nd_update(tf.ones([N], dtype=tf.int32), negative_indices, -tf.ones(Nneg, dtype=tf.int32))
            mapping = (mapping + 1) * mask

            # Fitting the curve
            coefficients = Values_Approximation_Helper.double_exponential_fit(X, tf.cast(values, tf.float64), N)
            num_of_coefficients = len(coefficients)     # coefficients = tf.cast(coefficients, tf.float32)

            ##################### Logging #####################
            filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            library = load_library.load_op_library(filename)
            logger = library.logger
            logger = logger(tensor_flatten, coefficients, tf.train.get_or_create_global_step(),
                            bloom_logs_path=params['bloom_logs_path'],
                            gradient_id=params['gradient_id'],
                            verbosity_frequency=params['bloom_verbosity_frequency'],
                            verbosity=params['bloom_verbosity'],
                            rank=rank())
            ##################### / Logging #####################

            compressed_indices = mapping            # Possible indices compression
            with tf.control_dependencies([logger]):
                coefficients = tf.reshape(coefficients, [-1])
                compressed_indices = tf.cast(compressed_indices, tf.float64)
                tensor_compressed = tf.concat([coefficients, compressed_indices], 0)
                params['message_size'] = num_of_coefficients
                params['X_train'] = X

        else:       # No approximation
            tensor_compressed = tensor

        ctx = tensor_shape
        params['tensors_size_are_same'] = True
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx

        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):
            message, indices = tf.split(tensor_compressed, [params['message_size'], params['N']])
            decompressed_indices = tf.cast(indices, tf.int32)
            negative_indices = tf.where(tf.less(decompressed_indices, 0))
            decompressed_indices = tf.math.abs(decompressed_indices)
            decompressed_indices = decompressed_indices - 1

            y_estimates = message[0] * tf.math.exp(message[2] * params['X_train']) + \
                          message[1] * tf.math.exp(message[3] * params['X_train'])

            Nneg = tf.size(negative_indices)
            mask = tf.tensor_scatter_nd_update(tf.ones([params['N']], dtype=tf.int32), negative_indices, -tf.ones(Nneg, dtype=tf.int32))
            y_estimates = y_estimates * tf.cast(mask, tf.float64)
            values = tf.reshape(y_estimates, [-1])
            decompressed_indices = tf.expand_dims(decompressed_indices, 1)
            tensor_decompressed = tf.scatter_nd(decompressed_indices, tf.cast(values, tf.float32), [params['N']])
            tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        else:
            tensor_decompressed = tensor_compressed

        return tensor_decompressed

class Polynomial_Segmented_Values_Approximation_Compressor(Compressor):

    @staticmethod
    def compress(tensor, params):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        N = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]
        params['N'] = params['K'] = int(N)

        print("Tensor", tensor, "size:", params['N'])

        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):

            abs_values = tf.math.abs(tensor_flatten)

            if params['approximation_mode'] == "topk":
                K = max(1, int(N * compress_ratio))
                params['K'] = K
                top_values, mapping = tf.math.top_k(abs_values, K, sorted=False)
                sorted_mapping = tf.argsort(top_values, axis=0, direction='ASCENDING')
                values = tf.gather(top_values, sorted_mapping)
                mapping = tf.gather(mapping, sorted_mapping)
            else:
                mapping = tf.argsort(abs_values, axis=0, direction='ASCENDING')
                values = tf.gather(abs_values, mapping)

            # Indices have a negative sign if they correspond to negative values and positive otherwise
            negative_indices = tf.where(tf.less(tf.gather(tensor_flatten, mapping), 0))
            Nneg = tf.size(negative_indices)
            mask = tf.tensor_scatter_nd_update(tf.ones([params['K']], dtype=tf.int32), negative_indices,
                                               -tf.ones(Nneg, dtype=tf.int32))
            mapping = (mapping + 1) * mask

            # Fitting the curve segments
            params['num_of_segments'] = Values_Approximation_Helper.get_num_of_segments(params['model_name'], params['N'])
            break_points, sizes = Values_Approximation_Helper.find_breaks(values, params['num_of_segments'], params['K'])
            # break_points = Values_Approximation_Helper.get_breaks(params['model_name'], N)
            # sizes = [break_points[i+1]-break_points[i] for i in range(params['num_of_segments'])]

            # params['X'] = {}
            coefficients = []
            for i in range(params['num_of_segments']):
                x = tf.reshape(tf.cast(tf.range(0, sizes[i]), tf.float64), [1, sizes[i]])
                X = Values_Approximation_Helper.GetInputMatrix_Polynomial(params['polynomial_degree'], x)
                y = tf.reshape(values[break_points[i]: break_points[i+1]], [sizes[i], 1])
                coefficients += [Values_Approximation_Helper.LeastSquares(X, tf.cast(y, tf.float64))]
                # params['X'][i] = X
            coefficients = tf.convert_to_tensor(coefficients)
            coefficients = tf.reshape(coefficients, [-1])

            ##################### Logging #####################
            filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            library = load_library.load_op_library(filename)
            logger = library.logger
            logger = logger(tensor_flatten, coefficients, tf.train.get_or_create_global_step(),
                            bloom_logs_path=params['bloom_logs_path'],
                            gradient_id=params['gradient_id'],
                            verbosity_frequency=params['bloom_verbosity_frequency'],
                            verbosity=params['bloom_verbosity'],
                            rank=rank())
            ##################### / Logging #####################

            compressed_indices = mapping  # Possible indices compression
            with tf.control_dependencies([logger]):
                sizes = tf.cast(sizes, tf.float64)
                compressed_indices = tf.cast(compressed_indices, tf.float64)
                tensor_compressed = tf.concat([sizes, coefficients, compressed_indices], 0)
                # tensor_compressed = tf.concat([coefficients, compressed_indices], 0)
        else:
            tensor_compressed = tensor

        ctx = tensor_shape
        params['tensors_size_are_same'] = True
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx

        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):
            sizes, coefficients, indices = tf.split(tensor_compressed, [params['num_of_segments'],
                                                                        params['polynomial_degree'] * params[
                                                                            'num_of_segments'],
                                                                        params['N']])
            # coefficients, indices = tf.split(tensor_compressed, [params['polynomial_degree'] *
            #                                                             params['num_of_segments'],
            #                                                             params['N']])
            coefficients = tf.reshape(coefficients, [params['num_of_segments'], params['polynomial_degree']])
            decompressed_indices = tf.cast(indices, tf.int32)
            sizes = tf.cast(sizes, tf.int32)
            negative_indices = tf.where(tf.less(decompressed_indices, 0))
            decompressed_indices = tf.math.abs(decompressed_indices)
            decompressed_indices = decompressed_indices - 1
            Nneg = tf.size(negative_indices)
            mask = tf.tensor_scatter_nd_update(tf.ones([params['K']], dtype=tf.int32), negative_indices,
                                               -tf.ones(Nneg, dtype=tf.int32))
            y_segments = []
            for i in range(params['num_of_segments']):
                x = tf.reshape(tf.cast(tf.range(0, sizes[i]), tf.float64), [1, sizes[i]])
                X = Values_Approximation_Helper.GetInputMatrix_Polynomial(params['polynomial_degree'], x)
                # X = params['X'][i]
                y_segments += [tf.matmul(X, tf.reshape(coefficients[i], [params['polynomial_degree'], 1]))]
            values = tf.reshape(tf.concat(y_segments, axis=0), [params['N']])
            values = values * tf.cast(mask, tf.float64)

            decompressed_indices = tf.expand_dims(decompressed_indices, 1)
            tensor_decompressed = tf.scatter_nd(decompressed_indices, tf.cast(values, tf.float32), [params['N']])
            tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        else:
            tensor_decompressed = tensor_compressed

        return tensor_decompressed


class TopK_Values_Approximation_Compressor(Compressor):

    @staticmethod
    def compress(tensor, params):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        N = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]
        K = max(1, int(N*compress_ratio))     # If compress ratio is set to 1 then K=N
        params['N'] = int(N) ; params['K'] = K
        print("Tensor", tensor, "size:", params['N'])

        abs_values = tf.math.abs(tensor_flatten)

        # Values Approximation
        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):
            top_values, mapping = tf.math.top_k(abs_values, K, sorted=False)
            sorted_mapping = tf.argsort(top_values, axis=0, direction='ASCENDING')
            values = tf.gather(top_values, sorted_mapping)
            mapping = tf.gather(mapping, sorted_mapping)

            # Indices have a negative sign if they correspond to negative values and positive otherwise
            negative_indices = tf.where(tf.less(tf.gather(tensor_flatten, mapping), 0))
            X = np.array(range(1, K+1), np.float64)
            Kneg = tf.size(negative_indices)
            mask = tf.tensor_scatter_nd_update(tf.ones([K], dtype=tf.int32), negative_indices, -tf.ones(Kneg, dtype=tf.int32))
            mapping = (mapping + 1) * mask

            # Fitting the curve
            coefficients = Values_Approximation_Helper.double_exponential_fit(X, tf.cast(values, tf.float64), K)
            num_of_coefficients = len(coefficients)     # coefficients = tf.cast(coefficients, tf.float32)

            ##################### Logging #####################
            filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            library = load_library.load_op_library(filename)
            logger = library.logger
            logger = logger(tensor_flatten, coefficients, tf.train.get_or_create_global_step(),
                            bloom_logs_path=params['bloom_logs_path'],
                            gradient_id=params['gradient_id'],
                            verbosity_frequency=params['bloom_verbosity_frequency'],
                            verbosity=params['bloom_verbosity'],
                            rank=rank())
            ##################### /Logging #####################

            compressed_indices = mapping  # Possible indices compression here
            with tf.control_dependencies([logger]):
                coefficients = tf.reshape(coefficients, [-1])
                compressed_indices = tf.cast(compressed_indices, tf.float64)
                tensor_compressed = tf.concat([coefficients, compressed_indices], 0)
                params['message_size'] = num_of_coefficients
                params['X_train'] = X

        # No approximation
        else:
            _, mapping = tf.math.top_k(abs_values, K, sorted=False)
            values = tf.gather(tensor_flatten, mapping)
            compressed_indices = mapping  # Possible indices compression here
            compressed_indices = tf.cast(compressed_indices, tf.float32)
            tensor_compressed = tf.concat([values, compressed_indices], 0)
            params['message_size'] = K

        ctx = tensor_shape
        params['tensors_size_are_same'] = True
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx
        tensor_compressed_size = tf.math.reduce_prod(tf.shape(tensor_compressed))
        message, indices = tf.split(tensor_compressed, [params['message_size'], tensor_compressed_size-params['message_size']])
        decompressed_indices = tf.cast(indices, tf.int32)

        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):
            negative_indices = tf.where(tf.less(decompressed_indices, 0))
            decompressed_indices = tf.math.abs(decompressed_indices)
            decompressed_indices = decompressed_indices - 1

            y_estimates = message[0] * tf.math.exp(message[2] * params['X_train']) + \
                          message[1] * tf.math.exp(message[3] * params['X_train'])

            Kneg = tf.size(negative_indices)
            mask = tf.tensor_scatter_nd_update(tf.ones([params['K']], dtype=tf.int32), negative_indices, -tf.ones(Kneg, dtype=tf.int32))
            y_estimates = y_estimates * tf.cast(mask, tf.float64)
            values = tf.cast(tf.reshape(y_estimates, [-1]), tf.float32)
        else:
            values = message

        decompressed_indices = tf.expand_dims(decompressed_indices, 1)
        tensor_decompressed = tf.scatter_nd(decompressed_indices, values, [params['N']])
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)

        return tensor_decompressed


class Two_TopK_Values_Approximation_Compressor(Compressor):

    @staticmethod
    def compress(tensor, params):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        N = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]
        K = max(1, int(N*compress_ratio))     # If compress ratio is set to 1 then K=N
        params['N'] = int(N) ; params['K'] = K
        print("Tensor", tensor, "size:", params['N'])

        abs_values = tf.math.abs(tensor_flatten)

        # Values Approximation
        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):
            _, mapping = tf.math.top_k(abs_values, K, sorted=False)
            top_values = tf.gather(tensor_flatten, mapping)

            sorted_mapping = tf.argsort(top_values, axis=0, direction='ASCENDING')
            values = tf.gather(top_values, sorted_mapping)
            mapping = tf.gather(mapping, sorted_mapping)

            # Indices have a negative sign if they correspond to negative values and positive otherwise
            negative_indices = tf.where(tf.less(tf.gather(tensor_flatten, mapping), 0))
            Kneg = tf.size(negative_indices) ; Kpos = K-Kneg
            mask = tf.tensor_scatter_nd_update(tf.ones([K], dtype=tf.int32), negative_indices, -tf.ones(Kneg, dtype=tf.int32))
            mapping = (mapping + 1) * mask

            # Fitting the curve of Negatives
            Xneg = tf.cast(tf.range(1, Kneg + 1), tf.float64)
            neg_coefficients = Values_Approximation_Helper.double_exponential_fit(Xneg, tf.cast(tf.gather(values, tf.range(0, Kneg)), tf.float64), Kneg)
            neg_num_of_coefficients = len(neg_coefficients)
            neg_coefficients = tf.reshape(neg_coefficients, [-1])

            # Fitting the curve of Positives
            Xpos = tf.cast(tf.range(1, Kpos + 1), tf.float64)
            pos_coefficients = Values_Approximation_Helper.double_exponential_fit(Xpos, tf.cast(tf.gather(values, tf.range(K-Kpos, K)), tf.float64), Kpos)
            pos_num_of_coefficients = len(pos_coefficients)
            pos_coefficients = tf.reshape(pos_coefficients, [-1])

            ##################### Logging #####################
            coefficients = tf.concat([neg_coefficients, pos_coefficients], 0)
            filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            library = load_library.load_op_library(filename)
            logger = library.logger
            logger = logger(tensor_flatten, coefficients, tf.train.get_or_create_global_step(),
                            bloom_logs_path=params['bloom_logs_path'],
                            gradient_id=params['gradient_id'],
                            verbosity_frequency=params['bloom_verbosity_frequency'],
                            verbosity=params['bloom_verbosity'],
                            rank=rank())
            ##################### /Logging #####################

            compressed_indices = mapping  # Possible indices compression here
            with tf.control_dependencies([logger]):
                compressed_indices = tf.cast(compressed_indices, tf.float64)
                tensor_compressed = tf.concat([coefficients, compressed_indices], 0)
                params['message_size'] = neg_num_of_coefficients + pos_num_of_coefficients
                params['Xneg'] = Xneg
                params['Xpos'] = Xpos

        # No approximation
        else:
            _, mapping = tf.math.top_k(abs_values, K, sorted=False)
            values = tf.gather(tensor_flatten, mapping)
            compressed_indices = mapping  # Possible indices compression here
            compressed_indices = tf.cast(compressed_indices, tf.float32)
            tensor_compressed = tf.concat([values, compressed_indices], 0)
            params['message_size'] = K

        ctx = tensor_shape
        params['tensors_size_are_same'] = True
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx
        tensor_compressed_size = tf.math.reduce_prod(tf.shape(tensor_compressed))
        message, indices = tf.split(tensor_compressed, [params['message_size'], tensor_compressed_size-params['message_size']])
        decompressed_indices = tf.cast(indices, tf.int32)

        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):
            negative_indices = tf.where(tf.less(decompressed_indices, 0))
            decompressed_indices = tf.math.abs(decompressed_indices)
            decompressed_indices = decompressed_indices - 1

            message_neg, message_pos = tf.split(message, 2)

            y_estimates_neg = message_neg[0] * tf.math.exp(message_neg[2] * params['Xneg']) + \
                          message_neg[1] * tf.math.exp(message_neg[3] * params['Xneg'])

            y_estimates_pos = message_pos[0] * tf.math.exp(message_pos[2] * params['Xpos']) + \
                              message_pos[1] * tf.math.exp(message_pos[3] * params['Xpos'])

            y_estimates = tf.concat([y_estimates_neg, y_estimates_pos], 0)

            Kneg = tf.size(negative_indices)
            mask = tf.tensor_scatter_nd_update(tf.ones([params['K']], dtype=tf.int32), negative_indices, -tf.ones(Kneg, dtype=tf.int32))
            y_estimates = y_estimates * tf.cast(mask, tf.float64)
            values = tf.cast(tf.reshape(y_estimates, [-1]), tf.float32)
        else:
            values = message

        decompressed_indices = tf.expand_dims(decompressed_indices, 1)
        tensor_decompressed = tf.scatter_nd(decompressed_indices, values, [params['N']])
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)

        return tensor_decompressed


class Values_Approximation_Logit_Compressor(Compressor):

    @staticmethod
    def compress(tensor, params):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        N = tensor_flatten.get_shape().as_list()[0]
        params['N'] = int(N)

        print("Tensor", tensor, "size:", params['N'])
        # params["layers"].add_data(tensor, params['N'])

        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):

            # p0 = [[0.004, -0.01, -0.04]]
            p0 = [[0.001, 0.1, 0.00001]]
            num_of_coefficients = len(p0[0])
            sorted_indices = tf.argsort(tensor_flatten, axis=0, direction='ASCENDING')
            values_sorted = tf.gather(tensor_flatten, sorted_indices)
            values_sorted = tf.reshape(values_sorted, [N, 1])

            X = np.array(range(1, N + 1), np.float64).reshape([1, N])
            X_train = Values_Approximation_Helper.GetInputMatrix(X, p0, N)
            coefficients = Values_Approximation_Helper.LeastSquares(X_train, tf.cast(values_sorted, tf.float64))

            ##################### Logging #####################
            filename = resource_loader.get_path_to_datafile('mpi_lib.cpython-36m-x86_64-linux-gnu.so')
            library = load_library.load_op_library(filename)
            logger = library.logger
            logger = logger(tensor_flatten, coefficients, tf.train.get_or_create_global_step(),
                            bloom_logs_path=params['bloom_logs_path'],
                            gradient_id=params['gradient_id'],
                            verbosity_frequency=params['bloom_verbosity_frequency'],
                            verbosity=params['bloom_verbosity'],
                            rank=rank())
            ##################### / Logging #####################

            compressed_indices = sorted_indices
            with tf.control_dependencies([logger]):
                coefficients = tf.reshape(coefficients, [-1])
                compressed_indices = tf.cast(compressed_indices, tf.float64)
                tensor_compressed = tf.concat([coefficients, compressed_indices], 0)
                params['message_size'] = num_of_coefficients
                params['X_train'] = X_train

        else:
            tensor_compressed = tensor

        ctx = tensor_shape
        params['tensors_size_are_same'] = True
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx

        if Values_Approximation_Helper.is_convolutional(params['model_name'], params['N']):
            message, indices = tf.split(tensor_compressed, [params['message_size'], params['N']])
            decompressed_indices = tf.cast(indices, tf.int32)
            message = tf.expand_dims(message, 1)

            y_estimates = tf.matmul(params['X_train'], message)
            y_estimates = tf.reshape(y_estimates, [-1])
            values = tf.reshape(y_estimates, [-1])
            decompressed_indices = tf.expand_dims(decompressed_indices, 1)
            tensor_decompressed = tf.scatter_nd(decompressed_indices, tf.cast(values, tf.float32), [params['N']])
            tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        else:
            tensor_decompressed = tensor_compressed

        return tensor_decompressed

##########################################################################################




class ThresholdCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        threshold_val = params["threshold_val"]
        thr_mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), threshold_val)
        values = tf.boolean_mask(tensor_flatten, thr_mask)
        indices = tf.reshape(tf.where(thr_mask), [-1])
        ctx = tensor_shape
        values = tf.bitcast(values, tf.int32)
        indices = tf.cast(indices, dtype=tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        params['tensors_size_are_same'] = False
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32), trainable=False)
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.float32))
        with tf.control_dependencies([op]):
            tensor_decompressed = tf.scatter_update(zero_tensor, indices, values)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class SignSGDCompressor(Compressor):
    """"""

    @staticmethod
    def aggregate(tensors, params):
        """Aggregate a list of tensors."""
        agged_tensor = tf.math.add_n(tensors)
        agged_tensor = tf.cast(tf.math.greater_equal(agged_tensor, 0), dtype=tf.float32)
        agged_tensor = agged_tensor * 2.0 - 1.0
        return agged_tensor

    @staticmethod
    def compress(tensor, params):

        """Encoding and compressing the signs """
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        tensor_compressed = tf.math.greater_equal(tensor_flatten, 0)
        ctx = tensor_shape
        params['tensors_size_are_same'] = True
        return tensor_compressed, ctx

    @staticmethod
    def decompress(sign_encode, ctx, params):
        """Decoding the signs to float format """
        tensor_shape = ctx
        sign_decode = tf.cast(sign_encode, dtype=tf.float32) * 2.0 - 1.0
        tensor_decompressed = tf.reshape(sign_decode, tensor_shape)
        return tensor_decompressed


class EFSignSGDCompressor(Compressor):
    """"""
    residuals = {}

    @classmethod
    def memory_compensate(cls, tensor, params):
        """Update the tensor with the residuals."""
        name = tensor.name
        lr = params["learning_rate"]
        cls.residuals[tensor.name] = tf.Variable(tf.zeros_like(tensor), trainable=False)
        tensor = cls.residuals[name] + lr * tensor
        return tensor

    @classmethod
    def memory_update(cls, tensor, tensor_compensate, tensor_compressed, ctx, params):
        """Update the residuals."""
        name = tensor.name
        tensor_decompressed = cls.decompress(tensor_compressed, ctx, params)
        delta = tensor_compensate - tensor_decompressed
        memory_update_op = cls.residuals[name].assign(delta)
        return [memory_update_op]

    @staticmethod
    def aggregate(tensors, params):
        """Aggregate a list of tensors."""
        lr = params["learning_rate"]
        agged_tensor = tf.math.add_n(tensors)
        agged_tensor = agged_tensor / lr
        return agged_tensor

    @staticmethod
    def compress(tensor, params):

        """Encoding and compressing the signs """

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        sign_encode = tf.math.greater_equal(tensor_flatten, 0)
        mean = tf.math.reduce_mean(tf.abs(tensor_flatten))
        ctx = tensor_shape
        tensor_compressed = mean, sign_encode
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decoding the signs to float format """
        mean, sign_encode = tensor_compressed
        tensor_shape = ctx
        sign_decode = tf.cast(sign_encode, dtype=tf.float32) * 2.0 - 1.0
        sign_decode = mean * sign_decode
        tensor_decompressed = tf.reshape(sign_decode, tensor_shape)
        return tensor_decompressed


class SignumCompressor(Compressor):
    """"""
    momentum = {}

    @staticmethod
    def aggregate(tensors, params):
        """Aggregate a list of tensors."""
        agged_tensor = tf.math.add_n(tensors)
        agged_tensor = tf.cast(tf.math.greater_equal(agged_tensor, 0), dtype=tf.float32)
        agged_tensor = agged_tensor * 2.0 - 1.0
        return agged_tensor

    @staticmethod
    def compress(tensor, params):

        """Encoding and compressing the signs """

        # update tensor by momentum
        momentum = params["momentum"]
        name = tensor.name
        SignumCompressor.momentum[name] = tf.Variable(tf.zeros_like(tensor), trainable=False)
        tensor = (1.0 - momentum) * tensor + momentum * SignumCompressor.momentum[name]
        temp = SignumCompressor.momentum[name].assign(tensor)
        tensor = tensor + temp - temp

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        tensor_compressed = tf.math.greater_equal(tensor_flatten, 0)
        ctx = tensor_shape
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(sign_encode, ctx, params):
        """Decoding the signs to float format """
        tensor_shape = ctx
        sign_decode = tf.cast(sign_encode, dtype=tf.float32) * 2.0 - 1.0
        tensor_decompressed = tf.reshape(sign_decode, tensor_shape)
        return tensor_decompressed


class QsgdCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):

        def encode2bool(tensor, quantiles):
            tensor = tf.cast(tensor, dtype=tf.int32)
            bits = tf.cast(math.log(quantiles, 2) + 1, dtype=tf.int32)
            def cond(step, input_tensor, output):
                return step < bits

            def encode(step, input_tensor, output):
                base = tf.constant(2, tf.int32)
                temp = tf.floormod(input_tensor, base)
                output = output.write(step, temp)
                input_tensor = tf.floordiv(input_tensor, base)
                return step + 1, input_tensor, output

            step = tf.constant(0)
            output = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
            _, _, final_output = tf.while_loop(cond, encode, loop_vars=[step, tensor, output])
            encode_output = tf.cast(final_output.stack(), dtype=tf.bool)
            return encode_output

        quantum_num = params["quantum_num"]
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        norm = tf.reshape(tf.norm(tensor_flatten), [-1])
        abs_gradient = tf.abs(tensor_flatten)
        qnum = tf.cast(quantum_num, dtype=tf.float32)

        level_float = qnum / norm * abs_gradient
        previous_level = tf.math.floor(level_float)
        prob = tf.random.uniform(tf.shape(tensor_flatten))
        is_next_level = tf.cast(tf.math.less(prob, (level_float - previous_level)), tf.float32)
        new_level = tf.cast(previous_level + is_next_level, tf.float32)
        #new_level = tf.cast(previous_level + is_next_level, tf.int32)
        #encode_output = encode2bool(new_level, quantum_num)
        #sign = tf.reshape(tf.greater_equal(tensor, 0), [1, -1])
        #encode_output = tf.concat([sign, encode_output], 0)
        sign = tf.sign(tensor_flatten)
        tensor_compressed = new_level * sign
        tensor_compressed = tf.cast(tensor_compressed, dtype=tf.int8 if quantum_num < 128 else tf.int16)
        tensor_compressed = tensor_compressed, norm
        ctx = tensor_shape
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):

        def decode4bool(tensor, quantiles):
            tensor = tf.cast(tensor, dtype=tf.int32)
            bits = tf.cast(math.log(quantiles, 2) + 1, dtype=tf.int32)
            def cond(step, input_tensor, output):
                return step < bits

            def decode(step, input_tensor, output):
                base = tf.constant(2, tf.int32)
                temp = input_tensor[step, :]
                output = output + temp * tf.math.pow(base, step)
                return step + 1, input_tensor, output
            output = tf.zeros([tf.shape(tensor)[1]], dtype=tf.int32)
            step = tf.constant(0)
            _, _, decode_output = tf.while_loop(cond, decode, loop_vars=[step, tensor, output])
            return decode_output
        quantum_num = params["quantum_num"]
        qnum = tf.cast(quantum_num, dtype=tf.float32)
        tensor_shape = ctx
        tensor_compressed, norm = tensor_compressed

        #encode_output = tf.cast(encode_output, dtype=tf.int32)
        #sign = encode_output[0, :] * 2 - 1
        #input_tensor = encode_output[1:, :]
        #decode_output = decode4bool(input_tensor, quantum_num)
        #decode_output = sign * decode_output
        #decode_output = tf.cast(decode_output, dtype=tf.float32)

        decode_output = tf.cast(tensor_compressed, dtype=tf.float32)
        tensor_decompressed = norm / qnum * decode_output
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class OnebitCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        mask0 = tf.math.less(tensor_flatten, 0)
        sum0 = tf.math.reduce_sum(tf.boolean_mask(tensor_flatten, mask0))
        num0 = tf.math.reduce_sum(tf.cast(mask0, dtype=tf.float32))
        num0 = tf.where(tf.math.greater(num0, 0), num0, 1.0)
        mean0 = sum0 / num0

        mask1 = tf.math.logical_not(mask0)
        sum1 = tf.math.reduce_sum(tf.boolean_mask(tensor_flatten, mask1))
        num1 = tf.math.reduce_sum(tf.cast(mask1, dtype=tf.float32))
        num1 = tf.where(tf.math.greater(num1, 0), num1, 1.0)
        mean1 = sum1 / num1

        mean0 = tf.reshape(mean0, [-1])
        mean1 = tf.reshape(mean1, [-1])
        mean = tf.concat([mean0, mean1], 0)
        ctx = tensor_shape
        tensor_compressed = mask0, mean
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx
        mask0, mean = tensor_compressed
        mean0, mean1 = tf.split(mean, 2)
        mask0 = tf.cast(mask0, dtype=tf.float32)
        tensor_decompressed = mask0 * mean0 + (1-mask0) * mean1
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class TerngradCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        std = tf.math.square(tensor_flatten - tf.math.reduce_mean(tensor_flatten))
        std = tf.math.sqrt(tf.math.reduce_mean(std))
        c = 2.5
        gradient = tf.clip_by_value(tensor_flatten, -c * std, c * std)
        scaler = tf.math.reduce_max(tf.math.abs(gradient))

        zeros = tf.zeros(tf.shape(tensor_flatten))
        abs_gradient = tf.abs(gradient)
        sign_gradient = tf.sign(gradient)
        rnd_sample = tf.random_uniform(tf.shape(tensor_flatten), 0, scaler)
        where_cond = tf.less(rnd_sample, abs_gradient)
        binarized_gradient = tf.where(where_cond, sign_gradient * scaler, zeros)
        new_sign = tf.sign(binarized_gradient)  # -1, 0, 1

        """
        a = tf.add(new_sign, 1)  # shift -1,0,1 to 0,1,2 (2'b00,2'b01,2'b10)
        a = tf.reshape(a, [-1])
        pad_size = 4 - tf.mod(tf.size(a), 4)
        pad = tf.range(0.0, pad_size)
        a = tf.concat([a, pad], 0)
        a_split1, a_split2, a_split3, a_split4 = tf.split(a, 4)  # assume the size is dividable by 4

        # encode 4 grads into 1 Byte
        sum_1 = tf.add(a_split1, a_split2 * 4)
        sum_2 = tf.add(a_split3 * 16, a_split4 * 64)
        sum_all = tf.add(sum_1, sum_2)

        tensor_compressed = tf.cast(sum_all, tf.uint8)
        """


        scaler = tf.reshape(scaler, [-1])
        ctx = tensor_shape
        tensor_compressed = tf.cast(new_sign, tf.int8), scaler
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):

        tensor_shape = ctx
        tensor_compressed, scaler = tensor_compressed
        """
        a = tf.cast(tensor_compressed, tf.int32)
        a_split1 = tf.mod(a, 4)
        a_split2 = tf.to_int32(tf.mod(a / 4, 4))
        a_split3 = tf.to_int32(tf.mod(a / 16, 4))
        a_split4 = tf.to_int32(tf.mod(a / 64, 4))
        a = tf.concat([a_split1, a_split2, a_split3, a_split4], 0)
        real_size = tf.reduce_prod(tensor_shape)
        a = tf.to_float(a)
        a = tf.gather(a, tf.range(0, real_size))

        a = tf.reshape(a, tensor_shape)
        sign = tf.subtract(a, 1)
        """
        sign = tf.cast(tensor_compressed, dtype=tf.float32)
        tensor_decompressed = sign * scaler
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class DgcCompressor(Compressor):
    """"""

    residuals = {}
    gradients = {}
    @classmethod
    def memory_compensate(cls, tensor, params):
        """Update the tensor with the residuals."""
        name = tensor.name

        horovod_size = params["horovod_size"]
        momentum = params["momentum"]
        gradient_clipping = params["gradient_clipping"]
        if gradient_clipping:
            tensor_squ_sum = tf.math.reduce_sum(tf.math.square(tensor))
            # if params['debug']:
            #     tensor_squ_sum = tf.Print(tensor_squ_sum, [tf.size(tensor_squ_sum)],
            #                               message=f"==Debug== tensor 0/1 on rank {rank()} {tensor_squ_sum.dtype} size:")
            thr_global = tf.math.sqrt(_allreduce(tensor_squ_sum))
            clipping_val = thr_global / tf.math.sqrt(float(horovod_size))
            tensor = tf.clip_by_value(tensor, -clipping_val, clipping_val)

        cls.residuals[name] = tf.Variable(tf.zeros_like(tensor), trainable=False)
        cls.gradients[name] = tf.Variable(tf.zeros_like(tensor), trainable=False)
        u = cls.residuals[name].assign(momentum * cls.residuals[name] + tensor)
        tensor_compensate = cls.gradients[name].assign(cls.gradients[name] + u) + tf.zeros_like(tensor)
        return tensor_compensate

    @classmethod
    def memory_update(cls, tensor, tensor_compensate, tensor_compressed, ctx, params):
        """Update the residuals."""
        name = tensor.name
        _, mask = ctx
        not_mask = tf.cast(tf.math.logical_not(mask), tf.float32)
        #not_mask = tf.Print(not_mask, ['not_mask2', tf.math.reduce_sum(not_mask)])
        not_mask = tf.reshape(not_mask, tf.shape(tensor))
        op1 = cls.residuals[name].assign(cls.residuals[name] * not_mask)
        op2 = cls.gradients[name].assign(cls.gradients[name] * not_mask)
        return [op1, op2]

    @staticmethod
    def compress(tensor, params):
        compress_ratio = params["compress_ratio"]
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]

        sample_shape = tf.reshape(tf.convert_to_tensor(max(1, int(elemnum * 0.01)), dtype=tf.int32), [-1])
        sample_index = tf.random.uniform(sample_shape, minval=0, maxval=elemnum, dtype=tf.int32)
        sample_tensor = tf.gather(tensor_flatten, sample_index)

        k = max(1, int(elemnum * compress_ratio * 0.01))
        vals, indices = tf.math.top_k(tf.math.abs(sample_tensor), k)
        thr = tf.math.reduce_min(vals)
        mask = tf.math.greater(tf.math.abs(tensor_flatten), thr)

        selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
        #selected = tf.Print(selected, ['selected:', selected])
        def body(thr, mask, selected):
            thr = tf.cond(selected > 1.25 * max(1, int(elemnum * compress_ratio)), lambda: 1.25 * thr, lambda: 0.9 * thr)
            mask = tf.math.greater(tf.math.abs(tensor_flatten), thr)
            selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
            return thr, mask, selected

        def condition(thr, mask, selected):
            cond_a = selected > 1.25 * max(1, int(elemnum * compress_ratio))
            cond_b = selected < 0.8 * max(1, int(elemnum * compress_ratio))
            return tf.math.logical_or(cond_a, cond_b)

        thr, mask, new_selected = tf.while_loop(condition, body, (thr, mask, selected), maximum_iterations=20)

        thr = tf.cond(new_selected < 1, lambda: 0.8 * thr, lambda: thr)
        # mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), thr)
        mask = tf.math.greater(tf.math.abs(tensor_flatten), thr) # fix the dgc NCF data volume issue

        indices = tf.reshape(tf.where(mask), [-1])
        #indices = tf.Print(indices, ["selected:", selected, new_selected, 'size changes:', tf.size(indices), tf.size(tensor), "ratio:", compress_ratio])
        values = tf.gather(tensor_flatten, indices)

        values = tf.bitcast(values, tf.int32)
        indices = tf.cast(indices, dtype=tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        ctx = tensor_shape, mask
        params['tensors_size_are_same'] = False

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape,_ = ctx
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32), trainable=False)
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.float32))
        with tf.control_dependencies([op]):
            tensor_decompressed = tf.scatter_update(zero_tensor, indices, values)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class AdaqCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):
        compress_ratio = params["compress_ratio"]

        def quan(tensor, tensor_mask, compress_ratio):
            # tensor_mask = tf.math.greater_equal(tf.math.abs(tensor),0) # for testing and debuging
            tensor_value = tf.boolean_mask(tensor, tensor_mask)
            mask_size = tf.reduce_sum(tf.cast(tensor_mask, dtype=tf.int32))
            sample_size = tf.cast(tf.reshape((tf.math.ceil(tf.cast(mask_size, dtype=tf.float32) * 0.01)), [-1]),
                                  dtype=tf.int32)
            sample_index = tf.random.uniform(sample_size, minval=0, maxval=mask_size, dtype=tf.int32)
            sample_tensor = tf.gather(tensor_value, sample_index)

            k = tf.cast((tf.math.ceil(tf.cast(mask_size, dtype=tf.float32) * 0.01 * compress_ratio)),
                        dtype=tf.int32)
            vals, indices = tf.math.top_k(tf.math.abs(sample_tensor), k)
            thr = tf.math.reduce_min(vals)
            tensor_masked = tf.cast(tensor_mask, dtype=tf.float32) * tensor
            mask = tf.math.greater(tf.math.abs(tensor_masked), thr)

            # fix the issue of sampling in topk
            selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
            elemnum = tf.cast(mask_size, dtype=tf.float32)

            def body(thr, mask, selected):
                thr = tf.cond(selected > 1.25 * tf.ceil(elemnum * compress_ratio), lambda: 1.25 * thr, lambda: 0.9 * thr)
                mask = tf.math.greater_equal(tf.math.abs(tensor_masked), thr)
                selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
                return thr, mask, selected

            def condition(thr, mask, selected):
                cond_a = selected > 1.25 * tf.ceil(elemnum * compress_ratio)
                cond_b = selected < 0.8 * tf.ceil(elemnum * compress_ratio)
                return tf.math.logical_or(cond_a, cond_b)

            thr2, mask2, selected2 = tf.while_loop(condition, body, (thr, mask, selected), maximum_iterations=20)
            thr2 = tf.cond(selected2 < 1, lambda: 0.8 * thr2, lambda: thr2)
            mask2 = tf.math.greater(tf.math.abs(tensor_masked), thr2)

            indices = tf.reshape(tf.where(mask2), [-1])
            #indices = tf.Print(indices, ["selected:", selected2, 'size changes:', tf.size(indices), elemnum, "ratio:", compress_ratio])
            tensor_value2 = tf.boolean_mask(tensor_masked, mask2)
            mean = tf.reshape(tf.math.reduce_mean(tensor_value2), [-1])

            return indices, mean, mask2

        tensor_shape = tf.shape(tensor)
        tensor_size = tf.size(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        plus_mask = tf.math.greater(tensor_flatten, 0)
        minus_mask = tf.math.less(tensor_flatten, 0)
        plus_indices, plus_mean, plus_mask = quan(tensor_flatten, plus_mask, compress_ratio)
        minus_indices, minus_mean, minus_mask = quan(tensor_flatten, minus_mask, compress_ratio)

        plus_mean = tf.bitcast(plus_mean, tf.int32)
        plus_indices = tf.reshape(tf.cast(plus_indices, dtype=tf.int32), [-1])
        minus_mean = tf.bitcast(minus_mean, tf.int32)
        minus_indices = tf.reshape(tf.cast(minus_indices, dtype=tf.int32), [-1])
        plus_indices_size = tf.reshape(tf.size(plus_indices), [-1])
        # minus_indices_size = tf.reshape(tf.size(minus_indices), [-1])
        tensor_compressed = tf.concat([plus_mean, minus_mean, plus_indices_size, plus_indices, minus_indices], 0)
        ctx = tensor_shape, tensor_size
        params['tensors_size_are_same'] = False
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        plus_mean = tensor_compressed[0]
        minus_mean = tensor_compressed[1]
        plus_indices_size = tensor_compressed[2]
        plus_indices = tensor_compressed[3:3 + plus_indices_size]
        minus_indices = tensor_compressed[3 + plus_indices_size:]

        plus_mean = tf.bitcast(plus_mean, tf.float32)
        minus_mean = tf.bitcast(minus_mean, tf.float32)
        tensor_shape, tensor_size = ctx

        zero_tensor = tf.Variable(tf.zeros([tensor_size]), trainable=False)  # solve the error 'Tensor' object has no attribute '_lazy_read'
        plus_mean = tf.ones(tf.shape(plus_indices), dtype=tf.float32) * plus_mean
        minus_mean = tf.ones(tf.shape(minus_indices), dtype=tf.float32) * minus_mean
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.float32))
        with tf.control_dependencies([op]):
            temp1 = tf.scatter_update(zero_tensor, plus_indices, plus_mean)
            tensor_decompressed = tf.scatter_update(temp1, minus_indices, minus_mean)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class AdapSparseCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):

        k = params["compress_ratio"]
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        tensor_size = tf.cast(tf.size(tensor), dtype=tensor.dtype)

        prob = k * tensor_size * tf.abs(tensor_flatten) / tf.math.reduce_sum(tf.abs(tensor_flatten))
        prob = tf.minimum(prob, 1.0)

        c = tf.constant(2.0)

        def body(c, prob):
            mask = tf.less(prob, 1.0)
            size_indices = tf.cast(tf.size(tf.where(mask)), dtype=tf.float32)
            sum_prob = tf.math.reduce_sum(tf.boolean_mask(prob, mask))
            c = ((k - 1.0) * tensor_size + size_indices) / sum_prob
            prob = tf.minimum(c * prob, 1.0)
            return c, prob

        def condition(c, prob):
            return tf.greater(c, 1.0)

        res = tf.while_loop(condition, body, (c, prob))
        prob = res[1]

        rnd_sample = tf.random_uniform(tf.shape(tensor_flatten))
        mask = tf.less(rnd_sample, prob)
        indices = tf.reshape(tf.where(mask), [-1])
        values = tf.gather(tensor_flatten / prob, indices)
        values = tf.bitcast(values, tf.int32)
        indices = tf.cast(indices, dtype=tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        ctx = tensor_shape
        params['tensors_size_are_same'] = False
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32), trainable=False)
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.float32))
        with tf.control_dependencies([op]):
            tensor_decompressed = tf.scatter_update(zero_tensor, indices, values)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class PowerSGDCompressor(Compressor):
    """"""
    momentum = {}
    q_memory = {}
    @classmethod
    def memory_compensate(cls, tensor, params):
        """Update the tensor with the residuals."""
        compress_rank = params["compress_rank"]
        tensor_name = params["tensor_name"]
        tensor_dims = params['tensor_dims']
        cls.momentum[tensor_name] = tf.Variable(tf.zeros_like(tensor), trainable=False)
        if tensor_dims == 1:
            return tensor

        cls.residuals[tensor_name] = tf.Variable(tf.zeros_like(tensor), trainable=False)
        tensor = cls.residuals[tensor_name] + tensor

        n = tensor.get_shape().as_list()[0]
        m = int(1)
        for dim in tensor.get_shape().as_list()[1:]:
            m = m * dim
        r = int(min([m, n, compress_rank]))
        cls.q_memory[tensor_name] = tf.Variable(tf.random.normal([m, r]), trainable=False)

        return tensor

    @classmethod
    def memory_update(cls, tensor, tensor_compensate, tensor_compressed, ctx, params):
        """Update the residuals."""
        tensor_name = params["tensor_name"]
        tensor_dims = params['tensor_dims']
        if tensor_dims == 1:
            return []

        new_tensor = cls.decompress(tensor, ctx, params)
        op = cls.residuals[tensor_name].assign(tensor_compensate - new_tensor)

        return [op]

    @classmethod
    def compress(cls, tensor, params):
        tensor_dims = params['tensor_dims']
        if tensor_dims == 1:
            return tensor, None

        horovod_size = params["horovod_size"]
        tensor_name = params["tensor_name"]
        tensor_shape = tf.shape(tensor)
        matrix = tf.reshape(tensor, [tensor_shape[0], -1])

        q = cls.q_memory[tensor_name]
        q, _ = tf.linalg.qr(q)

        p = tf.linalg.matmul(matrix, q)
        # if params['debug']:
        #     p = tf.Print(p, [tf.size(p)],
        #                  message=f"==Debug== tensor 0/1 on rank {rank()} {p.dtype} size:")
        p = _allreduce(p) / horovod_size
        p, _ = tf.linalg.qr(p)
        q = tf.linalg.matmul(matrix, p, transpose_a=True)
        # if params['debug']:
        #     q = tf.Print(q, [tf.size(q)],
        #                  message=f"==Debug== tensor 0/1 on rank {rank()} {q.dtype} size:")
        q = _allreduce(q) / horovod_size
        new_q = cls.q_memory[tensor_name].assign(q)
        ctx = p, new_q, tensor_shape

        return None, ctx

    @classmethod
    def decompress(cls, tensor, ctx, params):
        tensor_dims = params['tensor_dims']
        if tensor_dims == 1:
            return tensor

        p, q, tensor_shape = ctx
        new_tensor = tf.linalg.matmul(p, q, transpose_b=True)
        new_tensor = tf.reshape(new_tensor, tensor_shape)

        return new_tensor


class U8bitCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):

        dict128 = tf.constant(
            [
                1.5000001e-06, 2.7500000e-06, 7.2499997e-06, 1.8750001e-05,
                3.6250000e-05, 5.8749996e-05, 8.6249995e-05, 1.4375000e-04,
                2.3125000e-04, 3.1875001e-04, 4.0625001e-04, 5.1874999e-04,
                6.5624999e-04, 7.9374999e-04, 9.3124999e-04, 1.2187500e-03,
                1.6562500e-03, 2.0937501e-03, 2.5312500e-03, 2.9687500e-03,
                3.4062499e-03, 3.8437501e-03, 4.2812498e-03, 4.8437500e-03,
                5.5312500e-03, 6.2187500e-03, 6.9062500e-03, 7.5937500e-03,
                8.2812496e-03, 8.9687500e-03, 9.6562495e-03, 1.1093750e-02,
                1.3281250e-02, 1.5468750e-02, 1.7656250e-02, 1.9843750e-02,
                2.2031249e-02, 2.4218749e-02, 2.6406251e-02, 2.8593751e-02,
                3.0781250e-02, 3.2968748e-02, 3.5156250e-02, 3.7343752e-02,
                3.9531250e-02, 4.1718751e-02, 4.3906249e-02, 4.6718750e-02,
                5.0156251e-02, 5.3593751e-02, 5.7031251e-02, 6.0468748e-02,
                6.3906237e-02, 6.7343749e-02, 7.0781253e-02, 7.4218743e-02,
                7.7656247e-02, 8.1093743e-02, 8.4531240e-02, 8.7968737e-02,
                9.1406241e-02, 9.4843738e-02, 9.8281242e-02, 1.0546875e-01,
                1.1640625e-01, 1.2734374e-01, 1.3828126e-01, 1.4921875e-01,
                1.6015625e-01, 1.7109375e-01, 1.8203124e-01, 1.9296876e-01,
                2.0390625e-01, 2.1484375e-01, 2.2578125e-01, 2.3671874e-01,
                2.4765626e-01, 2.5859374e-01, 2.6953125e-01, 2.8046876e-01,
                2.9140624e-01, 3.0234376e-01, 3.1328124e-01, 3.2421875e-01,
                3.3515626e-01, 3.4609374e-01, 3.5703126e-01, 3.6796874e-01,
                3.7890625e-01, 3.8984376e-01, 4.0078124e-01, 4.1171876e-01,
                4.2265624e-01, 4.3359375e-01, 4.4453126e-01, 4.5859376e-01,
                4.7578123e-01, 4.9296874e-01, 5.1015621e-01, 5.2734375e-01,
                5.4453123e-01, 5.6171870e-01, 5.7890624e-01, 5.9609371e-01,
                6.1328125e-01, 6.3046873e-01, 6.4765620e-01, 6.6484374e-01,
                6.8203121e-01, 6.9921869e-01, 7.1640623e-01, 7.3359370e-01,
                7.5078118e-01, 7.6796871e-01, 7.8515619e-01, 8.0234367e-01,
                8.1953120e-01, 8.3671868e-01, 8.5390615e-01, 8.7109369e-01,
                8.8828117e-01, 9.0546864e-01, 9.2265618e-01, 9.3984365e-01,
                9.5703113e-01, 9.7421867e-01, 9.9140614e-01, 9.9570298e-01,
            ], dtype=tf.float32
        )

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        scaler = tf.math.reduce_max(tf.abs(tensor_flatten))
        new_tensor = tensor_flatten / scaler
        sign = tf.sign(tensor_flatten)
        new_tensor = tf.abs(new_tensor)
        """
        pivot = 64 * tf.ones([tensor_size], dtype=tf.int32)
        left = tf.zeros([tensor_size], dtype=tf.int32)
        right = 127 * tf.ones([tensor_size], dtype=tf.int32)

        step = 5

        def cond(step, input_tensor, pivot, left, right):
            return step > -1

        def body(step, input_tensor, pivot, left, right):
            base = tf.constant(2, tf.int32)
            vals_pivot = tf.gather(dict128, pivot)
            mask = tf.math.greater(input_tensor, vals_pivot)
            left = tf.where(mask, pivot, left)
            right = tf.where(mask, right, pivot)
            sign_mask = tf.cast(mask, dtype=tf.int32) * 2 - 1
            pivot = pivot + sign_mask * tf.math.pow(base, step)
            return step - 1, input_tensor, pivot, left, right

        step, _, pivot, left, right = tf.while_loop(cond, body, loop_vars=[step, new_tensor, pivot, left, right])
        tensor_compressed = tf.cast(pivot, dtype=tf.int8) * tf.cast(sign, dtype=tf.int8)
        """
        import tensorflow_probability as tfp
        edges = dict128
        bins = tf.cast(tfp.stats.find_bins(new_tensor, edges), dtype=tf.int8)

        scaler = tf.reshape(scaler, [-1])
        tensor_compressed = bins * tf.cast(sign, dtype=tf.int8), scaler
        ctx = tensor_shape
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx
        tensor, scaler = tensor_compressed
        dict128 = tf.constant(
            [
                1.5000001e-06, 2.7500000e-06, 7.2499997e-06, 1.8750001e-05,
                3.6250000e-05, 5.8749996e-05, 8.6249995e-05, 1.4375000e-04,
                2.3125000e-04, 3.1875001e-04, 4.0625001e-04, 5.1874999e-04,
                6.5624999e-04, 7.9374999e-04, 9.3124999e-04, 1.2187500e-03,
                1.6562500e-03, 2.0937501e-03, 2.5312500e-03, 2.9687500e-03,
                3.4062499e-03, 3.8437501e-03, 4.2812498e-03, 4.8437500e-03,
                5.5312500e-03, 6.2187500e-03, 6.9062500e-03, 7.5937500e-03,
                8.2812496e-03, 8.9687500e-03, 9.6562495e-03, 1.1093750e-02,
                1.3281250e-02, 1.5468750e-02, 1.7656250e-02, 1.9843750e-02,
                2.2031249e-02, 2.4218749e-02, 2.6406251e-02, 2.8593751e-02,
                3.0781250e-02, 3.2968748e-02, 3.5156250e-02, 3.7343752e-02,
                3.9531250e-02, 4.1718751e-02, 4.3906249e-02, 4.6718750e-02,
                5.0156251e-02, 5.3593751e-02, 5.7031251e-02, 6.0468748e-02,
                6.3906237e-02, 6.7343749e-02, 7.0781253e-02, 7.4218743e-02,
                7.7656247e-02, 8.1093743e-02, 8.4531240e-02, 8.7968737e-02,
                9.1406241e-02, 9.4843738e-02, 9.8281242e-02, 1.0546875e-01,
                1.1640625e-01, 1.2734374e-01, 1.3828126e-01, 1.4921875e-01,
                1.6015625e-01, 1.7109375e-01, 1.8203124e-01, 1.9296876e-01,
                2.0390625e-01, 2.1484375e-01, 2.2578125e-01, 2.3671874e-01,
                2.4765626e-01, 2.5859374e-01, 2.6953125e-01, 2.8046876e-01,
                2.9140624e-01, 3.0234376e-01, 3.1328124e-01, 3.2421875e-01,
                3.3515626e-01, 3.4609374e-01, 3.5703126e-01, 3.6796874e-01,
                3.7890625e-01, 3.8984376e-01, 4.0078124e-01, 4.1171876e-01,
                4.2265624e-01, 4.3359375e-01, 4.4453126e-01, 4.5859376e-01,
                4.7578123e-01, 4.9296874e-01, 5.1015621e-01, 5.2734375e-01,
                5.4453123e-01, 5.6171870e-01, 5.7890624e-01, 5.9609371e-01,
                6.1328125e-01, 6.3046873e-01, 6.4765620e-01, 6.6484374e-01,
                6.8203121e-01, 6.9921869e-01, 7.1640623e-01, 7.3359370e-01,
                7.5078118e-01, 7.6796871e-01, 7.8515619e-01, 8.0234367e-01,
                8.1953120e-01, 8.3671868e-01, 8.5390615e-01, 8.7109369e-01,
                8.8828117e-01, 9.0546864e-01, 9.2265618e-01, 9.3984365e-01,
                9.5703113e-01, 9.7421867e-01, 9.9140614e-01, 9.9570298e-01,
            ], dtype=tf.float32
        )
        # tensor is int8
        tensor = tf.cast(tensor, dtype=tf.int32)
        sign = tf.cast(tf.sign(tensor), dtype=tf.float32)
        index = tf.cast(tf.abs(tensor), dtype=tf.int32)
        tensor_decompressed = tf.gather(dict128, index) * scaler * sign
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class NaturalCompressor_old(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        sign = tf.cast(tf.sign(tensor_flatten), dtype=tf.int8)

        zeros = tf.zeros_like(tensor_flatten)
        exponent = tf.math.log(tf.abs(tensor_flatten)) / tf.math.log(2.0)
        h1 = tf.math.floor(tf.where(tf.math.abs(tensor_flatten) != 0, exponent, zeros))
        h2 = tf.where(tf.math.abs(tensor_flatten) != 0, tf.math.pow(2.0, h1), zeros)
        # extract probability
        p = tf.where(h2 != 0, tf.abs(tensor_flatten) / h2 - 1, h2)
        u = tf.floor(tf.random_uniform(tf.shape(p)) + p)
        tensor_compressed = h1 + u
        tensor_compressed = tf.cast(tensor_compressed, dtype=tf.int8), sign
        ctx = tensor_shape
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx
        tensor, sign = tensor_compressed
        tensor = tf.cast(tensor, dtype=tf.float32)
        sign = tf.cast(sign, dtype=tf.float32)
        tensor_decompressed = sign * tf.math.pow(2.0, tensor)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class NaturalCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        tensor_cast = tf.bitcast(tensor_flatten, tf.int32)
        sign = tf.bitwise.bitwise_and(tensor_cast, 0b10000000000000000000000000000000)
        exp = tf.bitwise.bitwise_and(tensor_cast, 0b01111111100000000000000000000000)
        mantissa = tf.bitwise.bitwise_and(tensor_cast, 0b00000000011111111111111111111111)
        exp_add_one = mantissa > tf.random.uniform(tf.shape(tensor_flatten), minval=0, maxval=0x007ffff,
                                                   dtype=tf.int32)
        # exp_add_one = mantissa > 0x00400000 # deterministic
        exponent = tf.where(exp_add_one, exp + 0b00000000100000000000000000000000, exp)
        # original exponent range: -128 ~ 127, clip to -110,  17
        # view as uint8_t:            0 ~ 255            18  145
        exp_shift = tf.clip_by_value(exponent, 0b00001001000000000000000000000000, 0b01001000100000000000000000000000)
        exps = tf.bitwise.right_shift(exp_shift, 23)
        # shift 18 so that 0 maps to -110 and 127 maps to 145
        # set MSB if negative
        exps = tf.bitwise.bitwise_or(tf.bitwise.right_shift(sign, 24), exps - 18)
        tensor_compressed = tf.cast(exps, tf.uint8)
        params['tensors_size_are_same'] = True

        return tensor_compressed, tensor_shape

    @staticmethod
    def decompress(tensor_compressed, tensor_shape, params=None):
        sign = tensor_compressed > 127
        exps = tf.bitwise.bitwise_and(tensor_compressed, 0b01111111)
        exps_shift = tf.bitwise.left_shift(tf.cast(exps + 18, tf.int32), 23)
        floats = tf.bitcast(exps_shift, tf.float32)
        tensor_decompressed = tf.where(sign, -floats, floats)
        tensor_decompressed = tf.multiply(tf.cast(exps >= 1, tensor_decompressed.dtype), tensor_decompressed)
        return tf.reshape(tensor_decompressed, tensor_shape)


class SketchCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):
        """
        def encode2bool(tensor, quantiles):
            tensor = tf.cast(tensor, dtype=tf.int32)
            bits = tf.cast(math.log(quantiles, 2) + 1, dtype=tf.int32)
            def cond(step, input_tensor, output):
                return step < bits

            def encode(step, input_tensor, output):
                base = tf.constant(2, tf.int32)
                temp = tf.floormod(input_tensor, base)
                output = output.write(step, temp)
                input_tensor = tf.floordiv(input_tensor, base)
                return step + 1, input_tensor, output

            step = tf.constant(0)
            output = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
            _, _, final_output = tf.while_loop(cond, encode, loop_vars=[step, tensor, output])
            encode_output = tf.cast(final_output.stack(), dtype=tf.bool)
            return encode_output
        """

        import tensorflow_probability as tfp
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        x = tensor_flatten
        quantiles = params["quantum_num"]
        edges = tfp.stats.quantiles(x, num_quantiles=quantiles, interpolation='linear')
        bins = tf.cast(tfp.stats.find_bins(x, edges), dtype=tf.int32)
        means = tf.unsorted_segment_mean(x, bins, num_segments=quantiles)

        tensor_compressed = tf.cast(bins, dtype=tf.uint8 if quantiles < 256 else tf.uint16)
        #tensor_compressed = encode2bool(tensor_compressed, quantiles)
        means = tf.reshape(means, [-1])
        ctx = tensor_shape
        tensor_compressed = tensor_compressed, means
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """
        def decode4bool(tensor, quantiles):
            tensor = tf.cast(tensor, dtype=tf.int32)
            bits = tf.cast(math.log(quantiles, 2) + 1, dtype=tf.int32)
            def cond(step, input_tensor, output):
                return step < bits

            def decode(step, input_tensor, output):
                base = tf.constant(2, tf.int32)
                temp = input_tensor[step, :]
                output = output + temp * tf.math.pow(base, step)
                return step + 1, input_tensor, output
            output = tf.zeros([tf.shape(tensor)[1]], dtype=tf.int32)
            step = tf.constant(0)
            _, _, decode_output = tf.while_loop(cond, decode, loop_vars=[step, tensor, output])
            return decode_output
        """
        # tensor_compressed = decode4bool(tensor_compressed, params["quantum_num"])
        tensor_shape = ctx
        tensor_compressed, means = tensor_compressed
        bins = tf.cast(tensor_compressed, dtype=tf.int32)
        tensor_decompressed = tf.gather(means, bins)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class INCEPTIONNCompressor(Compressor):
    """"""

    @staticmethod
    def compress(tensor, params):

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        tensor_cast = tf.bitcast(tensor_flatten, tf.uint32)
        sign = tf.bitwise.bitwise_and(tensor_cast, 0b10000000000000000000000000000000)
        exp = tf.bitwise.bitwise_and(tensor_cast, 0b01111111100000000000000000000000)
        mantissa = tf.bitwise.bitwise_and(tensor_cast, 0b00000000011111111111111111111111)

        exp = tf.bitwise.right_shift(exp, 23)

        error_bound_val = params["error_bound"]
        error_bound = 127 + int(math.log(error_bound_val / 2, 10))  # error_bound exponent: 117 for 2e-10
        radius = math.ceil((127 - error_bound) / 2)
        mid = error_bound + radius
        mask_32bit = exp >= 127
        mask_16bit = (exp >= mid) & (exp < 127)
        mask_8bit = (exp >= error_bound) & (exp < mid)
        indices_32bit = tf.reshape(tf.where(mask_32bit), [-1])
        indices_16bit = tf.reshape(tf.where(mask_16bit), [-1])
        indices_8bit = tf.reshape(tf.where(mask_8bit), [-1])

        # no compress
        v_32bit = tf.gather(tensor_flatten, indices_32bit)

        # 16bit compress
        s_16bit = tf.gather(sign, indices_16bit)
        e_16bit = tf.gather(exp, indices_16bit)
        m_16bit = tf.gather(mantissa, indices_16bit)
        n_shift = 127 - tf.cast(e_16bit, dtype=tf.int32)
        n_shift = tf.cast(n_shift, tf.uint32)
        shifted_s = tf.bitwise.right_shift(s_16bit, 8)
        marker = 0b00000000010000000000000000000000
        m_16bit_concat = tf.bitwise.bitwise_or(tf.bitwise.right_shift(m_16bit, 1), marker)
        shifted_m = tf.bitwise.right_shift(m_16bit_concat, n_shift)
        temp = tf.bitwise.bitwise_or(shifted_s, shifted_m)
        v_16bit = tf.cast(tf.bitwise.right_shift(temp, 8), dtype=tf.uint16)

        # 8bit compress
        s_8bit = tf.gather(sign, indices_8bit)
        e_8bit = tf.gather(exp, indices_8bit)
        m_8bit = tf.gather(mantissa, indices_8bit)
        n_shift = 127 - tf.cast(e_8bit, dtype=tf.int32)
        n_shift = tf.cast(n_shift, tf.uint32)
        shifted_s = tf.bitwise.right_shift(s_8bit, 8)
        marker = 0b00000000010000000000000000000000
        m_8bit_concat = tf.bitwise.bitwise_or(tf.bitwise.right_shift(m_8bit, 1), marker)
        shifted_m = tf.bitwise.right_shift(m_8bit_concat, n_shift)
        temp = tf.bitwise.bitwise_or(shifted_s, shifted_m)
        v_8bit = tf.cast(tf.bitwise.right_shift(temp, 16), dtype=tf.uint8)

        # concat indices
        # indices_all = tf.concat([indices_32bit, indices_16bit, indices_8bit], 0)
        # indices_all = tf.cast(indices_all, dtype=tf.int32)

        def encode_byte(a):
            # input: int32 type tensor with values in range 0,1,2,3 (2'b00,2'b01,2'b10,3'b11)
            # output: encoded uint8 type tensor
            a = tf.reshape(a, [-1])
            pad_size = 4 - tf.mod(tf.size(a), 4)
            pad = tf.range(0, pad_size)
            a = tf.concat([a, pad], 0)
            a_split1, a_split2, a_split3, a_split4 = tf.split(a, 4)

            # encode 4 grads into 1 Byte
            sum_1 = tf.add(a_split1, a_split2 * 4)
            sum_2 = tf.add(a_split3 * 16, a_split4 * 64)
            sum_all = tf.add(sum_1, sum_2)
            return tf.cast(sum_all, tf.uint8)

        # encode indices
        mask_encode = 0
        for mask, code in zip([mask_8bit, mask_16bit, mask_32bit], [1, 2, 3]):
            mask_encode += tf.cast(mask, tf.int32) * code
        mask_encode = encode_byte(mask_encode)
        tensor_compressed = v_32bit, v_16bit, v_8bit, mask_encode
        ctx = tensor_shape

        return tensor_compressed, ctx

    # decompress
    @staticmethod
    def decompress(tensor_compressed, ctx, params):

        def decode_byte(encoded, real_size):
            # input: encoded uint8 type tensor
            # output: int32 type tensor with values in range 0,1,2,3 (2'b00,2'b01,2'b10,3'b11)
            a = tf.cast(encoded, tf.int32)
            a_split1 = tf.mod(a, 4)
            a_split2 = tf.cast(tf.mod(a / 4, 4), tf.int32)
            a_split3 = tf.cast(tf.mod(a / 16, 4), tf.int32)
            a_split4 = tf.cast(tf.mod(a / 64, 4), tf.int32)
            a = tf.concat([a_split1, a_split2, a_split3, a_split4], 0)
            a = a[:real_size]
            return a

        v_32bit, v_16bit, v_8bit, mask_encode = tensor_compressed
        tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)

        # decode mask and gather indices
        mask_decode = decode_byte(mask_encode, tensor_size)
        mask_32bit = tf.equal(mask_decode, 3)
        mask_16bit = tf.equal(mask_decode, 2)
        mask_8bit = tf.equal(mask_decode, 1)
        indices_32bit = tf.reshape(tf.where(mask_32bit), [-1])
        indices_16bit = tf.reshape(tf.where(mask_16bit), [-1])
        indices_8bit = tf.reshape(tf.where(mask_8bit), [-1])

        edges_16bit = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        edges_8bit = [0, 2, 4, 8, 16, 32, 64, 128, 256]
        import tensorflow_probability as tfp

        # 16bit decompress
        # get the sign bit s_16bit and remove MSB from v_16bit
        s_16bit = tf.bitwise.bitwise_and(v_16bit, 0b1000000000000000)
        s_16bit = tf.cast(s_16bit, dtype=tf.int32)
        s_16bit = tf.bitwise.left_shift(s_16bit, 16)
        v_16bit = tf.bitwise.left_shift(v_16bit, 1)

        # 8bit decompress
        # get the sign bit s_8bit and remove MSB from v_8bit
        s_8bit = tf.bitwise.bitwise_and(v_8bit, 0b10000000)
        s_8bit = tf.cast(s_8bit, dtype=tf.int32)
        s_8bit = tf.bitwise.left_shift(s_8bit, 24)
        v_8bit = tf.bitwise.left_shift(v_8bit, 1)

        # find the marker bit in v_16bit and get the exponent
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.int32), trainable=False)
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.int32))
        with tf.control_dependencies([op]):
            temp = tf.scatter_update(zero_tensor, indices_16bit, tf.cast(v_16bit, tf.int32))
            temp = tf.scatter_update(temp, indices_8bit, tf.cast(v_8bit, tf.int32))
            n_shift_all = tfp.stats.find_bins(tf.cast(temp, dtype=tf.int32), edges_16bit)

        n_shift = 16 - tf.gather(n_shift_all, indices_16bit)
        e_16bit = 127 - (n_shift - 1)
        e_16bit = tf.bitwise.left_shift(e_16bit, 23)

        # restore the mantissa
        n_shift = tf.cast(n_shift, dtype=tf.uint16)
        v_16bit = tf.bitwise.left_shift(v_16bit, n_shift)
        v_16bit = tf.cast(v_16bit, dtype=tf.int32)
        m_16bit = tf.bitwise.left_shift(v_16bit, 7)

        # concat all
        temp = tf.bitwise.bitwise_or(s_16bit, e_16bit)
        v_16bit = tf.bitwise.bitwise_or(temp, m_16bit)
        v_16bit = tf.bitcast(v_16bit, tf.float32)


        # find the marker bit in v_8bit and get the exponent

        # zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=v_8bit.dtype), trainable=False)
        # op = zero_tensor.assign(tf.zeros([tensor_size], dtype=v_8bit.dtype))
        # with tf.control_dependencies([op]):
        #     temp = tf.scatter_update(zero_tensor, indices_8bit, v_8bit)
        # n_shift = 8 - tfp.stats.find_bins(tf.cast(temp, dtype=tf.int32), edges_8bit)
        n_shift = 8 - tf.gather(n_shift_all, indices_8bit)
        e_8bit = 127 - (n_shift - 1)
        e_8bit = tf.bitwise.left_shift(e_8bit, 23)

        # restore the mantissa
        n_shift = tf.cast(n_shift, dtype=tf.uint8)
        v_8bit = tf.bitwise.left_shift(v_8bit, n_shift)
        v_8bit = tf.cast(v_8bit, dtype=tf.int32)
        m_8bit = tf.bitwise.left_shift(v_8bit, 15)

        # concat all
        temp = tf.bitwise.bitwise_or(s_8bit, e_8bit)
        v_8bit = tf.bitwise.bitwise_or(temp, m_8bit)
        v_8bit = tf.bitcast(v_8bit, tf.float32)

        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32), trainable=False)
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.float32))
        with tf.control_dependencies([op]):
            temp = tf.scatter_update(zero_tensor, indices_32bit, v_32bit)
            temp = tf.scatter_update(temp, indices_16bit, v_16bit)
            temp = tf.scatter_update(temp, indices_8bit, v_8bit)
        tensor_decompressed = tf.reshape(temp, tensor_shape)
        return tensor_decompressed


class FakeCompressor(Compressor):
    """Default no-op compression."""

    @classmethod
    def memory_compensate(cls, tensor, params):
        """Update the tensor with the residuals."""

        return tensor

    @classmethod
    def memory_update(cls, tensor, tensor_compensate, tensor_compressed, ctx, params):
        """Update the residuals."""

        return []

    @classmethod
    def compress(cls, tensor, params):
        """Returns the tensor unmodified."""
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]

        if params['compress_method'] in ['randomk']:
            tensor_compressed = tensor_flatten[:max(1, int(elemnum * compress_ratio))]
            params['tensors_size_are_same'] = True

        elif params['compress_method'] in ['topk']:
            tensor_compressed = tensor_flatten[: 2 * max(1, int(elemnum * compress_ratio))]
            params['tensors_size_are_same'] = True

        elif params['compress_method'] in ['dgc', 'adas']:
            tensor_compressed = tensor_flatten[: 2 * max(1, int(elemnum * compress_ratio))]
            params['tensors_size_are_same'] = False

        elif params['compress_method'] in ['adaq']:
            tensor_compressed = tensor_flatten[: min(elemnum, 3 + int(elemnum * compress_ratio))]
            params['tensors_size_are_same'] = False

        elif params['compress_method'] in ['signsgd', 'signum', 'natural']:
            tensor_compressed = tf.cast(tensor_flatten, tf.int8)
            params['tensors_size_are_same'] = True

        elif params['compress_method'] in ['efsignsgd', 'terngrad', '8bit']:
            tensor_compressed = tensor_flatten[0], tf.cast(tensor_flatten, tf.int8)
            params['tensors_size_are_same'] = True

        elif params['compress_method'] in ['qsgd']:
            quantum_num = params["quantum_num"]

            if quantum_num < 128:
                tensor_compressed = tensor_flatten[0], tf.cast(tensor_flatten, tf.int8)
            else:
                tensor_compressed = tensor_flatten[0], tf.cast(tensor_flatten, tf.int16)
            params['tensors_size_are_same'] = True

        elif params['compress_method'] in ['sketch']:
            quantum_num = params["quantum_num"]
            mean_cache = tensor_flatten[0] * tf.ones([quantum_num], dtype=tf.float32)
            if quantum_num < 256:

                tensor_compressed = mean_cache, tf.cast(tensor_flatten, tf.int8)
            else:

                tensor_compressed = mean_cache, tf.cast(tensor_flatten, tf.int16)
            params['tensors_size_are_same'] = True

        elif params['compress_method'] in ['onebit']:
            tensor_compressed = tensor_flatten[:2], tf.cast(tensor_flatten, tf.int8)
            params['tensors_size_are_same'] = True

        elif params['compress_method'] in ['threshold']:
            #tensor_double = tf.concat([tensor_flatten, tensor_flatten], 0)

            if params['model_name'] == 'resnet20_v2':
                if params["threshold_val"] == 0.01 and params["use_memory"]:
                    tensor_compressed = tensor_flatten[:max(1, int(0.004551207 * elemnum))]

            elif params['model_name'] == 'densenet40_k12':
                if params["threshold_val"] == 0.01 and params["use_memory"]:
                    tensor_compressed = tensor_flatten[:max(1, int(0.016955392 * elemnum))]

            elif params['model_name'] == 'resnet50':
                if params["threshold_val"] == 0.01 and params["use_memory"]:
                    tensor_compressed = tensor_flatten[:max(1, int(0.116419225 * elemnum))]

            elif params['model_name'] == 'ncf':
                if params["threshold_val"] == 0.0001 and params["use_memory"]:
                    tensor_compressed = tensor_flatten[:max(1, int(0.001318898 * elemnum))]

            elif params['model_name'] == 'ptb':
                if params["threshold_val"] == 0.01 and params["use_memory"]:
                    tensor_compressed = tensor_flatten[:max(1, int(0.0182253132 * elemnum))]

            elif params['model_name'] == 'segmentation':
                if params["threshold_val"] == 0.01 and params["use_memory"]:
                    tensor_compressed = tensor_flatten[:max(1, int(0.0145350 * elemnum))]

            elif params['model_name'] == 'vgg19':
                tensor_compressed = tensor_flatten[:max(1, int(0.1 * elemnum))]

            params['tensors_size_are_same'] = True

        elif params['compress_method'] in ['powersgd']:
            if params['model_name'] == 'resnet20_v2':
                temp1 = tensor_flatten[:max(1, int(0.691594033 * 0.5 * elemnum))]
                temp2 = tensor_flatten[-max(1, int(0.691594033 * 0.5 * elemnum)):]

            elif params['model_name'] == 'densenet40_k12':
                temp1 = tensor_flatten[:max(1, int(0.5 * 0.5 * elemnum))]
                temp2 = tensor_flatten[-max(1, int(0.5 * 0.5 * elemnum)):]

            elif params['model_name'] == 'resnet50':
                temp1 = tensor_flatten[:max(1, int(0.53 * 0.5 * elemnum))]
                temp2 = tensor_flatten[-max(1, int(0.53 * 0.5 * elemnum)):]
            elif params['model_name'] == 'ncf':
                temp1 = tensor_flatten[:max(1, int(0.006557669 * 0.5 * elemnum))]
                temp2 = tensor_flatten[-max(1, int(0.006557669 * 0.5 * elemnum)):]
            elif params['model_name'] == 'ptb':
                temp1 = tensor_flatten[:max(1, int(0.003924611 * 0.5 * elemnum))]
                temp2 = tensor_flatten[-max(1, int(0.003924611 * 0.5 * elemnum)):]
            elif params['model_name'] == 'segmentation':
                temp1 = tensor_flatten[:max(1, int(0.69452657 * 0.5 * elemnum))]
                temp2 = tensor_flatten[-max(1, int(0.69452657 * 0.5 * elemnum)):]
            elif params['model_name'] == 'vgg19':
                temp1 = tensor_flatten[:max(1, int(0.5 * 0.5 * elemnum))]
                temp2 = tensor_flatten[-max(1, int(0.5 * 0.5 * elemnum)):]

            tensor_compressed = temp1, temp2
            params['tensors_size_are_same'] = True

        ctx = elemnum, tensor_shape
        return tensor_compressed, ctx

    @classmethod
    def decompress(cls, tensor_compressed, ctx, params):
        """Returns the tensor unmodified."""
        elemnum, tensor_shape = ctx

        if params['compress_method'] in ['randomk', 'topk', 'dgc', 'adas','adaq', 'threshold']:
            temp = tf.reshape(tensor_compressed[0], [-1])
            tensor_decompressed = tf.concat([temp, tf.ones([elemnum-1], dtype=tf.float32)], 0)

        elif params['compress_method'] in ['signsgd', 'signum', 'natural']:
            tensor_decompressed = tf.cast(tensor_compressed, tf.float32)

        elif params['compress_method'] in ['efsignsgd', 'terngrad', '8bit', 'qsgd']:
            temp = tf.reshape(tensor_compressed[0], [-1])
            tensor_decompressed = tf.concat([temp, tf.cast(tensor_compressed[1], tf.float32)[1:]], 0)

        elif params['compress_method'] in ['sketch', 'onebit']:
            temp = tf.reshape(tensor_compressed[0][0], [-1])
            tensor_decompressed = tf.concat([temp, tf.cast(tensor_compressed[1], tf.float32)[1:]], 0)

        elif params['compress_method'] in ['powersgd']:
            temp1, temp2 = tensor_compressed
            temp = tf.reshape((temp1+temp2)[0], [-1])
            tensor_decompressed = tf.concat([temp, tf.ones([elemnum - 1], dtype=tf.float32)], 0)

        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor
    randomk = RandomkCompressor
    topk = TopKCompressor
    threshold = ThresholdCompressor
    terngrad = TerngradCompressor
    qsgd = QsgdCompressor
    dgc = DgcCompressor
    adaq = AdaqCompressor
    signsgd = SignSGDCompressor
    efsignsgd = EFSignSGDCompressor
    signum = SignumCompressor
    adas = AdapSparseCompressor
    onebit = OnebitCompressor
    powersgd = PowerSGDCompressor
    u8bit = U8bitCompressor
    natural = NaturalCompressor
    sketch = SketchCompressor
    inceptionn = INCEPTIONNCompressor
    fake = FakeCompressor
    sparsifier = SparsifierCompressor
    bloom = Bloom_Filter_Compressor
    values_approximation = Values_Approximation_Compressor
    values_approximation_logit = Values_Approximation_Logit_Compressor
    topk_values_approximation = TopK_Values_Approximation_Compressor
    two_topk_values_approximation = Two_TopK_Values_Approximation_Compressor
    polynomial_segmented_values_approximation = Polynomial_Segmented_Values_Approximation_Compressor