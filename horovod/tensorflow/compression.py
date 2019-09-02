# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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
"""Gradient compression algorithms."""

import tensorflow as tf



class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor):
        print("====================non-compress========================")
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        print("====================non-decompress========================")
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        print("====================FP16 compress========================")
        print(tensor.shape,tensor.name,tensor.dtype)
        if tensor.dtype.is_floating:
            # Only allow compression from other floating point types
            tensor_compressed = tf.cast(tensor, dtype=tf.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        print("====================FP16 decompress========================")
        print(tensor.shape,tensor.name,tensor.dtype)
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating:
            tensor_decompressed = tf.cast(tensor, dtype=dtype)
        return tensor_decompressed

import random


class DRSCompressor(Compressor):
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

    @staticmethod
    def compress(tensor,compress_ratio):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""
        print("====================DRS_compress========================")
        print("ratio=",compress_ratio)
        rand = random.Random()
        shape = tf.shape(tensor)
        elemnum = tf.size(tensor)
        global_step_tensor = tf.Variable(1, trainable=False, name='global_step')
        with tf.Session() as sess:
            elemnum = elemnum.eval()
            sess.run(global_step_tensor.initializer)
            global_step = tf.train.global_step(sess, global_step_tensor)
            print('tensor size: %s and global step: %s', str(elemnum), global_step)
        h = hash(tensor.name + str(global_step))

        rand.seed(h)
        var = rand.sample(xrange(elemnum), max(1, int(elemnum * compress_ratio)))
        var.sort()
        indices = tf.convert_to_tensor(var, dtype=tf.int32)

        tensor_sparsed = tf.reshape(tensor, [-1])
        tensor_sparsed = tf.gather(tensor_sparsed, indices)
        ctx = (shape, indices, elemnum)
        return tensor_sparsed, ctx

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        print("====================DRS_decompress========================")
        shape = ctx[0]
        indices = ctx[1]
        elemnum = ctx[2]
        tensor_decompressed = tensor
        zero_tensor = tf.Variable(tf.zeros([elemnum], dtype=tf.float32))
        tensor_decompressed = tf.scatter_update(zero_tensor, indices, tensor_decompressed)
        tensor_decompressed = tf.reshape(tensor_decompressed, shape)
        return tensor_decompressed



class TopKCompressor(Compressor):
    """Default no-op sparser."""

    residuals = {}
    zero_conditions = {}
    @staticmethod
    def compress(tensor, size, compress_ratio):
        print('===============TopKCompressor called======================')

        name = tensor.name
        if name not in TopKCompressor.residuals:
            TopKCompressor.residuals[name] = tf.zeros_like(tensor, dtype=tensor.dtype)

        tensor = tf.math.add(tensor, TopKCompressor.residuals[name])
        with tf.device('/device:GPU:0'):
            #using TF API
            #new_tensor = tf.reshape(tensor, [-1])
            #vals, indices = tf.math.top_k(tensor, int(size * sparse_step / 100))
            nnz= max(1,int(size * compress_ratio))
            vals, indices = tf.math.top_k(tf.math.abs(tensor),nnz )
            vals = tf.gather(tensor, indices)


        zero_condition = tf.Variable(tf.ones_like(tensor,dtype=tensor.dtype))
        zero_condition = tf.scatter_update(zero_condition,indices, tf.zeros_like(indices,dtype=tensor.dtype))
        TopKCompressor.residuals[name] = tensor * zero_condition

        print('reshaped the tensor to sparse size', vals, indices)
        return vals, indices

    @staticmethod
    def decompress(tensor, ctx):
        print("====================non-decompress========================")
        return tensor



class ThresholdCompressor(Compressor):
    """Default no-op sparser."""

    residuals = {}
    zero_conditions = {}
    @staticmethod
    def compress(tensor, size, sparse_step):
        print('===============ThresholdCompressor called======================')
        with tf.device('/device:GPU:0'):
            name = tensor.name
            if name not in ThresholdCompressor.residuals:
                ThresholdCompressor.residuals[name] = tf.zeros_like(tensor, dtype=tensor.dtype)

            tensor = tf.math.add(tensor, ThresholdCompressor.residuals[name])

            #using TF API
            #new_tensor = tf.reshape(tensor, [-1])
            #vals, indices = tf.math.top_k(tensor, int(size * sparse_step / 100))
            # nnz= max(1,int(size * sparse_step / 100))
            thr_val = 0.01
            zero_tensor = tf.zeros_like(tensor, dtype=tensor.dtype)
            thr_condition = tf.math.greater_equal(tf.math.abs(tensor), thr_val)
            vals = tf.boolean_mask(tensor, thr_condition)
            indices = tf.reshape(tf.where(thr_condition), [-1])
            #index_size = tf.reshape(tf.math.count_nonzero(thr_condition),[-1])

            zero_condition = tf.Variable(tf.ones_like(tensor,dtype=tensor.dtype))
            zero_condition = tf.scatter_update(zero_condition,indices, tf.zeros_like(indices,dtype=tensor.dtype))
            ThresholdCompressor.residuals[name] = tensor * zero_condition

        print('reshaped the tensor to sparse size', vals, indices)
        return vals, indices, #index_size

    @staticmethod
    def decompress(tensor, ctx):
        print("====================non-decompress========================")
        return tensor


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor
    randomk = DRSCompressor
    topk = TopKCompressor
    threshold = ThresholdCompressor
