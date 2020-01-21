"""Gradient compression algorithms."""

import tensorflow as tf
import random, math
from horovod.tensorflow.mpi_ops import _allreduce
from horovod.tensorflow.mpi_ops import rank


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
            if name in cls.residuals:
                tensor = beta * cls.residuals[name] + gamma * tensor
        return tensor

    @classmethod
    def memory_update(cls, tensor, tensor_compressed, ctx, params):
        """Update the residuals."""
        use_memory = params['use_memory']
        if use_memory:
            name = tensor.name
            tensor_decompressed = cls.decompress(tensor_compressed, ctx, params)
            cls.residuals[name] = tensor - tensor_decompressed

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


class RandomkCompressor(Compressor):
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""
    global_step = 0

    @staticmethod
    def compress(tensor, params):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]

        rand = random.Random()
        h = hash(tensor.name + str(RandomkCompressor.global_step))
        RandomkCompressor.global_step += 1
        rand.seed(h)
        var = rand.sample(range(elemnum), max(1, int(elemnum * compress_ratio)))
        #var.sort()
        indices = tf.convert_to_tensor(var, dtype=tf.int32)
        values = tf.gather(tensor_flatten, indices)
        ctx = indices, tensor_shape
        tensor_compressed = values
        params['tensors_size_are_same'] = True

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, tensor_shape = ctx
        values = tensor_compressed
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
        tensor_decompressed = tf.scatter_update(zero_tensor, indices, values)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class TopKCompressor(Compressor):
    """"""
    @staticmethod
    def compress(tensor, params):

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]

        k = max(1, int(elemnum * compress_ratio))
        _, indices = tf.math.top_k(tf.math.abs(tensor_flatten), k)
        values = tf.gather(tensor_flatten, indices)
        values = tf.bitcast(values, tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        ctx = tensor_shape
        params['tensors_size_are_same'] = True
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
        tensor_decompressed = tf.scatter_update(zero_tensor, indices, values)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


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
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
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
        if name in cls.residuals:
            tensor = cls.residuals[name] + lr * tensor
        return tensor

    @classmethod
    def memory_update(cls, tensor, tensor_compressed, ctx, params):
        """Update the residuals."""
        name = tensor.name
        tensor_decompressed = cls.decompress(tensor_compressed, ctx, params)
        cls.residuals[name] = tensor - tensor_decompressed

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
        if name in SignumCompressor.momentum:
            tensor = (1.0 - momentum) * tensor + momentum * SignumCompressor.momentum[name]
        SignumCompressor.momentum[name] = tensor

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
        #name = tensor.name

        horovod_size = params["horovod_size"]
        momentum = params["momentum"]
        gradient_clipping = params["gradient_clipping"]
        if gradient_clipping:
            tensor_squ_sum = tf.math.reduce_sum(tf.math.square(tensor))
            if params['debug']:
                tensor_squ_sum = tf.Print(tensor_squ_sum, [tf.size(tensor_squ_sum)],
                                          message=f"==Debug== tensor 0/1 on rank {rank()} {tensor_squ_sum.dtype} size:")
            thr_global = tf.math.sqrt(_allreduce(tensor_squ_sum))
            clipping_val = thr_global / tf.math.sqrt(float(horovod_size))
            tensor = tf.clip_by_value(tensor, -clipping_val, clipping_val)
        name = tensor.name
        if name in cls.residuals:
            cls.residuals[name] = momentum * cls.residuals[name] + tensor
        else:
            cls.residuals[name] = tensor
        if name in cls.gradients:
            cls.gradients[name] += cls.residuals[name]
            tensor = cls.gradients[name]
        else:
            cls.gradients[name] = tensor
        return tensor

    @classmethod
    def memory_update(cls, tensor, tensor_compressed, ctx, params):
        """Update the residuals."""
        name = tensor.name
        mask = params['mask']
        not_mask = tf.math.logical_not(mask)
        tensor_shape = tf.shape(tensor)
        temp = tf.reshape(cls.residuals[name], [-1]) * tf.cast(not_mask, dtype=tf.float32)
        cls.residuals[name] = tf.reshape(temp, tensor_shape)
        temp = tf.reshape(cls.gradients[name], [-1]) * tf.cast(not_mask, dtype=tf.float32)
        cls.gradients[name] = tf.reshape(temp, tensor_shape)

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
        mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), thr)

        selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))

        def body(thr, mask, selected):
            thr = tf.cond(selected > 1.3 * elemnum * compress_ratio, lambda: 1.3 * thr, lambda: 0.7 * thr)
            mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), thr)
            selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
            return thr, mask, selected

        def condition(thr, mask, selected):
            cond_a = selected > 1.3 * elemnum * compress_ratio
            cond_b = selected < 0.7 * elemnum * compress_ratio
            return tf.math.logical_or(cond_a, cond_b)

        thr, mask, _ = tf.while_loop(condition, body, (thr, mask, selected), maximum_iterations=10)

        indices = tf.reshape(tf.where(mask), [-1])
        values = tf.boolean_mask(tensor_flatten, mask)

        values = tf.bitcast(values, tf.int32)
        indices = tf.cast(indices, dtype=tf.int32)
        tensor_compressed = tf.concat([values, indices], 0)
        ctx = tensor_shape
        params['mask'] = mask
        params['tensors_size_are_same'] = False
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        tensor_shape = ctx
        values, indices = tf.split(tensor_compressed, 2)
        values = tf.bitcast(values, tf.float32)
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
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
            sample_size = tf.cast(tf.reshape((tf.math.round(tf.cast(mask_size, dtype=tf.float32) * 0.01)), [-1]),
                                   dtype=tf.int32)
            sample_index = tf.random.uniform(sample_size, minval=0, maxval=mask_size, dtype=tf.int32)
            sample_tensor = tf.gather(tensor_value, sample_index)
            k = tf.cast((tf.math.round(tf.cast(mask_size, dtype=tf.float32) * 0.01 * compress_ratio)),
                        dtype=tf.int32)
            vals, indices = tf.math.top_k(tf.math.abs(sample_tensor), k)
            thr = tf.math.reduce_min(vals)
            tensor_masked = tf.cast(tensor_mask, dtype=tf.float32) * tensor
            mask = tf.math.greater(tf.math.abs(tensor_masked), thr)

            # fix the issue of sampling in topk
            selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
            elemnum = tf.cast(mask_size, dtype=tf.float32)

            def body(thr, mask, selected):
                thr = tf.cond(selected > 1.3 * elemnum * compress_ratio, lambda: 1.3 * thr, lambda: 0.7 * thr)
                mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), thr)
                selected = tf.math.reduce_sum(tf.cast(mask, dtype=tf.float32))
                return thr, mask, selected

            def condition(thr, mask, selected):
                cond_a = selected > 1.3 * elemnum * compress_ratio
                cond_b = selected < 0.7 * elemnum * compress_ratio
                return tf.math.logical_or(cond_a, cond_b)

            thr, mask, _ = tf.while_loop(condition, body, (thr, mask, selected), maximum_iterations=10)

            indices = tf.reshape(tf.where(mask), [-1])
            tensor_value = tf.boolean_mask(tensor, mask)
            mean = tf.reshape(tf.math.reduce_mean(tensor_value), [-1])

            return indices, mean, mask

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
        #minus_indices_size = tf.reshape(tf.size(minus_indices), [-1])
        tensor_compressed = tf.concat([plus_mean, minus_mean, plus_indices_size, plus_indices, minus_indices], 0)
        ctx = tensor_shape, tensor_size
        params['tensors_size_are_same'] = False
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        plus_mean = tensor_compressed[0]
        minus_mean = tensor_compressed[1]
        plus_indices_size = tensor_compressed[2]
        plus_indices = tensor_compressed[3:3+plus_indices_size]
        minus_indices = tensor_compressed[3+plus_indices_size:]

        plus_mean = tf.bitcast(plus_mean, tf.float32)
        minus_mean = tf.bitcast(minus_mean, tf.float32)
        tensor_shape, tensor_size = ctx
        zero_tensor = tf.Variable(tf.zeros([tensor_size]))  # solve the error 'Tensor' object has no attribute '_lazy_read'
        plus_mean = tf.ones(tf.shape(plus_indices), dtype=tf.float32) * plus_mean
        minus_mean = tf.ones(tf.shape(minus_indices), dtype=tf.float32) * minus_mean
        tensor_decompressed = tf.scatter_update(zero_tensor, plus_indices, plus_mean)
        tensor_decompressed = tf.scatter_update(tensor_decompressed, minus_indices, minus_mean)
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
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
        tensor_decompressed = tf.scatter_update(zero_tensor, indices, values)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class PowerSGDCompressor(Compressor):
    """"""

    q_memory = {}

    @classmethod
    def memory_compensate(cls, tensor, params):
        """Update the tensor with the residuals."""
        return tensor

    @classmethod
    def memory_update(cls, tensor, tensor_compressed, ctx, params):
        """Update the residuals."""
        pass

    @staticmethod
    def compress(tensor, params):
        tensor_rank = len(tensor.get_shape().as_list())
        if tensor_rank == 1:
            return tensor, None
        horovod_size = params["horovod_size"]
        use_memory = params["use_memory"]
        compressor = params["compressor"]
        name = tensor.name
        tensor_shape = tf.shape(tensor)
        matrix = tf.reshape(tensor, [tensor_shape[0], -1])
        n = tf.shape(matrix)[0]
        m = tf.shape(matrix)[1]
        r = tf.math.minimum(n, m)
        r = tf.math.minimum(r, tf.rank(tensor))
        if use_memory:
            if name in compressor.q_memory:
                q = compressor.q_memory[name]
            else:
                q = tf.random.normal([m, r])
                q, _ = tf.linalg.qr(q)
        else:
            q = tf.random.normal([m, r])
            q, _ = tf.linalg.qr(q)
        p = tf.linalg.matmul(matrix, q)
        if params['debug']:
            p = tf.Print(p, [tf.size(p)],
                         message=f"==Debug== tensor 0/1 on rank {rank()} {p.dtype} size:")
        p = _allreduce(p) / horovod_size
        p, _ = tf.linalg.qr(p)
        q = tf.linalg.matmul(matrix, p, transpose_a=True)
        if params['debug']:
            q = tf.Print(q, [tf.size(q)],
                         message=f"==Debug== tensor 0/1 on rank {rank()} {q.dtype} size:")
        q = _allreduce(q) / horovod_size
        ctx = p, q, tensor_shape
        if use_memory:
            compressor.q_memory[name] = q
        return None, ctx

    @staticmethod
    def decompress(tensor, ctx, params):
        if ctx is None:
            return tensor
        p, q, tensor_shape = ctx
        new_tensor = tf.linalg.matmul(p, q, transpose_b=True)
        tensor_decompressed = tf.reshape(new_tensor, tensor_shape)
        return tensor_decompressed


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
        exp_shift = tf.clip_by_value(exponent, 0b00001001000000000000000000000000, 0b01001000100000000000000000000000)
        exps = tf.bitwise.right_shift(exp_shift, 23)
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

class FakeCompressor(Compressor):
    """Default no-op compression."""

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
            tensor_double = tf.concat([tensor_flatten, tensor_flatten], 0)

            if params['data_name'] == 'cifar10':
                if params["threshold_val"] == 0.01:
                    tensor_compressed = tensor_double[:max(1, int(0.002801085 * elemnum))]
                elif params["threshold_val"] == 0.001:
                    tensor_compressed = tensor_double[:max(1, int(0.191417873 * elemnum))]
                elif params["threshold_val"] == 0.0001:
                    tensor_compressed = tensor_double[:max(1, int(1.593330538 * elemnum))]
                elif params["threshold_val"] == 0.00001:
                    tensor_compressed = tensor_double[:max(1, int(1.95826873 * elemnum))]

            elif params['data_name'] == 'imagenet':
                if params["threshold_val"] == 0.0001:
                    tensor_compressed = tensor_double[:max(1, int(1.78922153 * elemnum))]
                elif params["threshold_val"] == 0.00001:
                    tensor_compressed = tensor_double[:max(1, int(1.97886104 * elemnum))]
            params['tensors_size_are_same'] = False

        elif params['compress_method'] in ['powersgd']:
            if params['data_name'] == 'cifar10':
                temp1 = tensor_flatten[:max(1, int(0.691594033 * 0.5 * elemnum))]
                temp2 = tensor_flatten[-max(1, int(0.691594033 * 0.5 * elemnum)):]
                tensor_compressed = temp1, temp2
            elif params['data_name'] == 'imagenet':
                temp1 = tensor_flatten[:max(1, int(0.637016277 * 0.5 * elemnum))]
                temp2 = tensor_flatten[-max(1, int(0.637016277 * 0.5 * elemnum)):]
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
            temp1 = tf.reshape(tensor_compressed[0][0], [-1])
            temp2 = tf.reshape(tensor_compressed[1][0], [-1])
            tensor_decompressed = tf.concat([temp1, temp2,
                                             tf.ones([elemnum - 2], dtype=tf.float32)], 0)

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
    fake = FakeCompressor
