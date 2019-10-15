"""Gradient compression algorithms."""

import tensorflow as tf
import random, math
from horovod.tensorflow.mpi_ops import _allreduce

class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    @staticmethod
    def compress(tensor, params):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx, params):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""

    @staticmethod
    def compress(tensor, params):
        """Returns the tensor unmodified."""
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
    residuals = {}
    global_step = 0

    @staticmethod
    def compress(tensor, params):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        use_memory = params["use_memory"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]

        rand = random.Random()
        h = hash(tensor.name + str(RandomkCompressor.global_step))
        RandomkCompressor.global_step += 1
        rand.seed(h)
        var = rand.sample(xrange(elemnum), max(1, int(elemnum * compress_ratio)))
        # var.sort()
        indices = tf.convert_to_tensor(var, dtype=tf.int32)
        tensor_compressed = tf.gather(tensor_flatten, indices)
        ctx = indices, tensor_shape

        if use_memory:
            tensor_decompressed = compressor.decompress(tensor_compressed, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
        tensor_decompressed = tf.scatter_update(zero_tensor, indices, tensor_compressed)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class TopKCompressor(Compressor):

    residuals = {}

    @staticmethod
    def compress(tensor, params):

        use_memory = params["use_memory"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]
        compress_ratio = params["compress_ratio"]

        k = max(1, int(elemnum * compress_ratio))
        _, indices = tf.math.top_k(tf.math.abs(tensor_flatten), k)
        tensor_compressed = tf.gather(tensor_flatten, indices)
        ctx = indices, tensor_shape

        if use_memory:
            tensor_decompressed = compressor.decompress(tensor_compressed, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
        tensor_decompressed = tf.scatter_update(zero_tensor, indices, tensor_compressed)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class ThresholdCompressor(Compressor):

    residuals = {}

    @staticmethod
    def compress(tensor, params):
        use_memory = params["use_memory"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        threshold_val = params["threshold_val"]
        thr_mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), threshold_val)
        tensor_compressed = tf.boolean_mask(tensor_flatten, thr_mask)
        indices = tf.reshape(tf.where(thr_mask), [-1])
        ctx = indices, tensor_shape

        if use_memory:
            tensor_decompressed = compressor.decompress(tensor_compressed, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        indices, tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
        tensor_decompressed = tf.scatter_update(zero_tensor, indices, tensor_compressed)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class SignSGDCompressor(Compressor):

    residuals = {}

    @staticmethod
    def compress(tensor, params):

        use_memory = params["use_memory"]
        lr = params["learning_rate"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor = lr * tensor + compressor.residuals[name]

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        tensor_compressed = tf.math.greater_equal(tensor_flatten, 0)
        mean = tf.math.reduce_mean(tf.abs(tensor_flatten)) if use_memory else None
        ctx = mean, tensor_shape
        if use_memory:
            tensor_decompressed = compressor.decompress(tensor_compressed, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(sign_encode, ctx, params):

        mean, tensor_shape = ctx
        sign_decode = tf.cast(sign_encode, dtype=tf.float32) * 2.0 - 1.0
        use_memory = params["use_memory"]
        sign_decode = mean * sign_decode if use_memory else sign_decode
        tensor_decompressed = tf.reshape(sign_decode, tensor_shape)
        return tensor_decompressed


class SignumCompressor(Compressor):

    residuals = {}
    momentum = {}

    @staticmethod
    def compress(tensor, params):

        use_memory = params["use_memory"]
        lr = params["learning_rate"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor = lr * tensor + compressor.residuals[name]
        # update tensor by momentum
        momentum = params["momentum"]
        name = tensor.name
        if name in compressor.momentum:
            tensor = (1.0 - momentum) * tensor + momentum * compressor.momentum[name]
        compressor.momentum[name] = tensor

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        tensor_compressed = tf.math.greater_equal(tensor_flatten, 0)
        mean = tf.math.reduce_mean(tf.abs(tensor_flatten)) if use_memory else None
        ctx = mean, tensor_shape
        if use_memory:
            tensor_decompressed = compressor.decompress(tensor_compressed, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(sign_encode, ctx, params):

        mean, tensor_shape = ctx
        sign_decode = tf.cast(sign_encode, dtype=tf.float32) * 2.0 - 1.0
        use_memory = params["use_memory"]
        sign_decode = mean * sign_decode if use_memory else sign_decode
        tensor_decompressed = tf.reshape(sign_decode, tensor_shape)
        return tensor_decompressed


class QsgdCompressor(Compressor):

    residuals = {}

    @staticmethod
    def compress(tensor, params):

        use_memory = params["use_memory"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]

        quantum_num = params["quantum_num"]
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        norm = tf.reshape(tf.norm(tensor_flatten), [-1])
        abs_gradient = tf.abs(tensor_flatten)
        bits = tf.cast(math.log(quantum_num, 2) + 1, dtype=tf.int32)
        qnum = tf.cast(quantum_num, dtype=tf.float32)

        level_float = qnum / norm * abs_gradient
        previous_level = tf.math.floor(level_float)
        prob = tf.random.uniform(tf.shape(tensor_flatten))
        is_next_level = tf.cast(tf.math.less(prob, (level_float - previous_level)), tf.float32)
        new_level = tf.cast(previous_level + is_next_level, tf.int32)

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
        _, _, final_output = tf.while_loop(cond, encode, loop_vars=[step, new_level, output])
        encode_output = tf.cast(final_output.stack(), dtype=tf.bool)

        sign = tf.reshape(tf.greater_equal(tensor, 0), [1, tf.size(tensor)])
        encode_output = tf.concat([sign, encode_output], 0)

        tensor_compressed = encode_output
        ctx = norm, tensor_shape

        if use_memory:
            tensor_decompressed = compressor.decompress(tensor_compressed, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(encode_output, ctx, params):

        quantum_num = params["quantum_num"]
        norm, tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        bits = tf.cast(math.log(quantum_num, 2) + 1, dtype=tf.int32)
        qnum = tf.cast(quantum_num, dtype=tf.float32)

        def cond(step, input_tensor, output):
            return step < bits

        def decode(step, input_tensor, output):
            base = tf.constant(2, tf.int32)
            temp = input_tensor[step, :]
            output = output + temp * tf.math.pow(base, step)
            return step + 1, input_tensor, output

        encode_output = tf.cast(encode_output, dtype=tf.int32)
        sign = encode_output[0, :] * 2 - 1
        input_tensor = encode_output[1:, :]
        output = tf.zeros([tensor_size], dtype=tf.int32)
        step = tf.constant(0)
        _, _, decode_output = tf.while_loop(cond, decode, loop_vars=[step, input_tensor, output])

        decode_output = sign * decode_output
        decode_output = tf.cast(decode_output, dtype=tf.float32)
        tensor_decompressed = norm / qnum * decode_output
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class OnebitCompressor(Compressor):

    residuals = {}

    @staticmethod
    def compress(tensor, params):

        horovod_size = params["horovod_size"]
        use_memory = params["use_memory"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        mean = tf.math.reduce_mean(tensor_flatten)
        mean = _allreduce(mean) / horovod_size

        mask0 = tf.math.less(tensor_flatten, mean)
        sum0 = tf.math.reduce_sum(tf.boolean_mask(tensor_flatten, mask0))
        num0 = tf.math.reduce_sum(tf.cast(mask0, dtype=tf.float32))
        mask1 = tf.math.greater_equal(tensor_flatten, mean)
        sum1 = tf.math.reduce_sum(tf.boolean_mask(tensor_flatten, mask1))
        num1 = tf.math.reduce_sum(tf.cast(mask1, dtype=tf.float32))
        num0 = _allreduce(num0)
        num1 = _allreduce(num1)
        num0 = tf.where(tf.math.greater(num0, 0), num0, tf.ones_like(num0))
        num1 = tf.where(tf.math.greater(num1, 0), num1, tf.ones_like(num1))
        mean0 = _allreduce(sum0) / num0
        mean1 = _allreduce(sum1) / num1
        # for allgather
        # mean0 = (sum0) / num0
        # mean1 = (sum1) / num1

        newmean = (mean0 + mean1) * 0.5
        radius = (mean1 - newmean) * 2.0
        lower = newmean - radius
        upper = newmean + radius
        quantum_mid = (lower + upper) / 2.0
        tensor_compressed = tf.math.greater_equal(tensor_flatten, quantum_mid)

        quantum_mid = tf.reshape(quantum_mid, [-1])
        lower = tf.reshape(lower, [-1])
        ctx = quantum_mid, lower, tensor_shape

        if use_memory:
            tensor_decompressed = compressor.decompress(tensor_compressed, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):

        comm_method = params["comm_method"]
        quantum_mid, lower, tensor_shape = ctx
        new_tensor = tf.cast(tensor_compressed, dtype=tf.float32)
        # new_tensor = new_tensor * quantum_mid + 0.5 * quantum_mid + lower
        new_tensor = new_tensor * 2 - 1
        tensor_decompressed = new_tensor * (quantum_mid - lower) + quantum_mid
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class TerngradCompressor(Compressor):
    residuals = {}

    @staticmethod
    def compress(tensor, params):

        """Encoding and compressing the signs """

        use_memory = params["use_memory"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]

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
        scaler = tf.reshape(scaler, [-1])
        ctx = scaler, tensor_shape
        if use_memory:
            tensor_decompressed = tf.reshape(binarized_gradient, tensor_shape)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decoding the signs to float format """
        scaler, tensor_shape = ctx
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
        a = tf.subtract(a, 1)
        tensor_decompressed = a * scaler
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class DgcCompressor(Compressor):

    residuals = {}
    gradients = {}

    @staticmethod
    def compress(tensor, params):
        horovod_size = params["horovod_size"]
        compress_ratio = params["compress_ratio"]
        momentum = params["momentum"]
        gradient_clipping = params["gradient_clipping"]
        name = tensor.name
        compressor = params["compressor"]

        if gradient_clipping:
            tensor_squ_sum = tf.math.reduce_sum(tf.math.square(tensor))
            thr_global = tf.math.sqrt(_allreduce(tensor_squ_sum))
            clipping_val = thr_global / tf.math.sqrt(float(horovod_size))
            tensor = tf.clip_by_value(tensor, -clipping_val, clipping_val)

        if name in compressor.residuals:
            compressor.residuals[name] = momentum * compressor.residuals[name] + tensor
        else:
            compressor.residuals[name] = tf.zeros_like(tf.reshape(tensor, [-1]))
        if name in compressor.gradients:
            compressor.gradients[name] += compressor.residuals[name]
        else:
            compressor.gradients[name] = tf.zeros_like(tf.reshape(tensor, [-1]))
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]

        sample_shape = tf.reshape(tf.convert_to_tensor(max(1, int(elemnum * 0.1)), dtype=tf.int32), [-1])
        sample_index = tf.random.uniform(sample_shape, minval=0, maxval=elemnum, dtype=tf.int32)
        sample_tensor = tf.gather(compressor.gradients[name], sample_index)

        k = max(1, int(elemnum * compress_ratio * 0.1))
        vals, indices = tf.math.top_k(tf.math.abs(sample_tensor), k)
        thr = tf.math.reduce_min(vals)

        mask = tf.math.greater_equal(tf.math.abs(tensor_flatten), thr)

        tensor_compressed = tf.boolean_mask(tensor_flatten, mask)
        indices = tf.reshape(tf.where(mask), [-1])
        ctx = indices, tensor_shape

        not_mask = tf.math.logical_not(mask)
        compressor.residuals[name] = compressor.residuals[name] * tf.cast(not_mask, dtype=tf.float32)
        compressor.gradients[name] = compressor.gradients[name] * tf.cast(not_mask, dtype=tf.float32)

        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        indices, tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
        tensor_decompressed = tf.scatter_update(zero_tensor, indices, tensor_compressed)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class AdaqCompressor(Compressor):

    residuals = {}

    @staticmethod
    def compress(tensor, params):
        use_memory = params["use_memory"]
        compress_ratio = params["compress_ratio"]

        def quan(tensor, tensor_mask, compress_ratio):
            # tensor_mask = tf.math.greater_equal(tf.math.abs(tensor),0) # for testing and debuging
            tensor_value = tf.boolean_mask(tensor, tensor_mask)
            sample_size = tf.reduce_sum(tf.cast(tensor_mask, dtype=tf.int32))
            sample_shape = tf.cast(tf.reshape((tf.math.round(tf.cast(sample_size, dtype=tf.float32) * 0.1)), [-1]),
                                   dtype=tf.int32)
            sample_index = tf.random.uniform(sample_shape, minval=0, maxval=sample_size, dtype=tf.int32)
            sample_tensor = tf.gather(tensor_value, sample_index)

            k = tf.cast((tf.math.round(tf.cast(sample_size, dtype=tf.float32) * 0.1 * compress_ratio)),
                        dtype=tf.int32)
            vals, indices = tf.math.top_k(tf.math.abs(sample_tensor), k)
            thr = tf.math.reduce_min(vals)

            tensor_masked = tf.cast(tensor_mask, dtype=tf.float32) * tensor
            mask = tf.math.greater(tf.math.abs(tensor_masked), thr)
            indices = tf.reshape(tf.where(mask), [-1])
            mean = tf.reshape(tf.math.reduce_mean(tensor_value), [-1])

            return indices, mean, mask

        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        plus_tensor = tf.math.greater(tensor_flatten, 0)
        minus_tensor = tf.math.less(tensor_flatten, 0)
        plus_indices, plus_mean, plus_mask = quan(tensor_flatten, plus_tensor, compress_ratio)
        minus_indices, minus_mean, minus_mask = quan(tensor_flatten, minus_tensor, compress_ratio)

        ctx = plus_indices, plus_mean, minus_indices, minus_mean
        if use_memory:
            tensor_decompressed = compressor.decompress(tensor, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor, ctx

    @staticmethod
    def decompress(tensor, ctx, params):
        plus_indices, plus_mean, minus_indices, minus_mean = ctx
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        zero_tensor = tf.Variable(
            tf.zeros_like(tensor_flatten))  # solve the error 'Tensor' object has no attribute '_lazy_read'
        plus_mean = tf.ones(tf.shape(plus_indices), dtype=tf.float32) * plus_mean
        minus_mean = tf.ones(tf.shape(minus_indices), dtype=tf.float32) * minus_mean
        tensor_decompressed = tf.scatter_update(zero_tensor, plus_indices, plus_mean)
        tensor_decompressed = tf.scatter_update(tensor_decompressed, minus_indices, minus_mean)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class AdapSparseCompressor(Compressor):

    residuals = {}

    @staticmethod
    def compress(tensor, params):

        use_memory = params["use_memory"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]

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
        tensor_compressed = tf.gather(tensor_flatten / prob, indices)
        ctx = indices, tensor_shape

        if use_memory:
            residual_mask = tf.cast(tf.math.logical_not(mask), dtype=tf.float32)
            tensor_decompressed = tensor_flatten * residual_mask
            tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
            compressor.residuals[name] = tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32))
        tensor_decompressed = tf.scatter_update(zero_tensor, indices, tensor_compressed)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed


class PowerSGDCompressor(Compressor):

    q_memory = {}

    @staticmethod
    def compress(tensor, params):
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
        p = _allreduce(p) / horovod_size
        p, _ = tf.linalg.qr(p)
        q = tf.linalg.matmul(matrix, p, transpose_a=True)
        q = _allreduce(q) / horovod_size
        ctx = p, q, tensor_shape
        if use_memory:
            compressor.q_memory[name] = q
        return tensor, ctx

    @staticmethod
    def decompress(tensor, ctx, params):
        p, q, tensor_shape = ctx
        new_tensor = tf.linalg.matmul(p, q, transpose_b=True)
        tensor_decompressed = tf.reshape(new_tensor, tensor_shape)
        return tensor_decompressed


class U8bitCompressor(Compressor):

    residuals = {}

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

        use_memory = params["use_memory"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        tensor_size = tf.size(tensor)

        scaler = tf.math.reduce_max(tf.abs(tensor_flatten))
        new_tensor = tensor_flatten / scaler
        sign = tf.sign(tensor_flatten)
        new_tensor = tf.abs(new_tensor)

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
        scaler = tf.reshape(scaler, [-1])
        ctx = scaler, tensor_shape

        if use_memory:
            tensor_decompressed = compressor.decompress(tensor_compressed, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor, ctx, params):
        scaler, tensor_shape = ctx
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


class NaturalCompressor(Compressor):

    residuals = {}

    @staticmethod
    def compress(tensor, params):

        use_memory = params["use_memory"]
        compressor = params["compressor"]
        if use_memory:
            name = tensor.name
            if name in compressor.residuals:
                tensor += compressor.residuals[name]

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        zeros = tf.zeros_like(tensor_flatten)
        exponent = tf.math.log(tf.abs(tensor_flatten)) / tf.math.log(2.0)
        h1 = tf.math.floor(tf.where(tf.math.abs(tensor_flatten) != 0, exponent, zeros))
        h2 = tf.where(tf.math.abs(tensor_flatten) != 0, tf.math.pow(2.0, h1), zeros)
        # extract probability
        p = tf.where(h2 != 0, tf.abs(tensor_flatten) / h2 - 1, h2)
        u = tf.floor(tf.random_uniform(tf.shape(p)) + p)
        tensor_compressed = h1 + u
        tensor_compressed = tf.cast(tensor_compressed, dtype=tf.int8)
        sign = tf.cast(tf.sign(tensor_flatten), dtype=tf.int8)
        ctx = sign, tensor_shape

        if use_memory:
            tensor_decompressed = compressor.decompress(tensor_compressed, ctx, params)
            compressor.residuals[name] = tensor - tensor_decompressed
        return tensor_compressed, ctx

    @staticmethod
    def decompress(tensor_compressed, ctx, params):
        sign, tensor_shape = ctx
        tensor = tf.cast(tensor_compressed, dtype=tf.float32)
        sign = tf.cast(sign, dtype=tf.float32)
        tensor_decompressed = sign * tf.math.pow(2.0, tensor)
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
    signum = SignumCompressor
    adas = AdapSparseCompressor
    onebit = OnebitCompressor
    powersgd = PowerSGDCompressor
    u8bit = U8bitCompressor
    natural = NaturalCompressor