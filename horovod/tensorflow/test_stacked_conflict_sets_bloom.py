from __future__ import division
import tensorflow as tf
import math

init_tensor = tf.constant([100, 102, 203, 250, 360, 300, 66, 100, 330, 220, 33, 66, 67, 8, 9, 2])
N=16
k=10

_, indices = tf.math.top_k(tf.math.abs(init_tensor), k, sorted=False)
indices = tf.sort(indices, axis=0, direction='ASCENDING')
values = tf.gather(init_tensor, indices)
with tf.Session() as sess:
	print("Values Int: ", sess.run(values))
values = tf.bitcast(values, tf.int32)

bloom_compressor = tf.load_op_library('./stacked_bloom_compressor_conflict_sets.so').stacked_bloom_compressor_conflict_sets
bloom_decompressor = tf.load_op_library('./stacked_bloom_decompressor_conflict_sets.so').stacked_bloom_decompressor_conflict_sets

# bloom size is given in bytes, so for a bloom of 8 bits set bloom_size to 1
# bloom_size = 1

fpr=0.08
fpr2=0.001

bloom_size = (k * abs(math.log(fpr))) / (math.pow(math.log(2), 2))
quot = int(bloom_size / 8)
rem = bloom_size % 8
bloom_size = quot
if rem != 0:
	bloom_size += 1

h = (bloom_size * 8 / k) * math.log(2)
hash_num = int(math.ceil(h))

k2 = math.ceil(fpr*N)
print(k2)
# k2=1
bloom_size2 = (k2 * abs(math.log(fpr2))) / (math.pow(math.log(2), 2))
quot = int(bloom_size2 / 8)
rem = bloom_size2 % 8
bloom_size2 = quot
if rem != 0:
	bloom_size2 += 1
hash_num2 = int(math.ceil((bloom_size2 * 8 / k2) * math.log(2)))

# bloom_size=1
# hash_num=8
# bloom_size2=1
# hash_num2=8

print("BLOOM:", bloom_size)
print("HASHNUM:", hash_num)
print("BLOOM2:", bloom_size2)
print("HASHNUM2:", hash_num2)

mem_mode=0
step=tf.placeholder(tf.int64, name='step')
compressed_tensor = bloom_compressor(values, indices, init_tensor, step,
									 hash_num=hash_num,
									 bloom_size=bloom_size,
									 hash_num2=hash_num2,
									 bloom_size2=bloom_size2,
									 logfile_suffix=1,
									 logs_path_suffix=1,
									 verbosity=1)

decompressed_tensor = bloom_decompressor(compressed_tensor, N, step, k,
										 mem_mode=mem_mode,
										 hash_num=hash_num,
										 bloom_size=bloom_size,
										 hash_num2=hash_num2,
										 bloom_size2=bloom_size2,
										 logfile_suffix=1,
										 logs_path_suffix=1,
										 suffix=1,
										 verbosity=1)

with tf.Session() as sess:
	print("Initial Tensor: ", sess.run(init_tensor))
	print("Values: ", sess.run(values))
	print("Indices: ", sess.run(indices))

	# sess.run(compressed_tensor, feed_dict={step:0})
	# print("Compressed Tensor Shape: ", compressed_tensor.get_shape())
	sess.run(decompressed_tensor, feed_dict={step:1})
