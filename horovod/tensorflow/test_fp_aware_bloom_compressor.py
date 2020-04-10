from __future__ import division
import tensorflow as tf
import math


init_tensor = tf.constant([100, 102, 203, 250, 360, 300, 66, 100, 330, 220, 33, 66, 67, 8, 9, 2])
k=10


_, indices = tf.math.top_k(tf.math.abs(init_tensor), k, sorted=False)
indices = tf.sort(indices, axis=0, direction='ASCENDING')
# values = tf.gather(init_tensor, indices)
with tf.Session() as sess:
	print("Initial tensor: ", sess.run(init_tensor))
# values = tf.bitcast(values, tf.int32)

init_tensor = tf.bitcast(init_tensor, tf.int32)

bloom_compressor = tf.load_op_library('./fp_aware_bloom_compressor.so').fp_aware_bloom_compressor
bloom_decompressor = tf.load_op_library('./bloom_decompressor.so').bloom_decompressor

# bloom size is given in bytes, so for a bloom of 8 bits set bloom_size to 1
bloom_size = 1
h = (bloom_size / k) * math.log(2)
print(h)
hash_num = int(math.ceil(h))
decompressed_size = 16
print(hash_num)
step=tf.placeholder(tf.int32, name='step')
mem_mode=0
compressed_tensor = bloom_compressor(init_tensor, indices,
									 1,
									 hash_num=hash_num,
									 bloom_size=bloom_size,
									 logfile_suffix=1,
									 logs_path_suffix=1,
									 verbosity=1)

decompressed_tensor = bloom_decompressor(compressed_tensor, decompressed_size,
										 1, k,
										 mem_mode=mem_mode,
										 hash_num=hash_num,
										 bloom_size=bloom_size,
										 logfile_suffix=1,
										 logs_path_suffix=1,
										 suffix=1,
										 verbosity=1)

with tf.Session() as sess:
	print("Initial Tensor: ", sess.run(init_tensor))
	# print("Values: ", sess.run(values))
	print("Indices: ", sess.run(indices))
	# sess.run(compressed_tensor, feed_dict={step:0})
	# print("Compressed Tensor Shape: ", compressed_tensor.get_shape())
	sess.run(decompressed_tensor, feed_dict={step:1})
