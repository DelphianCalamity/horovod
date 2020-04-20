from __future__ import division
import tensorflow as tf
import math

init_tensor = tf.constant([100, 102, 203, 250, 360, 300, 66, 100, 330, 220, 33, 66, 67, 8, 9, 2])
N=16
k=10
log_init_tensor = tf.bitcast(init_tensor, tf.int32)
_, indices = tf.math.top_k(tf.math.abs(init_tensor), k, sorted=False)
indices = tf.sort(indices, axis=0, direction='ASCENDING')
values = tf.gather(init_tensor, indices)
with tf.Session() as sess:
	print("Values Int: ", sess.run(values))
values = tf.bitcast(values, tf.int32)

bloom_compressor = tf.load_op_library('./bloom_filter_compression.so').bloom_compressor
bloom_decompressor = tf.load_op_library('./bloom_filter_compression.so').bloom_decompressor
fpr=0.01
bloom_size = (k * abs(math.log(fpr))) / (math.pow(math.log(2), 2))
quot = int(bloom_size / 8)
rem = bloom_size % 8
bloom_size = quot
if rem != 0:
	bloom_size += 1

h = (bloom_size * 8 / k) * math.log(2)
hash_num = int(math.ceil(h))

# bloom_size=1
# hash_num=8

print("BLOOM:", bloom_size)
print("HASHNUM:", hash_num)

policy = "conflict_sets"
mem_mode=0
step=tf.placeholder(tf.int64, name='step')
compressed_tensor = bloom_compressor(values, indices,
									 log_init_tensor,
									 step,
									 false_positives_aware=True,
									 policy=policy,
									 hash_num=hash_num,
									 bloom_size=bloom_size,
									 bloom_logs_path="./logs",
									 gradient_id=0,
									 rank=0,
									 verbosity_frequency=1,
									 verbosity=2)

decompressed_tensor = bloom_decompressor(compressed_tensor, N, step,
										 policy=policy,
										 mem_mode=mem_mode,
										 hash_num=hash_num,
										 bloom_size=bloom_size,
										 bloom_logs_path="./logs",
										 gradient_id=0,
										 rank=0,
										 suffix=1,
										 verbosity_frequency=1,
										 verbosity=2)

with tf.Session() as sess:
	print("Initial Tensor: ", sess.run(init_tensor))
	print("Values: ", sess.run(values))
	print("Indices: ", sess.run(indices))
	# sess.run(compressed_tensor, feed_dict={step:0})
	# print("Compressed Tensor Shape: ", compressed_tensor.get_shape())
	sess.run(decompressed_tensor, feed_dict={step:1})
