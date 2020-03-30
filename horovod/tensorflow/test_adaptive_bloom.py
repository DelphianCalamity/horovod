import tensorflow as tf
import math


# init_tensor = tf.constant([1.3, 10.2, 20.3, 2.5, 3.6])
# k=3
# log_init_tensor = tf.bitcast(init_tensor, tf.int32)
#
# _, indices = tf.math.top_k(tf.math.abs(init_tensor), k, sorted=False)
# indices = tf.sort(indices, axis=0, direction='ASCENDING')
# values = tf.gather(init_tensor, indices)
# with tf.Session() as sess:
# 	print("Values Int: ", sess.run(values))
# values = tf.bitcast(values, tf.int32)

values = tf.constant([1007309995])
indices = tf.constant([1])
log_init_tensor = tf.constant([1001874297, 1007309995, 943642542, 982145325, -1158051351, 969390216, 995928611, 977709979, -1164039257, -1150973382, -1145755296, -1153600937, 999516240, 983204270, -1151637699, -1180861488])

bloom_compressor = tf.load_op_library('./bloom_adaptive_compressor.so').bloom_adaptive_compressor
bloom_decompressor = tf.load_op_library('./bloom_adaptive_decompressor.so').bloom_adaptive_decompressor
decompressed_size = 16

partitioning = 1
k=1
fpr=0.01
m = (k * abs(math.log(fpr))) / (math.pow(math.log(2), 2))
bloom_size = m
quot = int(bloom_size / 8)
rem = bloom_size % 8
bloom_size = quot
if rem != 0:
	bloom_size += 1

print(bloom_size)



step=tf.placeholder(tf.int32, name='step')
compressed_tensor = bloom_compressor(values, indices,
									 log_init_tensor,
									 1,
									 partitioning=partitioning,
									 bloom_size=bloom_size,
									 logfile_suffix=1,
									 logs_path_suffix=1,
									 verbosity=1)
decompressed_tensor = bloom_decompressor(compressed_tensor, decompressed_size,
										 1,
										 partitioning=partitioning,
										 bloom_size=bloom_size,
										 logfile_suffix=1,
										 logs_path_suffix=1,
										 suffix=1,
										 verbosity=1)

with tf.Session() as sess:
	# print("Initial Tensor: ", sess.run(init_tensor))
	# print("Values: ", sess.run(values))
	# print("Indices: ", sess.run(indices))

	# sess.run(compressed_tensor, feed_dict={step:0})
	# print("Compressed Tensor Shape: ",
	# compressed_tensor.get_shape())
	sess.run(decompressed_tensor, feed_dict={step:0})
