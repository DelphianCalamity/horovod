import tensorflow as tf


init_tensor = tf.constant([1.3, 10.2, 20.3, 2.5, 3.6])
k=3
log_init_tensor = tf.bitcast(init_tensor, tf.int32)

_, indices = tf.math.top_k(tf.math.abs(init_tensor), k, sorted=False)
indices = tf.sort(indices, axis=0, direction='ASCENDING')
values = tf.gather(init_tensor, indices)
with tf.Session() as sess:
	print("Values Int: ", sess.run(values))
values = tf.bitcast(values, tf.int32)

bloom_compressor = tf.load_op_library('./bloom_adaptive_compressor.so').bloom_adaptive_compressor
bloom_decompressor = tf.load_op_library('./bloom_adaptive_decompressor.so').bloom_adaptive_decompressor

partitioning = 1
bloom_size = 2


step=tf.placeholder(tf.int32, name='step')
compressed_tensor = bloom_compressor(values, indices,
									 log_init_tensor,
									 1,
									 partitioning=partitioning,
									 bloom_size=bloom_size,
									 logfile_suffix=1,
									 logs_path_suffix=1,
									 verbosity=1)
decompressed_size = 5
decompressed_tensor = bloom_decompressor(compressed_tensor, decompressed_size,
										 1,
										 partitioning=partitioning,
										 bloom_size=bloom_size,
										 logfile_suffix=1,
										 logs_path_suffix=1,
										 suffix=1,
										 verbosity=1)

with tf.Session() as sess:
	print("Initial Tensor: ", sess.run(init_tensor))
	print("Values: ", sess.run(values))
	print("Indices: ", sess.run(indices))

	# sess.run(compressed_tensor, feed_dict={step:0})
	# print("Compressed Tensor Shape: ",
	# compressed_tensor.get_shape())
	sess.run(decompressed_tensor, feed_dict={step:0})
