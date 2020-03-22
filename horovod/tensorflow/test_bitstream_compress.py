import tensorflow as tf

init_tensor = tf.constant([-1.3, 10.2, 20.3, 2.5, 3.6])
k=2
log_init_tensor = tf.bitcast(init_tensor, tf.int32)

# _, indices = tf.math.top_k(tf.math.abs(init_tensor), k, sorted=False)
# indices = tf.sort(indices, axis=0, direction='ASCENDING')
# values = tf.gather(init_tensor, indices)
# with tf.Session() as sess:
# 	print("Values Int: ", sess.run(values))
# values = tf.bitcast(values, tf.int32)

indices = tf.constant([4000, 2550, 2444, 1000])

bitstream_compressor = tf.load_op_library('./bitstream_compression.so').bitstream_compressor
bitstream_decompressor = tf.load_op_library('./bitstream_compression.so').bitstream_decompressor

step=tf.placeholder(tf.int64, name='step')
compressed_indices = bitstream_compressor(indices, 4500,
									 1,
									 logfile_suffix=1,
									 logs_path_suffix=1,
									 verbosity=1)

# tensor_compressed = tf.concat([values, compressed_indices], 0)

decompressed_tensor = bitstream_decompressor(compressed_indices, 4,
											 1,
											 logfile_suffix=1,
											 logs_path_suffix=1,
											 verbosity=1,
											 suffix=0)

with tf.Session() as sess:
	print("Initial Tensor: ", sess.run(init_tensor))
	print("Indices: ", sess.run(indices))
	# print("Compressed Indices: ", sess.run(compressed_indices))
	# print("Tensor Compressed: ", sess.run(tensor_compressed, feed_dict={step:0}))
	print("Tensor De-Compressed: ", sess.run(decompressed_tensor))
	# print("Compressed Tensor Shape: ", compressed_tensor.get_shape())
	# sess.run(decompressed_tensor, feed_dict={step:0})
