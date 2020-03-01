import tensorflow as tf

# Assuming Intiial tensor = [1, 10, 20, 2, 3]

init_tensor = tf.constant([1.3, 10.2, 20.3, 2.5, 3.6])
k=2
log_init_tensor = tf.bitcast(init_tensor, tf.int32)



_, indices = tf.math.top_k(tf.math.abs(init_tensor), k, sorted=False)
values = tf.gather(init_tensor, indices)
with tf.Session() as sess:
	print("Values Int: ", sess.run(values))
values = tf.bitcast(values, tf.int32)

bloom_compressor = tf.load_op_library('./bloom_compressor_op.so').bloom_compressor
bloom_decompressor = tf.load_op_library('./bloom_decompressor_op.so').bloom_decompressor

hash_num = 1
# bloom size is given in bytes, so for a bloom of 8 bits set bloom_size to 1
bloom_size = 1
decompressed_size = 5

step=tf.placeholder(tf.int32, name='step')
compressed_tensor = bloom_compressor(values, indices,
									 log_init_tensor,
									 1,
									 hash_num=hash_num,
									 bloom_size=bloom_size,
									 logfile_suffix=1,
									 logs_path_suffix=1,
									 verbosity=1)

decompressed_tensor = bloom_decompressor(compressed_tensor, decompressed_size,
										 1,
										 hash_num=hash_num,
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
	# print("Compressed Tensor Shape: ", compressed_tensor.get_shape())
	sess.run(decompressed_tensor, feed_dict={step:0})
