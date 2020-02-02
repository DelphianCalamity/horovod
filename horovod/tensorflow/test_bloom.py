import tensorflow as tf

# Assuming Intiial tensor = [1, 10, 20, 2, 3]

init_tensor = tf.constant([1.3, 10.2, 20.3, 2.5, 3.6])
k=2

_, indices = tf.math.top_k(tf.math.abs(init_tensor), k, sorted=False)
values = tf.gather(init_tensor, indices)
with tf.Session() as sess:
	print("Values Int: ", sess.run(values))
values = tf.bitcast(values, tf.int32)

bloom_compressor = tf.load_op_library('./bloom_compressor_op.so').bloom_compressor
bloom_decompressor = tf.load_op_library('./bloom_decompressor_op.so').bloom_decompressor

hash_num = 2			# Number of hash functions used for bloom filter
bloom_size = 10			# Size of Bloom Filter
decompressed_size = 5	# Size of initial tensor

compressed_tensor = bloom_compressor(values, indices, init_tensor, hash_num=hash_num, bloom_size=bloom_size, logfile_suffix=1)
decompressed_tensor = bloom_decompressor(compressed_tensor, decompressed_size, hash_num=hash_num, bloom_size=bloom_size, logfile_suffix=1, suffix=1)

with tf.Session() as sess:
	print("Initial Tensor: ", sess.run(init_tensor))
	print("Values: ", sess.run(values))
	print("Indices: ", sess.run(indices))

	print("Compressed Tensor: ", sess.run(compressed_tensor))
	print("Compressed Tensor Shape: ", compressed_tensor.get_shape())
	print("Decompressed Tensor", sess.run(decompressed_tensor))
