import tensorflow as tf

# Assuming Intiial tensor = [1, 10, 20, 2, 3]
values = tf.constant([10,20])
print(values.shape)
print(values.get_shape())

indices = tf.constant([1,2])
print(indices.shape)

bloom_compressor = tf.load_op_library('./bloom_compressor_op.so').bloom_compressor
bloom_decompressor = tf.load_op_library('./bloom_decompressor_op.so').bloom_decompressor

hash_num = 2			# Number of hash functions used for bloom filter
bloom_size = 10			# Size of Bloom Filter
decompressed_size = 5	# Size of initial tensor

compressed_tensor = bloom_compressor(values, indices, hash_num=hash_num, bloom_size=bloom_size)
decompressed_tensor = bloom_decompressor(compressed_tensor, decompressed_size, hash_num=hash_num, bloom_size=bloom_size)

with tf.Session() as sess:
	print(sess.run(compressed_tensor))
	print(compressed_tensor.get_shape())
	print(sess.run(decompressed_tensor))
