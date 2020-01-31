import tensorflow as tf

# Assuming Intiial tensor = [1, 10, 20, 2, 3]
values = tf.constant([10,20])
print(values.shape)
print(values.get_shape())

indices = tf.constant([1,2])
print(indices.shape)

bloom_compressor = tf.load_op_library('./bloom_compressor_op.so').bloom_compressor
bloom_decompressor = tf.load_op_library('./bloom_decompressor_op.so').bloom_decompressor

compressed_tensor = bloom_compressor(values, indices)
decompressed_tensor = bloom_decompressor(compressed_tensor, 5)



with tf.Session() as sess:
	print(sess.run(compressed_tensor))
	print(compressed_tensor.get_shape())
	print(sess.run(decompressed_tensor))


# Prints
#array([[1, 0], [0, 0]], dtype=int32)
