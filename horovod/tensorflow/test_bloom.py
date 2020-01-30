import tensorflow as tf

values = tf.constant([10,20])
print(values.shape)
print(values.get_shape())

indices = tf.constant([1,2])
print(indices.shape)

bloom_module = tf.load_op_library('./bloom_compressor_op.so')
bloom_compressor = bloom_module.bloom_compressor


#bloom = bloom_compressor([1, 2], [3, 4])
bloom = bloom_compressor(values, indices)

with tf.Session() as sess:
#	print(bloom.get_shape())
	print(sess.run(bloom))
        print(bloom.get_shape())


# Prints
#array([[1, 0], [0, 0]], dtype=int32)
