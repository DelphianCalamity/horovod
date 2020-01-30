import tensorflow as tf

bloom_module = tf.load_op_library('./bloom_compressor_op.so')
bloom_compressor = bloom_module.bloom_compressor

with tf.Session() as sess:
	print(sess.run(bloom_compressor([1, 2], [3, 4])))

# Prints
#array([[1, 0], [0, 0]], dtype=int32)
