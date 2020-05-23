import tensorflow as tf

indices = tf.constant([2, 4, 261, 262, 263])

rle_compressor = tf.load_op_library('./rle_compression0_8.so').rle_compressor_v0_code8
rle_decompressor = tf.load_op_library('./rle_compression0_8.so').rle_decompressor_v0_code8

# rle_compressor = tf.load_op_library('./rle_compression1_8.so').rle_compressor_v1_code8
# rle_decompressor = tf.load_op_library('./rle_compression1_8.so').rle_decompressor_v1_code8

# rle_compressor = tf.load_op_library('./rle_compression0_32.so').rle_compressor_v0_code32
# rle_decompressor = tf.load_op_library('./rle_compression0_32.so').rle_decompressor_v0_code32

# rle_compressor = tf.load_op_library('./rle_compression1_32.so').rle_compressor_v1_code32
# rle_decompressor = tf.load_op_library('./rle_compression1_32.so').rle_decompressor_v1_code32

step = tf.placeholder(tf.int64, name='step')
compressed_indices = rle_compressor(indices, 350, 1,
                                    logs_path="./logs",
                                    gradient_id=1,
                                    verbosity_frequency=1,
                                    verbosity=2,
                                    rank=1)

# tensor_compressed = tf.concat([values, compressed_indices], 0)

decompressed_tensor = rle_decompressor(compressed_indices, 5, 1,
                                       logs_path="./logs",
                                       gradient_id=1,
                                       verbosity_frequency=1,
                                       verbosity=2,
                                       suffix=0,
                                       rank=1)

with tf.Session() as sess:
    print("Indices: ", sess.run(indices))
    print("Tensor De-Compressed: ", sess.run(decompressed_tensor))
