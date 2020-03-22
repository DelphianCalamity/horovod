import tensorflow as tf

init_tensor = tf.constant([-1.3, 10.2, 20.3, 2.5, 3.6])
k = 2
log_init_tensor = tf.bitcast(init_tensor, tf.int32)

_, indices = tf.math.top_k(tf.math.abs(init_tensor), k, sorted=False)
indices = tf.sort(indices, axis=0, direction='ASCENDING')
values = tf.gather(init_tensor, indices)
shape = tf.shape(values)

with tf.Session() as sess:
    print("Values Float32: ", sess.run(values))
    print("Values Float32 shape", sess.run(tf.shape(values)))
    values = tf.bitcast(values, tf.uint8)
    print("Values Int8: ", sess.run(values))
    shape = tf.shape(values)
    print("Values Float32 shape", sess.run(shape))
    flatten = tf.reshape(values, [-1])
    print("Values flattened Int8: ", sess.run(flatten))
    unflatten = tf.reshape(flatten, shape)
    print("Values Back Shape Int8: ", sess.run(unflatten))
    values_back = tf.bitcast(values, tf.float32)
    print("Values Back Float32: ", sess.run(values_back))

# # rle_compressor = tf.load_op_library('./rle_compression0_8.so').rle_compressor_v0_code8
# # rle_decompressor = tf.load_op_library('./rle_compression0_8.so').rle_decompressor_v0_code8
#
# # rle_compressor = tf.load_op_library('./rle_compression1_8.so').rle_compressor_v1_code8
# # rle_decompressor = tf.load_op_library('./rle_compression1_8.so').rle_decompressor_v1_code8
#
# # rle_compressor = tf.load_op_library('./rle_compression0_32.so').rle_compressor_v0_code32
# # rle_decompressor = tf.load_op_library('./rle_compression0_32.so').rle_decompressor_v0_code32
#
# rle_compressor = tf.load_op_library('./rle_compression1_32.so').rle_compressor_v1_code32
# rle_decompressor = tf.load_op_library('./rle_compression1_32.so').rle_decompressor_v1_code32
#
# step = tf.placeholder(tf.int64, name='step')
# compressed_indices = rle_compressor(indices, 350,
#                                     1,
#                                     logfile_suffix=1,
#                                     logs_path_suffix=1,
#                                     verbosity=1)
#
# # tensor_compressed = tf.concat([values, compressed_indices], 0)
#
# decompressed_tensor = rle_decompressor(compressed_indices, 5,
#                                        1,
#                                        logfile_suffix=1,
#                                        logs_path_suffix=1,
#                                        verbosity=1,
#                                        suffix=0)
#
# with tf.Session() as sess:
#     print("Indices: ", sess.run(indices))
#     print("Tensor De-Compressed: ", sess.run(decompressed_tensor))
