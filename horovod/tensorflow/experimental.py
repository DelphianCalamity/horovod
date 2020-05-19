
# class Stacked_Bloom_Filter_Compressor_Conflict_Sets(Compressor):
#     """"""
#
#     @staticmethod
#     def compress(tensor, params):
#
#         tensor_shape = tf.shape(tensor)
#         tensor_flatten = tf.reshape(tensor, [-1])
#         elemnum = tensor_flatten.get_shape().as_list()[0]
#
#         compress_ratio = params["compress_ratio"]
#         k = max(1, int(elemnum * compress_ratio))
#         params['topk'] = k
#         # Bloom filter size and number of hashes
#         # Default values
#         params['m'] = 10000
#         params['k'] = 3
#
#         # Configure bloom filter's m, k values
#         if params["bloom_size"] is not None:
#             params['m'] = params['bloom_size']
#         if params["hash_functions_number"] is not None:
#             params['k'] = params['hash_functions_number']
#         if params["fpr"] is not None:
#             # Given FPR compute M and H
#             m = (k * abs(math.log(params["fpr"]))) / (math.pow(math.log(2), 2))
#             params['m'] = m
#
#         quot = int(params['m']/8)
#         rem = params['m'] % 8
#         params['m'] = quot
#         if rem != 0:
#             params['m'] += 1
#
#         h = (params['m']*8 / k) * math.log(2)
#         params['k'] = int(math.ceil(h))
#         assert params['k'] < 256, "Number of hash functions too big"
#
#         k2 = math.ceil(params['fpr']*elemnum)   # Estimated number of false-positives to be insterted in the second bloom filter
#         params['m2'] = (k2 * abs(math.log(params["fpr2"]))) / (math.pow(math.log(2), 2))
#         quot = int(params['m2'] / 8)
#         rem = params['m2'] % 8
#         params['m2'] = quot
#         if rem != 0:
#             params['m2'] += 1
#         params['h2'] = int(math.ceil((params['m2'] * 8 / k2) * math.log(2)))
#         assert params['h2'] < 256, "Number of hash functions too big"
#
#         params["bloom_config"].add_data(k, params['m']*8, params['k'], params["fpr"])
#         params["throughput_info"].add_data(elemnum, elemnum/8,  (params['m']+params['m2'])*8,
#                                            ((params['m']+params['m2'])*8)/8, elemnum-(params['m']+params['m2'])*8,
#                                            (elemnum-(params['m']+params['m2'])*8)/8)
#
#         _, indices = tf.math.top_k(tf.math.abs(tensor_flatten), k, sorted=False)
#         indices = tf.sort(indices, axis=0, direction='ASCENDING')
#         values = tf.gather(tensor_flatten, indices)
#         values = tf.bitcast(values, tf.int32)
#
#         filename = resource_loader.get_path_to_datafile('mpi_lib.so')
#         library = load_library.load_op_library(filename)
#         bloom_compressor = library.stacked_bloom_compressor_conflict_sets
#
#         # For debugging
#         log_initial_tensor = tf.bitcast(tensor_flatten, tf.int32)
#         compressed_tensor = bloom_compressor(values, indices,
#                                              log_initial_tensor,
#                                              tf.train.get_or_create_global_step(),
#                                              hash_num=params['k'],
#                                              bloom_size=params['m'],
#                                              hash_num2=params['h2'],
#                                              bloom_size2=params['m2'],
#                                              logfile_suffix=params['logfile_suffix'],
#                                              logs_path_suffix=params['logs_path_suffix'],
#                                              verbosity=params['verbosity'])
#         ctx = tensor_shape
#         params['tensors_size_are_same'] = True
#         return compressed_tensor, ctx
#
#     @staticmethod
#     def decompress(compressed_tensor, ctx, params):
#         """Decompress by filling empty slots with zeros and reshape back using the original shape"""
#
#         tensor_shape = ctx
#         tensor_size = tf.math.reduce_prod(tensor_shape)
#
#         filename = resource_loader.get_path_to_datafile('mpi_lib.so')
#         library = load_library.load_op_library(filename)
#         bloom_decompressor = library.stacked_bloom_decompressor_conflict_sets
#
#         decompressed_tensor = bloom_decompressor(compressed_tensor, tensor_size,
#                                                  tf.train.get_or_create_global_step(), params['topk'],
#                                                  mem_mode=params['mem_mode'],
#                                                  hash_num=params['k'],
#                                                  bloom_size=params['m'],
#                                                  hash_num2=params['h2'],
#                                                  bloom_size2=params['m2'],
#                                                  logfile_suffix=params['logfile_suffix'],
#                                                  logs_path_suffix=params['logs_path_suffix'],
#                                                  suffix=params['suffix'],
#                                                  verbosity=params['verbosity'])
#
#         decompressed_tensor = tf.bitcast(decompressed_tensor, tf.float32)
#         decompressed_tensor = tf.reshape(decompressed_tensor, tensor_shape)
#         return decompressed_tensor

# class Bloom_Filter_Adaptive_Compressor(Compressor):
#
#     @staticmethod
#     def compress(tensor, params):
#
#         tensor_shape = tf.shape(tensor)
#         tensor_flatten = tf.reshape(tensor, [-1])
#         elemnum = tensor_flatten.get_shape().as_list()[0]
#         compress_ratio = params["compress_ratio"]
#         k = max(1, int(elemnum * compress_ratio))
#
#         params['m'] = 100000
#
#         if params["partitioning"] is None:
#             params["partitioning"] = 1
#         if params["bloom_size"] is not None:
#             params['m'] = params['bloom_size']
#         if params["fpr"] is not None:
#             m = (k * abs(math.log(params["fpr"]))) / (math.pow(math.log(2), 2))
#             params['m'] = m
#
#         # Give bloom size in number of bytes bloom size must be a multiple of 8
#         quot = int(params['m']/8)
#         rem = params['m'] % 8
#         params['m'] = quot
#         if rem != 0:
#             params['m'] += 1
#
#         params["bloom_config"].add_data(k, params['m']*8, None, params["fpr"])
#         params["throughput_info"].add_data(elemnum, elemnum/8,  params['m']*8, (params['m']*8)/8, elemnum-params['m']*8, (elemnum-params['m']*8)/8)
#
#         _, indices = tf.math.top_k(tf.math.abs(tensor_flatten), k, sorted=False)
#         indices = tf.sort(indices, axis=0, direction='ASCENDING')
#         values = tf.gather(tensor_flatten, indices)
#         values = tf.bitcast(values, tf.int32)
#
#         filename = resource_loader.get_path_to_datafile('mpi_lib.so')
#         library = load_library.load_op_library(filename)
#         bloom_adaptive_compressor = library.bloom_adaptive_compressor
#
#         log_initial_tensor = tf.bitcast(tensor_flatten, tf.int32)
#         compressed_tensor = bloom_adaptive_compressor(values, indices,
#                                              log_initial_tensor,
#                                              tf.train.get_or_create_global_step(),
#                                              partitioning=params['partitioning'],
#                                              bloom_size=params['m'],
#                                              logfile_suffix=params['logfile_suffix'],
#                                              logs_path_suffix=params['logs_path_suffix'],
#                                              verbosity=params['verbosity'])
#         ctx = tensor_shape
#         params['tensors_size_are_same'] = True
#         return compressed_tensor, ctx
#
#     @staticmethod
#     def decompress(compressed_tensor, ctx, params):
#         """Decompress by filling empty slots with zeros and reshape back using the original shape"""
#
#         tensor_shape = ctx
#         tensor_size = tf.math.reduce_prod(tensor_shape)
#
#         filename = resource_loader.get_path_to_datafile('mpi_lib.so')
#         library = load_library.load_op_library(filename)
#         bloom_adaptive_decompressor = library.bloom_adaptive_decompressor
#
#         decompressed_tensor = bloom_adaptive_decompressor(compressed_tensor, tensor_size,
#                                                  tf.train.get_or_create_global_step(),
#                                                  partitioning=params['partitioning'],
#                                                  bloom_size=params['m'],
#                                                  logfile_suffix=params['logfile_suffix'],
#                                                  logs_path_suffix=params['logs_path_suffix'],
#                                                  suffix=params['suffix'],
#                                                  verbosity=params['verbosity'])
#
#         decompressed_tensor = tf.bitcast(decompressed_tensor, tf.float32)
#         decompressed_tensor = tf.reshape(decompressed_tensor, tensor_shape)
#         return decompressed_tensor
