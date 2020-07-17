from __future__ import division
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import load_library
import tensorflow as tf
import numpy as np
import math


def double_exponential_fit(X_, Y_, K):
	# S, SS initialization
	Ysum = Y_ + tf.roll(Y_, shift=-1, axis=0)
	Xsum = tf.roll(X_, shift=-1, axis=0) - X_
	S = tf.tensor_scatter_nd_update(tf.roll(0.5 * Ysum * Xsum, shift=1, axis=0), [[0]], tf.zeros(1, tf.float64))
	S = tf.math.cumsum(S)
	Ssum = S + tf.roll(S, shift=-1, axis=0)
	SS = tf.tensor_scatter_nd_update(tf.roll(0.5 * Ssum * Xsum, shift=1, axis=0), [[0]], tf.zeros(1, tf.float64))
	SS = tf.math.cumsum(SS)

	sum_SSk_squared = tf.math.reduce_sum(tf.math.pow(SS, 2))
	sum_SSk_Sk = tf.math.reduce_sum(S * SS)
	sum_SSk_xk = tf.math.reduce_sum(SS * X_)
	sum_SSk = tf.math.reduce_sum(SS)
	sum_Sk_squared = tf.math.reduce_sum(tf.math.pow(S, 2))
	sum_Sk_xk = tf.math.reduce_sum(S * X_)
	sum_Sk = tf.math.reduce_sum(S)
	sum_data_x = tf.cast(K * (K + 1) / 2, tf.float64)
	sum_data_x_squared = tf.cast(K * (K + 1) * (2 * K + 1) / 6, tf.float64)
	K = tf.cast(K, tf.float64)

	# Form the first system
	values = tf.stack([sum_SSk_squared, sum_Sk_squared, sum_data_x_squared, K,
					   sum_SSk_Sk, sum_SSk_xk, sum_SSk, sum_Sk_xk, sum_Sk, sum_data_x], axis=0)

	A_LS_1 = tf.scatter_nd([[0, 0], [1, 1], [2, 2], [3, 3],
							[0, 1], [0, 2], [0, 3],
							[1, 2], [1, 3],
							[2, 3]],
						   values, [4, 4])
	A_LS_1 = tf.tensor_scatter_nd_update(A_LS_1,
										 [[0, 0], [1, 1], [2, 2], [3, 3],
										  [1, 0], [2, 0], [3, 0],
										  [2, 1], [3, 1],
										  [3, 2]],
										 values)

	a = tf.math.reduce_sum(tf.transpose(SS) * Y_)
	b = tf.math.reduce_sum(tf.transpose(S) * Y_)
	c = tf.math.reduce_sum(tf.transpose(X_) * Y_)
	d = tf.math.reduce_sum(Y_)

	b_vector_1 = tf.stack([a, b, c, d], axis=0)
	b_vector_1 = tf.reshape(b_vector_1, [4, 1])

	# Solve the first system
	Coefficient_vector_1 = tf.linalg.solve(A_LS_1, b_vector_1)

	# Calculate p1 and q1
	p1 = 0.5 * (Coefficient_vector_1[1] + tf.math.sqrt(
		tf.math.pow(Coefficient_vector_1[1], 2) + 4 * Coefficient_vector_1[0]))
	q1 = 0.5 * (Coefficient_vector_1[1] - tf.math.sqrt(
		tf.math.pow(Coefficient_vector_1[1], 2) + 4 * Coefficient_vector_1[0]))

	beta_k = tf.math.exp(p1 * X_)
	eta_k = tf.math.exp(q1 * X_)

	sum_betak_square = tf.math.reduce_sum(tf.math.pow(beta_k, 2))
	sum_etak_square = tf.math.reduce_sum(tf.math.pow(eta_k, 2))
	sum_betak_etak = tf.math.reduce_sum(beta_k * eta_k)

	# Form the second system
	A_LS_2 = tf.stack([sum_betak_square, sum_betak_etak, sum_betak_etak, sum_etak_square], axis=0)
	A_LS_2 = tf.reshape(A_LS_2, [2, 2])
	a = tf.reshape(tf.math.reduce_sum(tf.transpose(beta_k) * Y_), [1, ])
	b = tf.reshape(tf.math.reduce_sum(tf.transpose(eta_k) * Y_), [1, ])
	b_vector_2 = tf.stack([a, b], axis=0)
	b_vector_2 = tf.reshape(b_vector_2, [2, 1])

	# Solve the second system
	Coefficient_vector_2 = tf.linalg.solve(A_LS_2, b_vector_2)

	# print("Coefficient_vector_1: \n", Coefficient_vector_1)
	# print("p1:\n", p1)
	# print("Coefficient_vector_2:\n", Coefficient_vector_2)
	# print("q1:\n", q1)
	return Coefficient_vector_2[0], Coefficient_vector_2[1], p1, q1

# init_tensor = tf.constant([1001874297, 1007309995, 943642542, 982145325, -1158035745, 969390216, 995928611, 977786471, -1164039257, -1150973382, -1145765814, -1153600937, 999516240, 983204270, -1151637699, -1180861488])
init_tensor = tf.constant([-1126533733, 982072357, -1151693744, -1151799612, 979173021, 980033213, -1147788336, -1145933480, 997509220, 1005356913, 1000662654, -1145981561, 1009131314, 1013736002, -1144151260, -1161414024, -1121588285, 969565401, 982460597, 975781123, 999037901, 1002380139, -1145410715, -1169082620, 994360572, 985236736, 1000095879, -1140284814, 1010632429, 1012291774, -1147726467, 986810251, -1117059335, 999905654, 1011638759, 994825786, -1156928141, 992212962, -1151006545, -1127576615, -1154500315, 994602528, -1156217968, -1131803534, 1012419592, 998253235, -1145771675, 959178848, -1126298849, 989651225, -1151614300, -1156174938, -1162765643, 991720814, -1146040339, -1169667435, 982759884, 1003211792, 1015268569, -1140075638, 985892531, 1013155001, -1148553619, -1175406627, -1121409942, 973757045, -1203086768, 990980520, 998866484, 1003522222, -1142597662, 996772265, 979636318, 984862398, 1009132710, -1138258285, 999154184, 1011641198, -1157040960, 995007145, -1117251847, 996335161, 1011890653, 999177172, -1166706310, 995550796, -1146612531, -1129201078, -1146045828, 993871566, -1174186688, -1130309931, 1001725190, 1003436762, -1153106746, 989880701, -1133017603, 962943544, -1149077074, -1159995596, 991691607, -1159484834, -1150890335, -1157007842, -1151319583, 1008373426, 1020102576, -1139105582, -1146686254, 1010426736, -1159859652, -1160560132, -1125240474, -1164423079, -1160753390, 994104677, 1005476013, 976244874, -1145491128, 1000085638, -1152885505, 1001244494, 1015285274, -1138066736, -1155232965, 1005979214, 935783675, 992983852, -1120480788, 989275588, 1010267614, 1001450197, 990731987, -1147677258, -1145352785, -1129978074, -1139141756, 1004529620, 998392867, -1130649027, -1173162403, -1174210252, -1172234369, 964532838, -1129052713, -1166103702, 991190611, -1151141822, 996943568, 1009915717, -1135552208, -1138734367, -1159194651, 995747053, 1000069764, -1188506692, 1007330131, 1017350326, -1141490196, 993319022, -1123008698, -1155228969, 1005920723, 988503726, 1002905132, 1015723221, -1130324380, -1145258188, 965429420, -1155576348, 1004544600, -1150787161, 1007852822, 1016342642, -1145788589, 1002284184, -1118864377, 991190461, 1016297539, 998533674, 916429824, 1010914901, -1132189112, -1126779691, -1146021846, -1200801575, -1153533848, -1133154948, 1008038565, 1008970593, -1145416848, 998015001, -1129636835, 976847877, 968558199, -1171641755, 987703559, 1007095124, -1139515645, -1148443985, -1162293438, 986674647, 1014564418, -1144150992, 990906049, 1008236491, -1145826548, -1202638281, -1123308202, -1161944374, 1001808562, 998633256, 1005342286, 1013403587, -1134484886, 973230655, 980216594, -1152873077, 1011530293, -1139990067, 996820299, 1006650244, -1151565128, 996469801, -1119213502, 985461935, 1015478455, 1002669451, 978402811, 1007939424, -1137280799, -1130004142, -1146327506, -1172235917, -1170550102, -1130061572, 999428294, 999590365, -1148101665, 990827068, -1137894593, 980502106, -1161708846, 981485447, 992353276, 1003476411, -1137980671, 972389035, 983096196, 1003116440, 1017835283, -1138721632, -1172593771, 994102189, 939968918, -1147561885, -1127962506, -1171152581, 990375925, 1001039219, 1006937238, 1007823664, -1134675230, 1005212524, 988505456, 986247085, 1015266023, -1136582198, 979271204, -1190955327, 964372503, -1163687860, -1122856712, 990513697, 1011800363, 1005011003, 986533820, 993457445, -1133782828, -1134089674, -1147726908, 998693905, 992827633, -1129594183, 986929134, -1147401615, -1171060356, -1151590259, -1125635699, -1169313664, 1001647082, -1160941992, 994848972, 1019386517, -1151168869, -1137782435, 982510152, -1192790487, 954427181, -1174489039, 1007346708, 1016142052, -1143087760, 991097011, -1120805574, -1198243061, 1009465958, 991885044, 1000442416, 1021081881, -1133562503, -1143171429, 997297891, -1144005541, 983050751, -1156618346, 1008684557, 1012759070, -1147266081, 999665721, -1117175081, 998485110, 1017428341, 999035952, -1172014909, 1017041906, -1137046009, -1127468561, -1153160868, -1156004266, -1137623674, -1136415310, 1007844330, 1003878006, -1150118175, 995617410, -1130270151, 981756568, -1156019298, 995393868, 982068429, 1015650165, -1153964987, -1142843996, 981582754, -1157303886, -1192357249, -1144567588, 1008099399, 999076015, -1153011896, -1157347218, -1123406675, 970181214, 995808227, 1004447278, 1002110327, 1018176651, -1139468155, -1154910509, 998851237, -1140622010, -1155725815, -1142680796, 1009395038, 998218371, -1160057353, 986330384, -1119925782, 999989209, 1013917606, 1007135058, -1173440047, 1013723982, -1142429091, -1129725202, -1152782039, -1148458948, -1132065579, -1131648014, 1009122477, -1169406711, -1159914243, 977083449, -1132796989, -1161723275, -1146794068, 1003117239, -1164141584, 1013647377, -1156069204, -1176413429, 990286979, 971127814, 1006431541, -1140487577, 999026657, -1150361667, 977747024, -1149264997, -1126006738, -1160990299, -1188600129, 1009066087, 998545362, 1016019138, -1145678536, 1001281090, 998912201, -1146848728, 999751973, -1139419456, 1000803713, -1150481559, 985176027, -1173107144, -1122308136, 990885197, 1009692101, 1010210299, -1153736783, 1009434208, -1141512110, -1134760825, -1152369539, -1163492520, -1137772942, -1131504896, 999699417, -1137039407, 981117333, -1157348275])
tensor = tf.bitcast(init_tensor, tf.float32)
params = {}
params["compress_ratio"] = 0.2



def compress(tensor, params):
	tensor_shape = tf.shape(tensor)
	tensor_flatten = tf.reshape(tensor, [-1])
	N = tensor_flatten.get_shape().as_list()[0]
	compress_ratio = params["compress_ratio"]
	K = max(1, int(N * compress_ratio))  # If compress ratio is set to 1 then K=N
	params['N'] = int(N);
	params['K'] = K
	print("Tensor", tensor, "size:", params['N'])

	abs_values = tf.math.abs(tensor_flatten)

	_, mapping = tf.math.top_k(abs_values, K, sorted=False)
	top_values = tf.gather(tensor_flatten, mapping)

	sorted_mapping = tf.argsort(top_values, axis=0, direction='ASCENDING')
	values = tf.gather(top_values, sorted_mapping)
	mapping = tf.gather(mapping, sorted_mapping)

	# Indices have a negative sign if they correspond to negative values and positive otherwise
	negative_indices = tf.where(tf.less(tf.gather(tensor_flatten, mapping), 0))
	Kneg = tf.size(negative_indices);
	Kpos = K - Kneg
	mask = tf.tensor_scatter_nd_update(tf.ones([K], dtype=tf.int32), negative_indices,
									   -tf.ones(Kneg, dtype=tf.int32))
	mapping = (mapping + 1) * mask

	# Fitting the curve of Negatives
	Xneg = tf.cast(tf.range(1, Kneg + 1), tf.float64)
	neg_coefficients = double_exponential_fit(Xneg, tf.cast(
		tf.gather(values, tf.range(0, Kneg)), tf.float64), Kneg)
	neg_num_of_coefficients = len(neg_coefficients)
	neg_coefficients = tf.reshape(neg_coefficients, [-1])

	# Fitting the curve of Positives
	Xpos = tf.cast(tf.range(1, Kpos + 1), tf.float64)
	pos_coefficients = double_exponential_fit(Xpos, tf.cast(
		tf.gather(values, tf.range(K - Kpos, K)), tf.float64), Kpos)
	pos_num_of_coefficients = len(pos_coefficients)
	pos_coefficients = tf.reshape(pos_coefficients, [-1])

	##################### Logging #####################
	coefficients = tf.concat([neg_coefficients, pos_coefficients], 0)

	filename = resource_loader.get_path_to_datafile('./logger.so')
	library = load_library.load_op_library(filename)
	logger = library.logger
	logger = logger(tensor_flatten, tf.cast(coefficients, tf.float64), 0,
					bloom_logs_path="./logs",
					gradient_id=1,
					verbosity_frequency=1,
					verbosity=2,
					rank=1)
	##################### /Logging #####################

	compressed_indices = mapping  # Possible indices compression here
	with tf.control_dependencies([logger]):
		# coefficients = tf.reshape(coefficients, [-1])
		compressed_indices = tf.cast(compressed_indices, tf.float64)
		tensor_compressed = tf.concat([coefficients, compressed_indices], 0)
		params['message_size'] = neg_num_of_coefficients + pos_num_of_coefficients
		params['Xneg'] = Xneg
		params['Xpos'] = Xpos

	ctx = tensor_shape
	params['tensors_size_are_same'] = True
	return tensor_compressed, ctx


def decompress(tensor_compressed, ctx, params):
	tensor_shape = ctx
	tensor_compressed_size = tf.math.reduce_prod(tf.shape(tensor_compressed))
	message, indices = tf.split(tensor_compressed,
								[params['message_size'], tensor_compressed_size - params['message_size']])
	decompressed_indices = tf.cast(indices, tf.int32)

	negative_indices = tf.where(tf.less(decompressed_indices, 0))
	decompressed_indices = tf.math.abs(decompressed_indices)
	decompressed_indices = decompressed_indices - 1

	message_neg, message_pos = tf.split(message, 2)

	y_estimates_neg = message_neg[0] * tf.math.exp(message_neg[2] * params['Xneg']) + \
					  message_neg[1] * tf.math.exp(message_neg[3] * params['Xneg'])

	y_estimates_pos = message_pos[0] * tf.math.exp(message_pos[2] * params['Xpos']) + \
					  message_pos[1] * tf.math.exp(message_pos[3] * params['Xpos'])

	y_estimates = tf.concat([y_estimates_neg, y_estimates_pos], 0)

	Kneg = tf.size(negative_indices)
	mask = tf.tensor_scatter_nd_update(tf.ones([params['K']], dtype=tf.int32), negative_indices,
									   -tf.ones(Kneg, dtype=tf.int32))
	y_estimates = y_estimates * tf.cast(mask, tf.float64)
	values = tf.cast(tf.reshape(y_estimates, [-1]), tf.float32)

	decompressed_indices = tf.expand_dims(decompressed_indices, 1)
	tensor_decompressed = tf.scatter_nd(decompressed_indices, values, [params['N']])
	tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)

	return tensor_decompressed

tensor_compressed, ctx = compress(tensor, params)
tensor_decompressed = decompress(tensor_compressed, ctx, params)


with tf.Session() as sess:
	print("Initial Tensor: ", sess.run(tensor))
	# sess.run(compressed_tensor, feed_dict={step:0})
	# print("Compressed Tensor Shape: ", compressed_tensor.get_shape())
	sess.run(tensor_decompressed)
