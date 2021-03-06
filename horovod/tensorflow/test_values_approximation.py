from __future__ import division
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import load_library
import tensorflow as tf
import numpy as np
import math


def logit_basis(X, a, N):  # log(p/(1-p))
	return tf.cast(a * tf.math.log(X / ((N + 1) - X)), dtype=tf.float32)


def exp_basis(X, b, c):
	return tf.cast(b * tf.math.exp(c * X), dtype=tf.float32)


def GetInputMatrix(x, p0, N):
	Xtrans = tf.ones([1, N], tf.float32)  # [np.ones(N)] #{1}
	for [a, b, c] in p0:
		basis = logit_basis(x, a, N)
		Xtrans = tf.concat([Xtrans, basis], axis=0)
		basis = exp_basis(x, b, c)
		Xtrans = tf.concat([Xtrans, basis], axis=0)
	return tf.transpose(Xtrans)


def LeastSquares(X, y):  # returns (X'X)^-1 X'y
	Xtrans = tf.transpose(X)
	tmp = tf.matmul(Xtrans, X)
	inverse = tf.linalg.inv(tmp)
	theta_estimates = tf.matmul(tf.matmul(inverse, Xtrans), y)
	return theta_estimates


# init_tensor = tf.constant([1001874297, 1007309995, 943642542, 982145325, -1158035745, 969390216, 995928611, 977786471, -1164039257, -1150973382, -1145765814, -1153600937, 999516240, 983204270, -1151637699, -1180861488])
init_tensor = tf.constant([-1126533733, 982072357, -1151693744, -1151799612, 979173021, 980033213, -1147788336, -1145933480, 997509220, 1005356913, 1000662654, -1145981561, 1009131314, 1013736002, -1144151260, -1161414024, -1121588285, 969565401, 982460597, 975781123, 999037901, 1002380139, -1145410715, -1169082620, 994360572, 985236736, 1000095879, -1140284814, 1010632429, 1012291774, -1147726467, 986810251, -1117059335, 999905654, 1011638759, 994825786, -1156928141, 992212962, -1151006545, -1127576615, -1154500315, 994602528, -1156217968, -1131803534, 1012419592, 998253235, -1145771675, 959178848, -1126298849, 989651225, -1151614300, -1156174938, -1162765643, 991720814, -1146040339, -1169667435, 982759884, 1003211792, 1015268569, -1140075638, 985892531, 1013155001, -1148553619, -1175406627, -1121409942, 973757045, -1203086768, 990980520, 998866484, 1003522222, -1142597662, 996772265, 979636318, 984862398, 1009132710, -1138258285, 999154184, 1011641198, -1157040960, 995007145, -1117251847, 996335161, 1011890653, 999177172, -1166706310, 995550796, -1146612531, -1129201078, -1146045828, 993871566, -1174186688, -1130309931, 1001725190, 1003436762, -1153106746, 989880701, -1133017603, 962943544, -1149077074, -1159995596, 991691607, -1159484834, -1150890335, -1157007842, -1151319583, 1008373426, 1020102576, -1139105582, -1146686254, 1010426736, -1159859652, -1160560132, -1125240474, -1164423079, -1160753390, 994104677, 1005476013, 976244874, -1145491128, 1000085638, -1152885505, 1001244494, 1015285274, -1138066736, -1155232965, 1005979214, 935783675, 992983852, -1120480788, 989275588, 1010267614, 1001450197, 990731987, -1147677258, -1145352785, -1129978074, -1139141756, 1004529620, 998392867, -1130649027, -1173162403, -1174210252, -1172234369, 964532838, -1129052713, -1166103702, 991190611, -1151141822, 996943568, 1009915717, -1135552208, -1138734367, -1159194651, 995747053, 1000069764, -1188506692, 1007330131, 1017350326, -1141490196, 993319022, -1123008698, -1155228969, 1005920723, 988503726, 1002905132, 1015723221, -1130324380, -1145258188, 965429420, -1155576348, 1004544600, -1150787161, 1007852822, 1016342642, -1145788589, 1002284184, -1118864377, 991190461, 1016297539, 998533674, 916429824, 1010914901, -1132189112, -1126779691, -1146021846, -1200801575, -1153533848, -1133154948, 1008038565, 1008970593, -1145416848, 998015001, -1129636835, 976847877, 968558199, -1171641755, 987703559, 1007095124, -1139515645, -1148443985, -1162293438, 986674647, 1014564418, -1144150992, 990906049, 1008236491, -1145826548, -1202638281, -1123308202, -1161944374, 1001808562, 998633256, 1005342286, 1013403587, -1134484886, 973230655, 980216594, -1152873077, 1011530293, -1139990067, 996820299, 1006650244, -1151565128, 996469801, -1119213502, 985461935, 1015478455, 1002669451, 978402811, 1007939424, -1137280799, -1130004142, -1146327506, -1172235917, -1170550102, -1130061572, 999428294, 999590365, -1148101665, 990827068, -1137894593, 980502106, -1161708846, 981485447, 992353276, 1003476411, -1137980671, 972389035, 983096196, 1003116440, 1017835283, -1138721632, -1172593771, 994102189, 939968918, -1147561885, -1127962506, -1171152581, 990375925, 1001039219, 1006937238, 1007823664, -1134675230, 1005212524, 988505456, 986247085, 1015266023, -1136582198, 979271204, -1190955327, 964372503, -1163687860, -1122856712, 990513697, 1011800363, 1005011003, 986533820, 993457445, -1133782828, -1134089674, -1147726908, 998693905, 992827633, -1129594183, 986929134, -1147401615, -1171060356, -1151590259, -1125635699, -1169313664, 1001647082, -1160941992, 994848972, 1019386517, -1151168869, -1137782435, 982510152, -1192790487, 954427181, -1174489039, 1007346708, 1016142052, -1143087760, 991097011, -1120805574, -1198243061, 1009465958, 991885044, 1000442416, 1021081881, -1133562503, -1143171429, 997297891, -1144005541, 983050751, -1156618346, 1008684557, 1012759070, -1147266081, 999665721, -1117175081, 998485110, 1017428341, 999035952, -1172014909, 1017041906, -1137046009, -1127468561, -1153160868, -1156004266, -1137623674, -1136415310, 1007844330, 1003878006, -1150118175, 995617410, -1130270151, 981756568, -1156019298, 995393868, 982068429, 1015650165, -1153964987, -1142843996, 981582754, -1157303886, -1192357249, -1144567588, 1008099399, 999076015, -1153011896, -1157347218, -1123406675, 970181214, 995808227, 1004447278, 1002110327, 1018176651, -1139468155, -1154910509, 998851237, -1140622010, -1155725815, -1142680796, 1009395038, 998218371, -1160057353, 986330384, -1119925782, 999989209, 1013917606, 1007135058, -1173440047, 1013723982, -1142429091, -1129725202, -1152782039, -1148458948, -1132065579, -1131648014, 1009122477, -1169406711, -1159914243, 977083449, -1132796989, -1161723275, -1146794068, 1003117239, -1164141584, 1013647377, -1156069204, -1176413429, 990286979, 971127814, 1006431541, -1140487577, 999026657, -1150361667, 977747024, -1149264997, -1126006738, -1160990299, -1188600129, 1009066087, 998545362, 1016019138, -1145678536, 1001281090, 998912201, -1146848728, 999751973, -1139419456, 1000803713, -1150481559, 985176027, -1173107144, -1122308136, 990885197, 1009692101, 1010210299, -1153736783, 1009434208, -1141512110, -1134760825, -1152369539, -1163492520, -1137772942, -1131504896, 999699417, -1137039407, 981117333, -1157348275])
tensor = tf.bitcast(init_tensor, tf.float32)

params = {}
params["compress_ratio"] = 0.01

def compress(tensor, params):

	tensor_shape = tf.shape(tensor)
	tensor_flatten = tf.reshape(tensor, [-1])
	N = tensor_flatten.get_shape().as_list()[0]
	compress_ratio = params["compress_ratio"]
	k = max(1, int(N * compress_ratio))

	if k > 3:
		p0 = [[0.004, -0.01, -0.04]]
		num_of_coefficients = len(p0[0])
		x_train = np.array(range(1, N + 1), np.int32).reshape([1, N])
		mapping = tf.argsort(tensor_flatten, axis=0, direction='ASCENDING', stable=False)
		y_train = tf.gather(tensor_flatten, mapping)
		y_train = tf.reshape(y_train, [N, 1])

		X_train = GetInputMatrix(x_train, p0, N)
		theta_estimates = LeastSquares(X_train, y_train)
		y_estimates = tf.matmul(X_train, theta_estimates)
		y_estimates = tf.reshape(y_estimates, [-1])
		_, estimated_indices = tf.math.top_k(tf.math.abs(y_estimates), k, sorted=False)
		mapped_estimated_indices = tf.gather(mapping, estimated_indices)

		##################### Logging #####################
		filename = resource_loader.get_path_to_datafile('./logger.so')
		library = load_library.load_op_library(filename)
		logger = library.logger
		logger = logger(tensor_flatten, tf.cast(theta_estimates, tf.float64),
						0,
						bloom_logs_path="./logs",
						gradient_id=1,
						verbosity_frequency=1,
						verbosity=2,
						rank=1)
		##################### / Logging #####################

		compressed_indices = mapped_estimated_indices

		with tf.control_dependencies([logger]):
			theta_estimates = tf.bitcast(theta_estimates, tf.int32)
		theta_shape = tf.shape(theta_estimates)
		theta_estimates = tf.reshape(theta_estimates, [-1])
		tensor_compressed = tf.concat([theta_estimates, compressed_indices], 0)
		ctx = [tensor_shape, theta_shape]
		params['message_size'] = num_of_coefficients
		params['X_train'] = X_train
		params['p0'] = p0

	else:
		_, indices = tf.math.top_k(tf.math.abs(tensor_flatten), k, sorted=False)
		indices = tf.sort(indices, axis=0, direction='ASCENDING')
		values = tf.gather(tensor_flatten, indices)
		compressed_indices = indices
		values = tf.bitcast(values, tf.int32)
		values_shape = tf.shape(values)
		# theta_estimates = tf.reshape(theta_estimates, [-1])
		tensor_compressed = tf.concat([values, compressed_indices], 0)
		ctx = [tensor_shape, values_shape]
		params['message_size'] = k

	params['tensors_size_are_same'] = True
	params['topk_k'] = k
	return tensor_compressed, ctx


def decompress(tensor_compressed, ctx, params):

	compressed_tensor_size = tf.math.reduce_prod(tf.shape(tensor_compressed))
	message, indices = tf.split(tensor_compressed, [params['message_size'], compressed_tensor_size - params['message_size']])
	message = tf.bitcast(message, tf.float32)
	message = tf.reshape(message, ctx[1])
	tensor_shape = ctx[0]
	N = tf.math.reduce_prod(tensor_shape)
	decompressed_indices = indices

	if params['topk_k'] > 3:
		y_estimates = tf.matmul(params['X_train'], message)
		y_estimates = tf.reshape(y_estimates, [-1])
		_, estimated_indices = tf.math.top_k(tf.math.abs(y_estimates), params['topk_k'], sorted=False)
		values = tf.gather(y_estimates, estimated_indices)

	else:
		values = message

	zero_tensor = tf.Variable(tf.zeros([N], dtype=tf.float32), trainable=False)
	op = zero_tensor.assign(tf.zeros([N], dtype=tf.float32))
	with tf.control_dependencies([op]):
		tensor_decompressed = tf.scatter_update(zero_tensor, decompressed_indices, values)
	tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
	return tensor_decompressed







tensor_compressed, ctx = compress(tensor, params)
tensor_decompressed = decompress(tensor_compressed, ctx, params)


with tf.Session() as sess:
	print("Initial Tensor: ", sess.run(tensor))
	# sess.run(compressed_tensor, feed_dict={step:0})
	# print("Compressed Tensor Shape: ", compressed_tensor.get_shape())
	sess.run(tensor_decompressed)
