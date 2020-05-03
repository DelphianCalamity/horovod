from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

tf.enable_eager_execution()
print(tf.executing_eagerly())

path_prefix = "logs/"
path = path_prefix + "1/step_1/1/"
with open(path+'log.txt') as f:
    K = int(next(f).split()[0])
init_tensor = pd.read_csv(path+'values.csv', header=None, sep="\s+", names=["Y", "Yest"])

# print(init_tensor)
Y = init_tensor['Y'].to_numpy()
Yest = init_tensor['Yest'].to_numpy()
N = Y.size
_, indices_Y = tf.math.top_k(tf.math.abs(Y), K, sorted=False)
_, indices_Yest = tf.math.top_k(tf.math.abs(Yest), K, sorted=False)

true_values = tf.gather(Y, indices_Y)
estimated_values = tf.gather(Yest, indices_Yest)

mapping = tf.argsort(Y, axis=0, direction='ASCENDING', stable=False)
# print(mapping)
sorted_Y = tf.gather(Y, mapping)
mapped_estimated_indices = tf.gather(mapping, indices_Yest)
sorted_Yest = tf.sort(Yest, axis=0, direction='ASCENDING')
print(indices_Y)
print(mapped_estimated_indices)
plt.rcParams["figure.figsize"] = [20, 10]

# Compute Root Mean Squared Error
true_values_sent = tf.gather(Y, mapped_estimated_indices)
rmse = tf.math.sqrt(tf.reduce_sum(tf.math.pow(true_values_sent-estimated_values, 2)))
print(rmse)
with open(path_prefix+'rmse.txt', 'a') as f:
    f.write(str(rmse) + "\n")

non_topk_errors = np.where(indices_Y.numpy() != mapped_estimated_indices.numpy())
with open(path_prefix+'non-topk-errors.txt', 'a') as f:
    f.write(str(non_topk_errors[0].size) + "\n")


plt.plot(range(1, N+1), Y, 'c.', markersize=2, label="True")
plt.plot(indices_Y.numpy(), true_values.numpy(), 'ko', markersize=5, label="Top " + str(K) + " True Values")
plt.plot(mapped_estimated_indices.numpy(), estimated_values.numpy(), 'mo', markersize=5, label="Top " + str(K) + " Estimated Values")
plt.plot(range(1, N+1), sorted_Y, 'bo', markersize=6, label="True Sorted")
plt.plot(range(1, N+1), sorted_Yest, 'ro', markersize=6, label="Estimated Sorted")
plt.legend()
# plt.show()
plt.savefig(path+'gradient.png')
