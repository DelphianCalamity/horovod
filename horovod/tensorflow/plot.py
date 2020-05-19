from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import math
import csv

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
# parser.add_argument('--compress_ratio', type=float, default=0.01, help='compress ratio')
parser.add_argument('--path', type=str, default="./", help='')
args = parser.parse_args()

path = args.path
# compress_ratio = args.compress_ratio
init_tensor = pd.read_csv(path+'/values.csv', header=None, sep="\n")

csv = open(path+'/values.csv', "r").read()
rows = csv.split("\n")

Y = np.array([float(x) for x in rows[0].split(" ") if x != ''])
coefficients = np.array([float(x) for x in rows[1].split(" ") if x != ''])
# print(Y) ; print(coefficients)

N = Y.size

y_abs = tf.math.abs(Y)
mapping = tf.argsort(y_abs, axis=0, direction='ASCENDING', stable=False)
sorted_Y = tf.gather(y_abs, mapping)

X = np.array(range(1, N + 1), np.float64)
y_est_abs = coefficients[0] * tf.math.exp(coefficients[2] * X) + coefficients[1] * tf.math.exp(coefficients[3] * X)

negative_indices = tf.where(tf.less(tf.gather(Y, mapping), 0))
Nneg = tf.size(negative_indices)
mask = tf.tensor_scatter_nd_update(tf.ones([N], dtype=tf.int32), negative_indices, -tf.ones(Nneg, dtype=tf.int32))
y = y_est_abs * tf.cast(mask, tf.float64)


plt.rcParams["figure.figsize"] = [20, 10]

# Compute Root Mean Squared Error
# print(Y)
# print(mapping)
# print(tf.gather(y, mapping))
rmse = tf.math.sqrt(tf.reduce_sum(tf.math.pow(Y-tf.gather(y, mapping), 2)))
print(rmse)
with open(path+'/rmse.txt', 'w') as f:
    f.write(str(rmse) + "\n")

mapping = mapping+1
plt.plot(range(1, N+1), Y, 'c.', markersize=5, label="True")
plt.plot(mapping.numpy(), y, 'mo', markersize=5, label="Estimated Values")
plt.plot(range(1, N+1), sorted_Y, 'bo', markersize=6, label="True Sorted")
plt.plot(range(1, N+1), y_est_abs, 'ro', markersize=6, label="Estimated Sorted")
plt.legend()
# plt.show()
plt.savefig(path+'gradient.png')

