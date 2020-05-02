from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

tf.enable_eager_execution()
print(tf.executing_eagerly())

path = "logs/1/step_1/1/"
with open(path+'log.txt') as f:
    K = int(next(f).split()[0])
init_tensor = pd.read_csv(path+'values.csv', header=None, sep="\s+", names=["Y", "Yest"])

print(init_tensor)
Y = init_tensor['Y'].to_numpy()
Yest = init_tensor['Yest'].to_numpy()
N = Y.size
_, indices_Y = tf.math.top_k(tf.math.abs(Y), K, sorted=False)
_, indices_Yest = tf.math.top_k(tf.math.abs(Yest), K, sorted=False)
true_values = tf.gather(Y, indices_Y)
estimated_values = tf.gather(Yest, indices_Yest)

sorted_Y = tf.sort(Y, axis=0, direction='ASCENDING')
sorted_Yest = tf.sort(Yest, axis=0, direction='ASCENDING')
print(sorted_Y)
print(sorted_Yest)
plt.rcParams["figure.figsize"] = [20, 10]

plt.plot(range(1, N+1), Y, 'c.', markersize=2, label="True")
plt.plot(indices_Y.numpy(), true_values.numpy(), 'bo', markersize=2, label="Top " + str(K) + " True Values")

plt.plot(range(1, N+1), sorted_Y, 'bo', markersize=6, label="True Sorted")
plt.plot(range(1, N+1), sorted_Yest, 'ro', markersize=6, label="Estimated Sorted")

plt.legend()
plt.show()
plt.savefig(path+'gradient.png')
