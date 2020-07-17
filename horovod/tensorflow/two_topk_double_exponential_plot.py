from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import tensorflow as tf
import math
import csv

pd.set_option("display.precision", 40)
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="./", help='')
args = parser.parse_args()

path = args.path
Y = pd.read_csv(path+'/values.csv', header=None, sep="\n")[0].values
coefficients = pd.read_csv(path+'/coefficients.csv', header=None, sep="\n")[0].values
#print(Y) ; print(coefficients)
N = Y.size
compress_ratio = 0.2
K = max(1, int(N*compress_ratio))  # If compress ratio is set to 1 then K=N

y_abs = np.abs(Y)
indices = np.argsort(y_abs, axis=0)[-K:]
top_values = Y[indices]

negative_indices = np.where(np.less(Y[indices], 0))[0]
Kneg = negative_indices.size ; Kpos = K-Kneg

Xneg = np.array(range(1, Kneg+1), np.float64)
Xpos = np.array(range(1, Kpos+1), np.float64)

y_estimates_neg = coefficients[0] * np.exp(coefficients[2] * Xneg) + \
                  coefficients[1] * np.exp(coefficients[3] * Xneg)
y_estimates_pos = coefficients[4] * np.exp(coefficients[6] * Xpos) + \
                  coefficients[5] * np.exp(coefficients[7] * Xpos)
y = np.concatenate([y_estimates_neg, y_estimates_pos], axis=0)

sorted_mapping = np.argsort(top_values, axis=0)
values = top_values[sorted_mapping]
mapping = indices[sorted_mapping]

# Compute Root Mean Squared Error
rmse = np.sqrt(np.sum(np.power(values-y, 2)))

plt.rcParams["figure.figsize"] = [20, 10]
print(rmse)
with open(path+'/rmse.txt', 'w') as f:
        f.write(str(rmse) + "\n")
        mapping = mapping+1
        indices = indices+1
        plt.plot(range(1, N+1), Y, 'c.', markersize=1, label="All values")
        plt.plot(range(1, N + 1), np.sort(Y, axis=0), 'bo', markersize=6, label="All values Sorted")
        plt.plot(indices, top_values, 'co', markersize=8, label="True Top values")

        plt.plot(mapping, y, 'mo', markersize=5, label="Estimated Values")

        plt.plot(range(1, Kneg + 1), y_estimates_neg, 'r.', markersize=8, label="Estimated Sorted Negative")
        plt.plot(range(N-Kpos, N), y_estimates_pos, 'g.', markersize=8, label="Estimated Sorted Positive")

        plt.legend()
        plt.savefig(path + 'gradient.png')