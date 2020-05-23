from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
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
mapping = np.argsort(y_abs, axis=0)
sorted_Y = y_abs[mapping]

X = np.array(range(1, K+1), np.float64)
y_est_abs = coefficients[0] * np.exp(coefficients[2] * X) + coefficients[1] * np.exp(coefficients[3] * X)

top_values = sorted_Y[-K:]
indices = mapping[-K:]

negative_indices = np.where(np.less(Y[indices], 0))[0]
Kneg = negative_indices.size
mask = np.ones(K)
mask[negative_indices] = -np.ones(Kneg)
y = y_est_abs * mask
top_values = top_values * mask
# Compute Root Mean Squared Error
rmse = np.sqrt(np.sum(np.power(sorted_Y[-K:]-y_est_abs, 2)))

plt.rcParams["figure.figsize"] = [20, 10]
print(rmse)
with open(path+'/rmse.txt', 'w') as f:
        f.write(str(rmse) + "\n")
        mapping = mapping+1
        indices = indices+1
        plt.plot(range(1, N+1), Y, 'c.', markersize=1, label="All values")
        plt.plot(indices, top_values, 'co', markersize=8, label="True Top values")
        plt.plot(indices, y, 'mo', markersize=5, label="Estimated Values")
        plt.plot(range(1, N+1), sorted_Y, 'bo', markersize=6, label="True Sorted")
        plt.plot(range(N-K+1, N+1), y_est_abs, 'ro', markersize=6, label="Estimated Sorted")
        plt.legend()
        plt.savefig(path + 'gradient.png')