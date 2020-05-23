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

y_abs = np.abs(Y)
mapping = np.argsort(y_abs, axis=0)
sorted_Y = y_abs[mapping]

X = np.array(range(1, N+1), np.float64)

y_est_abs = coefficients[0] * np.exp(coefficients[2] * X) + coefficients[1] * np.exp(coefficients[3] * X)

negative_indices = np.where(np.less(Y[mapping], 0))[0]
Nneg = negative_indices.size
mask = np.ones(N)
mask[negative_indices] = -np.ones(Nneg)

y = y_est_abs * mask

# Compute Root Mean Squared Error
rmse = np.sqrt(np.sum(np.power(sorted_Y-y_est_abs, 2)))

plt.rcParams["figure.figsize"] = [20, 10]
print(rmse)
with open(path+'/rmse.txt', 'w') as f:
        f.write(str(rmse) + "\n")

        mapping = mapping+1
        plt.plot(range(1, N+1), Y, 'c.', markersize=5, label="True")
        plt.plot(mapping, y, 'mo', markersize=5, label="Estimated Values")
        plt.plot(range(1, N+1), sorted_Y, 'bo', markersize=6, label="True Sorted")
        plt.plot(range(1, N+1), y_est_abs, 'ro', markersize=6, label="Estimated Sorted")
        plt.legend()
        # plt.show()
        plt.savefig(path + 'gradient.png')