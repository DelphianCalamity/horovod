from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import itertools

def find_breaks(curve, num_of_segments=2):
    y=curve
    breaks = []
    break_index = 0
    breaks.append(break_index)
    for i in range(num_of_segments):
        line = np.linspace(y[0], y[-1], len(y))
        distance = list(np.abs(line - y))
        break_index += distance.index(max(distance))
        breaks.append(break_index)
        y=curve[break_index:]
    breaks.append(len(curve))
    return breaks

def GetInputMatrix_Polynomial(xcol, x):
    N = len(x)
    Xtrans = [np.ones(N)]
    for i in range(1, xcol):
        Xtrans = np.vstack([Xtrans, np.power(x, i)])
    X = np.transpose(Xtrans)
    return X

num_of_segments = 3
polynomial_degree = 4

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

breaks = find_breaks(sorted_Y, num_of_segments-1)
sizes = [breaks[i+1]-breaks[i] for i in range(num_of_segments)]

negative_indices = np.where(np.less(Y[mapping], 0))[0]
Nneg = negative_indices.size
mask = np.ones(N)
mask[negative_indices] = -np.ones(Nneg)


# coefficients = [np_LeastSquares(X_segments[i], y_segments[i]) for i in range(num_of_segments)]

y_est_abs = []
x_segments_ = [] ; x_segments = [] ; y_segments = [] ; X_segments = []

for i in range(num_of_segments):
        x_segments_ += [np.arange(breaks[i], breaks[i + 1])]
        x_segments += [np.arange(0, sizes[i])]
        y_segments += [sorted_Y[breaks[i]: breaks[i + 1]]]
        X_segments += [GetInputMatrix_Polynomial(polynomial_degree, x_segments[i])]
        offset = i*polynomial_degree
        y_est_abs += [np.matmul(X_segments[i], coefficients[offset : offset+polynomial_degree])]

y_est_abs_np = np.concatenate(y_est_abs)
print(y_est_abs_np)
y = y_est_abs_np * mask

# Compute Root Mean Squared Error
rmse = np.sqrt(np.sum(np.power(sorted_Y-y_est_abs_np, 2)))

plt.rcParams["figure.figsize"] = [20, 10]
print(rmse)
colors = itertools.cycle(['co', 'ro', 'go', 'yo', 'mo'])
with open(path+'/rmse.txt', 'w') as f:
        f.write(str(rmse) + "\n")

        mapping = mapping+1
        plt.plot(range(1, N+1), Y, 'mo', markersize=5, label="True")
        plt.plot(mapping, y, 'c.', markersize=5, label="Estimated Values")


        plt.plot(range(1, N+1), sorted_Y, 'bo', markersize=6, label="True Sorted")
        for x, X, c, y_est, color in zip(x_segments_, X_segments, coefficients, y_est_abs, colors):
                plt.plot(x, y_est, color, markersize=2, label="Estimates")
        plt.plot(breaks, np.zeros(len(breaks)), 'ko', markersize=6, label="Breaks")

        plt.legend()
        # plt.show()
        plt.savefig(path + 'gradient.png')