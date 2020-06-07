from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import itertools

def get_breaks(model, N):
    if model == "resnet20_v2":
        breaks = {2304:3, 4608:3, 9216:3, 18432:3, 36864:3} #432
    elif model == "vgg16":
        breaks = {
            1728 : [0, 1443, 1663, 1728],
            36864 : [0, 34097, 36467, 36815, 36864],
            73728 : [0, 67595, 73032, 73630, 73728],
            147456 : [0, 132193, 145286, 147125, 147456],
            294912 : [0, 272485, 292623, 294580, 294844, 294912],
            589824 : [0, 553577, 586620, 589431, 589764, 589824],
            1179648 : [0, 1099105, 1172811, 1179005, 1179543, 1179648],
            2359296 : [0, 2195844, 2343594, 2357633, 2359102, 2359296]}
    elif model == "resnet50":
        breaks = {16384, 36864, 131072, 32768, 147456, 65536, 524288,
                      589824, 262144, 2097152, 524288, 2359296, 1048576, 2050048}   #4096 #9408
    return breaks[N]

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

def get_num_of_segments(model, N):
    if model == "resnet20_v2":
        segments = {2304:3, 4608:3, 9216:3, 18432:3, 36864:3} #432
    elif model == "vgg16":
        segments = {1728:3, 36864:4, 73728:4, 147456:4, 294912:5, 589824:5, 1179648:5, 2359296:5}
    elif model == "resnet50":
        segments = {16384, 36864, 131072, 32768, 147456, 65536, 524288, 589824, 262144, 2097152, 524288, 2359296, 1048576, 2050048}   #4096 #9408
    return segments[N]

def GetInputMatrix_Polynomial(xcol, x):
    N = len(x)
    Xtrans = [np.ones(N)]
    for i in range(1, xcol):
        Xtrans = np.vstack([Xtrans, np.power(x, i)])
    X = np.transpose(Xtrans)
    return X

polynomial_degree = 4

pd.set_option("display.precision", 40)
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="./", help='')
parser.add_argument('--model', type=str, default="./", help='')
args = parser.parse_args()

path = args.path
Y = pd.read_csv(path+'/values.csv', header=None, sep="\n")[0].values
coefficients = pd.read_csv(path+'/coefficients.csv', header=None, sep="\n")[0].values
#print(Y) ; print(coefficients)
N = Y.size

num_of_segments = get_num_of_segments(args.model, N)

y_abs = np.abs(Y)
mapping = np.argsort(y_abs, axis=0)
sorted_Y = y_abs[mapping]

# breaks = find_breaks(sorted_Y, num_of_segments-1)
breaks = get_breaks(args.model, N)
sizes = [breaks[i+1]-breaks[i] for i in range(num_of_segments)]

negative_indices = np.where(np.less(Y[mapping], 0))[0]
Nneg = negative_indices.size
mask = np.ones(N)
mask[negative_indices] = -np.ones(Nneg)

y_est_abs = []
x_segments_ = [] ; x_segments = [] ; y_segments = [] ; X_segments = []

for i in range(num_of_segments):
        x_segments_ += [np.arange(breaks[i], breaks[i + 1])]
        x_segments += [np.cast['float64'](np.arange(0, sizes[i]))]
        y_segments += [sorted_Y[breaks[i]: breaks[i + 1]]]
        X_segments += [GetInputMatrix_Polynomial(polynomial_degree, x_segments[i])]
        offset = i*polynomial_degree
        y_est_abs += [np.matmul(X_segments[i], coefficients[offset : offset+polynomial_degree])]

y_est_abs_np = np.concatenate(y_est_abs)
y = y_est_abs_np * mask

# Compute Root Mean Squared Error
rmse = np.sqrt(np.sum(np.power(sorted_Y-y_est_abs_np, 2)))

plt.rcParams["figure.figsize"] = [20, 10]
print(rmse)
colors = itertools.cycle(['yo', 'ro', 'go', 'yo', 'mo'])
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