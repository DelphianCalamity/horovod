from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def logit_basis(X, a, N):  # log(p/(1-p))
    return a * np.log(X / ((N + 1) - X))

def exp_basis(X, b, c):
    return b * np.exp(c * X)

def GetInputMatrix(x, p0, N):
    Xtrans = np.ones([N], np.float64)
    for [a, b, c] in p0:
        basis = logit_basis(x, a, N)
        Xtrans = np.vstack([Xtrans, basis])
        basis = exp_basis(x, b, c)
        Xtrans = np.vstack([Xtrans, basis])
    return np.transpose(Xtrans)

pd.set_option("display.precision", 40)
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="./", help='')
args = parser.parse_args()

path = args.path
Y = pd.read_csv(path+'/values.csv', header=None, sep="\n")[0].values
coefficients = pd.read_csv(path+'/coefficients.csv', header=None, sep="\n")[0].values
#print(Y) ; print(coefficients)
N = Y.size
mapping = np.argsort(Y, axis=0)
sorted_Y = Y[mapping]

X = np.array(range(1, N+1), np.float64)
p0 = [[0.004, -0.01, -0.04]]
X_train = GetInputMatrix(X, p0, N)
y_estimates = np.matmul(X_train, coefficients)

# Compute Root Mean Squared Error
rmse = np.sqrt(np.sum(np.power(sorted_Y-y_estimates, 2)))

plt.rcParams["figure.figsize"] = [20, 10]
print(rmse)
with open(path+'/rmse.txt', 'w') as f:
    f.write(str(rmse) + "\n")

mapping = mapping+1
plt.plot(range(1, N+1), Y, 'c.', markersize=5, label="True")
plt.plot(mapping, y_estimates, 'mo', markersize=5, label="Estimated Values")
plt.plot(range(1, N+1), sorted_Y, 'bo', markersize=6, label="True Sorted")
plt.plot(range(1, N+1), y_estimates, 'ro', markersize=6, label="Estimated Sorted")
plt.legend()
# plt.show()
plt.savefig(path + 'gradient.png')