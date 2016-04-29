import numpy as np
import matplotlib.pyplot as plt


def map_feature(X1, X2):
    out = []
    for i in range(1, 7):
        for j in range(i + 1):
            ...

data = np.loadtxt('ex2data2.txt', delimiter=',')
positive = data[data[:, 2] == 1, :]
negative = data[data[:, 2] == 0, :]
plt.scatter(positive[:, 0], positive[:, 1], c='b', marker='+', label='y = 1')
plt.scatter(negative[:, 0], negative[:, 1], c='y', marker='o', label='y = 0')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show()

# Part 1: Regularized Logistic Regression
