from __future__ import print_function
import numpy as np
import time
import matplotlib as mpl
import math
import matplotlib.pyplot as plt

plt.figure()


for method in range(0,3,1):
    iteration = np.load('./data/method_' + str(method) + '/result/iteration.npy')
    reward = np.load('./data/method_' + str(method) + '/result/reward.npy')

    if method == 0:
        plt.plot(iteration, reward, 'r', linewidth=2.0)
    if method == 1:
        plt.plot(iteration, reward, 'g', linewidth=2.0)
    if method == 2:
        plt.plot(iteration, reward, 'b', linewidth=2.0)


plt.show()

