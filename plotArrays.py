import numpy as np
from matplotlib import pyplot as plt

from graphicalTool.Plotter import *

clicks = np.load('presentazione/trueClikcs.npy')
budgets = np.load('presentazione/trueBudgets.npy')
clicks = clicks.reshape(-1)
budgets = budgets.reshape(-1)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N
clicks = running_mean(clicks,5)
clicks = np.append([0],clicks)
plt.figure(0)
plt.plot(budgets[:-3],clicks)
plt.xlabel('Budget')
plt.ylabel('Clicks')
plt.ylim(0,200)
plt.show()