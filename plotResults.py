import numpy as np
import csv
from matplotlib import pyplot as plt

path = '../results0/'
results = np.load(path + "allExperiments.npy" )
opt = np.load(path + "opt.npy")

conv = np.mean(results,axis=0)
std = np.std(results.reshape(-1),axis=0)
plt.plot(conv)
plt.plot(np.ones(len(conv))*np.sum(opt))
plt.plot(conv +std)


