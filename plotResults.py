import numpy as np
import csv
from matplotlib import pyplot as plt
from AgentOracle import *

path = '../resultsOK0/'
results = np.load(path + "allExperiments.npy" )


opt = np.load(path + "opt.npy")
results= results[:,0]
conv = np.mean(results)
std = np.std(results)
plt.plot(conv)
plt.plot(np.ones(len(conv))*np.sum(opt))
plt.plot(conv +std*2/np.sqrt(60))
plt.ylim(0,100)

oracle = np.load(path + "oracle.npy")
name = "policy_"
nExp = 60
policy = []
for e in range(0,nExp):
    policy.append(np.load(path+name + str(e)+".npy"))

p = np.array(policy)
