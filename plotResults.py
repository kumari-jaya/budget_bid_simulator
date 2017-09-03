import numpy as np
import csv
from matplotlib import pyplot as plt
from AgentOracle import *

path = '../results3D/'
#results = np.load(path + "allExperiments.npy" )


nExperiments = 59
conv = []
pol = []
for e in range(0, nExperiments):
    if e!=34:
        conv.append(np.load(path + "experiment3D_" + str(e) + ".npy"))
        pol.append(np.load(path + "policy3D_" + str(e) + ".npy"))
plt.figure(1)

conv = np.array(conv)
plt.plot(np.mean(conv[:], axis=0))
pol = np.array(pol)
#plt.plot(np.ones(500) * np.sum(optimum), '--')
"""
opt = np.load(path + "opt.npy")
results= results[:,0]
conv = np.mean(results)
std = np.std(results)
plt.plot(conv)
#plt.plot(np.ones(len(conv))*np.sum(opt))
#plt.plot(conv +std*2/np.sqrt(60))
plt.ylim(40,100)
plt.xlabel("Days")
plt.ylabel("Conversions")
plt.title("Conversions per Day -2D Algorithm")
oracle = np.load(path + "oracle.npy")
name = "policy_"
nExp = 60
policy = []
for e in range(0,nExp):
    policy.append(np.load(path+name + str(e)+".npy"))

p = np.array(policy)
"""