import numpy as np
import csv
from matplotlib import pyplot as plt
from AgentOracle import *

path = '../results_bellman/'
agentPath = ["Sampling/","Mean/","UCB/"]#,"3D/"]
nExperiments = 2
optimum = np.load(path+"opt.npy")
print optimum
optPol = np.load(path+"optPolicy.npy")
plt.figure(33)
plt.plot(optPol[0],'o')
plt.title('OptimalPolicy')
print optPol
for a in range(0,len(agentPath)):
    conv =[]
    pol =[]
    for e in range(0,nExperiments):
        temp = np.load(path+ agentPath[a]+ "experiment_"+ str(e)+".npy")
        tempPol = np.load(path+ agentPath[a]+ "policy_"+ str(e)+".npy")
        if len(temp)>0:
            conv.append(temp)
            pol.append(tempPol)
    plt.figure(a)
    conv = np.array(conv)
    plt.plot(np.mean(conv[:],axis=0))
    plt.ylim(6,25)
    plt.plot(np.ones(500)*np.sum(optimum),'--')
    plt.title(agentPath[a])

    pol = np.array(pol)
    #plt.figure(a+100)
    #plt.plot(pol[ :, 1,:, 4].T)
    #plt.title("Traiettorie" +agentPath[a])

"""
results = np.load(path + "allExperiments.npy" )


opt = np.load(path + "opt.npy")
results= results[:,0]
conv = np.mean(results)
std = np.std(results)
#plt.plot(conv)
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