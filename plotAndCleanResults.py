import numpy as np
import csv
from matplotlib import pyplot as plt
from AgentOracle import *
from matplotlib2tikz import save as tikz_save


path = '../results_06_09_bellman/'
path = '../results_06_09_multipleSettings_bellman/9/'
#path = '../results_06_09_multipleDiscretizations_bellman/9/'
#path = '../results_06_09_multipleSettings_UCB/9/'
#path = '../results_81431_multipleDiscretizations/3/'
#path = '../results_06_09/'
path = '../results_101757_experimentsTrue2/'

agentPath = np.load(path + "Agents.npy")
#agentPath = ["Sampling/","Mean/","UCB/"]
nExperiments = 64
nCampaigns = 4

optimum = np.load(path + "opt.npy")
print "Oracle Optimum ", optimum
optPol = np.load(path + "optPolicy.npy")

optBidBudMatrix = np.load(path + "OracleBidBudMatrix.npy")
bids = np.linspace(0.0, 1.0, 5)
#legend = ['AdComB-BUCB']


budgets = np.linspace(0.0, 100.0, 10)

convparams = np.load(path + "ConversionValues.npy")

#T = np.load(path + "Deadline.npy")
T=200

def budIndex(bud):
    return np.argwhere(np.isclose(budgets, bud)).reshape(-1)

def bidIndex(bid):
    return np.argwhere(np.isclose(bids,bid)).reshape(-1)

def calcClicks(bidArray,budArray,convParams):
    clicks = []
    for i in range(0,len(bidArray)):
        budInd = budIndex(budArray[i])
        bidInd = bidIndex(bidArray[i])
        c = optBidBudMatrix[i, bidInd, budInd]
        clicks.append(c)
    clicks = np.array(clicks)
    return np.sum(clicks.reshape(-1)*convParams[:nCampaigns,0])

def calcClicksForExperiment(policy, deadline, convParams):
    clicks = np.array([])
    for t in range(0,deadline):
        bidArray = policy[0,t,:]
        budArray = policy[1,t,:]
        clicks = np.append(clicks, calcClicks(bidArray, budArray, convParams))
    return clicks


plt.figure(33)
plt.plot(optPol[0], 'o')
plt.title('OptimalPolicy')
print optPol

# Plotting single agents results
res = []
observedRes = []
for a in range(0, len(agentPath)):
    conv = []
    pol = []
    expClicks = []
    for e in range(0,nExperiments):
        temp = np.load(path + agentPath[a] + "experiment_" + str(e) + ".npy")
        tempPol = np.load(path+ agentPath[a] + "policy_" + str(e) + ".npy")
        if len(temp) > 0:
            conv.append(temp)
            pol.append(tempPol)
            expClicks.append(calcClicksForExperiment(tempPol,T,convparams))
    #plt.figure(1+a)
    conv = np.array(conv)
    #plt.plot(np.mean(conv[:], axis=0))
    #plt.ylim(0, 25)
    #plt.plot(np.ones(T) * np.sum(optimum), '--')
    #plt.title(agentPath[a])
    pol = np.array(pol)

    #plt.figure(1 + a)
    expClicks = np.array(expClicks)
    #plt.plot(np.ones(T) * np.sum(optimum), '--')
    #plt.title(agentPath[a] + "Oracle values")

    #plt.plot(np.mean(expClicks,  axis=0))
    std = np.std(expClicks, axis=0)
    #plt.plot(np.mean(expClicks, axis=0) + 2 * std / np.sqrt(nExperiments), 'b')
    #plt.plot(np.mean(expClicks, axis=0) - 2 * std / np.sqrt(nExperiments), 'b')

    res.append(np.mean(expClicks, axis=0))
    #observedRes.append(np.mean(conv, axis=0))


legend = ['AdComB-TS', 'AdComB-Mean', 'AdComB-BUCB', 'AdComB-3D', 'Oracle']


# All mean results in a single plot
res = np.array(res)
plt.figure(190)
plt.plot(res.T)
plt.plot(np.ones(T) * np.sum(optimum), '--')
plt.legend(legend)
plt.xlabel("t",fontsize=20)
plt.ylabel(r'$P_t(\mathfrak{U})$',fontsize=20)
plt.tick_params(labelsize=18)

tikz_save('reward_t100.tex');

# ???
"""
plt.figure(290)
observedRes = np.array(observedRes)
plt.plot(observedRes.T)
plt.plot(np.ones(T) * np.sum(optimum), '--')
plt.legend(legend)
plt.plot(np.mean(conv[:], axis=0))
"""

# REGRET plot
plt.figure(501)
opt = np.ones((len(agentPath), T)) * np.sum(optimum)
regret = np.cumsum((opt - res[0:len(agentPath), 0:T]), axis=1)
print opt
regret = np.cumsum((19.4 - res[0:len(agentPath), 0:T]), axis=1)

plt.plot(regret.T)
plt.legend(legend[0:len(legend)])
plt.xlabel("t",fontsize=20)
plt.ylabel(r'$R_t(\mathfrak{U})$',fontsize=20)
plt.ylim(0,200)
plt.tick_params(labelsize=18)
tikz_save('regret_t100.tex');


#plt.title("Cumulated Expected Pseudo-Regret")
