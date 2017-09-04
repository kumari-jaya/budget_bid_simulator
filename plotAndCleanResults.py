import numpy as np
import csv
from matplotlib import pyplot as plt
from AgentOracle import *

path = '../results_bellman/'
agentPath = ["Sampling/","Mean/","UCB/","3D/"]
nExperiments = 60
optimum = np.load(path+"opt.npy")
print optimum
optPol = np.load(path+"optPolicy.npy")

nCampaigns=3
optBidBudMatrix = np.load(path+"OracleBidBudMatrix.npy")
bids = np.linspace(0.0, 5.0, 5)
budgets = np.linspace(0.0, 100.0, 10)
convparams = np.array([[0.4, 100, 200],[0.2, 100, 200],[0.3, 100, 200],[0.2, 100, 200],[0.35, 100, 200]])
#convparams = np.array([[0.4, 100, 200],[0.4, 100, 200],[0.4, 100, 200],[0.4, 100, 200],[0.4, 100, 200]])

T=100

def budIndex(bud):
    return np.argwhere(np.isclose(budgets,bud)).reshape(-1)

def bidIndex(bid):
    return np.argwhere(np.isclose(bids,bid)).reshape(-1)

def calcClicks(bidArray,budArray,convParams):
    clicks = []
    for i in range(0,len(bidArray)):
        budInd = budIndex(budArray[i])
        bidInd = bidIndex(bidArray[i])
        c=optBidBudMatrix[i,bidInd,budInd]
        clicks.append(c)
    clicks =np.array(clicks)
    return np.sum(clicks.reshape(-1)*convParams[:nCampaigns,0])

def calcClicksForExperiment(policy,deadline,convParams):
    clicks = np.array([])
    for t in range(0,deadline):
        bidArray=policy[0,t,:]
        budArray=policy[1,t,:]
        clicks = np.append(clicks,calcClicks(bidArray,budArray,convParams))
    return clicks








plt.figure(33)
plt.plot(optPol[0],'o')
plt.title('OptimalPolicy')
print optPol

res =[]
observedRes =[]
for a in range(0,len(agentPath)):
    conv =[]
    pol =[]
    expClicks = []
    for e in range(0,nExperiments):
        temp = np.load(path+ agentPath[a]+ "experiment_"+ str(e)+".npy")
        tempPol = np.load(path+ agentPath[a]+ "policy_"+ str(e)+".npy")
        if len(temp)>0:
            conv.append(temp)
            pol.append(tempPol)
            expClicks.append(calcClicksForExperiment(tempPol,T,convparams))
    plt.figure(a)
    conv = np.array(conv)
    plt.plot(np.mean(conv[:],axis=0))
    plt.ylim(0,25)
    plt.plot(np.ones(T)*np.sum(optimum),'--')
    plt.title(agentPath[a])
    pol = np.array(pol)

    #clicks =calcClicksForExperiment(pol[-1],400)
    plt.figure(10+a)
    expClicks=np.array(expClicks)
    plt.plot(np.ones(T)*np.sum(optimum),'--')
    plt.title(agentPath[a] + "oracle values")
    plt.plot(np.mean(conv[:],axis=0))

    plt.plot(np.mean(expClicks,axis=0))

    res.append(np.mean(expClicks,axis=0))
    observedRes.append(np.mean(conv,axis=0))


res = np.array(res)
plt.figure(190)
plt.plot(res.T)

plt.plot(np.ones(T) * np.sum(optimum), '--')
leg = ['Sampling','Mean','UCB','Optimum']
plt.legend(leg)


plt.figure(290)
observedRes=np.array(observedRes)
plt.plot(observedRes.T)

plt.plot(np.ones(T) * np.sum(optimum), '--')
leg = ['Sampling','Mean','UCB','Optimum']
plt.legend(leg)
plt.plot(np.mean(conv[:], axis=0))

