#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from AuctionTrueData import *
#from Agent import *
from Core import *
from Plotter import *
from matplotlib import pyplot as plt
from AgentFactored import *
from PlotterFinal import *
from AgentPrior import *
from AgentOracle import *
from joblib import Parallel, delayed
import copy
from numpy import genfromtxt


show = False
save = True
path = '../results/'
# Auction parameter initialization
nCampaigns =3

convparams = np.array([[0.4, 100, 200],[0.3, 100, 200],[0.3, 100, 200],[0.2, 100, 200],[0.35, 100, 200]])
## Discount probabilities
lambdas = np.array([1.0, 0.71, 0.56, 0.53, 0.49, 0.47, 0.44, 0.44, 0.43, 0.43])
## Click probabilities of the considered ad
probClick = np.array([0.5, 0.6, 0.6, 0.5, 0.4, 0.1, 0.4, 0.5, 0.2, 0.4])*0.2
## Number of research per day
nMeanResearch = np.ones(nCampaigns)*1000.0
sigmaResearch = np.ones(nCampaigns)*1.0

sigmaResearch = np.array([100,400,200,10,10,10,10])
## Number of other bidders in the auction
nBidders = [5, 6, 6, 5, 5, 5, 6, 5, 6, 6]

nBidders = np.ones(nCampaigns)*5
nSlots = 5


auctionsFile = './data/BidData.csv'



allData = genfromtxt(auctionsFile, delimiter=',')
index = np.random.randint(0, 100,nCampaigns)
#probClick = np.random.beta(allData[index, 4], allData[index, 5])
probClick=np.array([ 0.02878113  ,0.24013416,  0.02648224,  0.01104576,  0.06390204])


campaigns = []
index = np.array([1,4,8,17,22])
index = np.array([2,6,8,60,22])

for c in range(0, nCampaigns):
    a = AuctionTrueData(nBidders=int(nBidders[c]), nSlots=nSlots,lambdas=lambdas, myClickProb=probClick[c],fixedIndex=index[c])
    campaigns.append(Campaign(a, nMeanResearch=nMeanResearch[c], nStdResearch=sigmaResearch[c],probClick =probClick[c], convParams=convparams[c]))





# Environment setting
env = Environment(copy.copy(campaigns))

# Experiment setting
nBids = 5
nIntervals = 10
deadline = 100
maxBudget = 100.0
maxBid = 5.0

# Baseline computation
oracle = Oracle(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns,
                     nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, environment=copy.copy(env))

oracle.generateBidBudgetMatrix(nSimul=20)

values = np.ones(nCampaigns) * convparams[:nCampaigns,0]
oracle.updateValuesPerClick(values)

[optBud,optBid,optConv]=oracle.chooseAction()
print "budget pOlicy",optBud
print "bid Policy",optBid

#print "policy val",oracle.bidBudgetMatrix[2,2,-1]
print "Conversion Oracle",optConv
oracle.initGPs()
oracle.initGPs3D()
print "initGPs"
oracle.updateMultiGP(500)
oracle.updateMultiGP3D(500)
print "updated GPS"
if save==True:
    np.save(path+"opt",optConv)
    np.save(path + "optPolicy",[optBud,optBid])
    np.save(path+"oracle",oracle)
    np.save(path+"OracleBidBudMatrix",oracle.bidBudgetMatrix)
print "budget pOlicy",optBud

agentPath = ["Sampling/","Mean/","UCB/","3D/"]
def experiment(k):
        # Agent initialization
    np.random.seed()
    agents=[]
    agents.append(AgentFactored(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, method="Sampling"))
    agents.append(AgentFactored(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, method="Mean"))
    agents.append(AgentFactored(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids,  maxBid=maxBid,maxBudget=maxBudget, method="UCB"))
    agents.append(AgentPrior(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget,usePrior=False))

    ret = []

    for a,agent in enumerate(agents):
        agent.initGPs()
        print "Experiment : ",k

        # Set the GPs hyperparameters
        for c in range(0,nCampaigns):
            if a ==3:
                agent.setGPKernel(c,oracle.gps3D[c].kernel_)
            else:
                agent.setGPKernel(c , oracle.gpsClicks[c].kernel_ , oracle.gpsCosts[c].kernel_)

        # Init the Core and execute the experiment
        env = Environment(copy.copy(campaigns))
        core = Core(agent, copy.copy(env), deadline)

        core.runEpisode()
        np.save(path+agentPath[a]+"policy_" +str(k), [agent.prevBids,agent.prevBudgets])
        np.save(path+agentPath[a]+"experiment_" + str(k),np.sum(agent.prevConversions,axis=1))

        ret.append(np.sum(agent.prevConversions,axis=1))
    return ret



nExperiments = 4

out = Parallel(n_jobs=2)(
        delayed(experiment)(k) for k in xrange(nExperiments))

np.save(path+"allExperiments", out)
#plt.plot(np.sum(agent.prevConversions, axis=1))

