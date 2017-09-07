#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from numpy import genfromtxt

from Campaign import *
from Environment import *
from AuctionTrueData import *
from Core import *

from AgentFactoredExperiment import *
from AgentPrior import *
from AgentOracle import *
from joblib import Parallel, delayed
import copy
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

save = True
path = '../results_06_09/'
ensure_dir(path)

# Experiment setting
nBids = 5
nIntervals = 10
maxBudget = 100.0
maxBid = 1.0


deadline = 60
nExperiments = 10
nSimul = 50
nTrainingInputs = 500


# Auction setting
nCampaigns = 5
nBidders = np.ones(nCampaigns) * 10
nSlots = 5

# Conversion probabilities
convparams = np.array([[0.5, 100, 200],[0.6, 100, 200],[0.4, 100, 200],[0.5, 100, 200],[0.35, 100, 200]])

## Discount probabilities
lambdas = np.array([1.0, 0.71, 0.56, 0.53, 0.49, 0.47, 0.44, 0.44, 0.43, 0.43])
## Number of research per day
nMeanResearch = np.ones(nCampaigns) * 1000.0
sigmaResearch = np.ones(nCampaigns) * 1.0

#sigmaResearch = np.array([400,400,400,10,10,10,10])
## Number of other bidders in the auction
#nBidders = [5, 6, 6, 5, 5, 5, 6, 5, 6, 6]

## Click probabilities of the considered ad
auctionsFile = './data/BidData.csv'
allData = genfromtxt(auctionsFile, delimiter=',')
#index = np.random.randint(0, 100, nCampaigns)
#probClick = np.random.beta(allData[index, 4], allData[index, 5])
#probClick = np.array([ 0.02878113  ,0.24013416,  0.02648224,  0.01104576,  0.06390204])
probClick = np.array([ 0.02878113  , 0.02648224,  0.01104576,  0.0134576 ,0.0639])

campaigns = []
index = np.array([2, 6, 8, 60, 22])

# Campaign setting
for c in range(0, nCampaigns):
    a = AuctionTrueData(nBidders=int(nBidders[c]), nSlots=nSlots,lambdas=lambdas, myClickProb=probClick[c],fixedIndex=index[c])
    campaigns.append(Campaign(a, nMeanResearch=nMeanResearch[c], nStdResearch=sigmaResearch[c],probClick =probClick[c], convParams=convparams[c]))

# Environment setting
envOracle = Environment(copy.copy(campaigns))

# Baseline computation
oracle = Oracle(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns,
                nBudget=nIntervals, nBids=nBids, maxBid=maxBid,
                maxBudget=maxBudget, environment=copy.copy(envOracle))

oracle.generateBidBudgetMatrix(nSimul=nSimul)
values = np.ones(nCampaigns) * convparams[:nCampaigns, 0]
oracle.updateValuesPerClick(values)
[optBud, optBid, optConv] = oracle.chooseAction()

print "budget policy", optBud
print "bid policy", optBid
print "Optimal conversion given by the oracle", optConv

oracle.initGPs()
oracle.initGPs3D()
print "initGPs"
oracle.updateMultiGP(nTrainingInputs)
oracle.updateMultiGP3D(nTrainingInputs)
print "updated GPS"
if save == True:
    np.save(path + "opt", optConv)
    np.save(path + "optPolicy", [optBud,optBid])
    np.save(path + "OracleBidBudMatrix", oracle.bidBudgetMatrix)
    np.save(path + "ConversionValues", convparams)
    np.save(path + "Deadline", deadline)
print "budget policy", optBud

#agentPath = ["Sampling/", "Mean/", "UCB/", "3D/"]
agentPath = ["3D/"]

if save == True:
    np.save(path + "Agents", agentPath)


def experiment(k):
    # Agent initialization
    np.random.seed()
    agents = []
    #agents.append(AgentFactoredExperiment(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, method="Sampling"))
    #agents.append(AgentFactoredExperiment(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, method="Mean"))
    #agents.append(AgentFactoredExperiment(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, method="UCB"))
    agents.append(AgentPrior(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, usePrior=False))


    results = []

    for idxAgent, agent in enumerate(agents):
        agent.initGPs()
        print "Experiment : ", k

        # Set the GPs hyperparameters
        for c in range(0, nCampaigns):
            if agentPath[idxAgent] == "3D/":
                agent.setGPKernel(c, oracle.gps3D[c].kernel_)
            else:
                agent.setGPKernel(c, oracle.gpsClicks[c].kernel_ , oracle.gpsCosts[c].kernel_)

        # Init the Core and execute the experiment
        envi = Environment(copy.copy(campaigns))
        core = Core(agent, copy.copy(envi), deadline)

        core.runEpisode()

        ensure_dir(path + agentPath[idxAgent])
        np.save(path + agentPath[idxAgent] + "policy_" + str(k), [agent.prevBids, agent.prevBudgets])
        np.save(path + agentPath[idxAgent] + "experiment_" + str(k), np.sum(agent.prevConversions, axis=1))

        results.append(np.sum(agent.prevConversions, axis=1))
    return [results, agents, envi]



out = Parallel(n_jobs=2)(
        delayed(experiment)(k) for k in xrange(nExperiments))

np.save(path + "allExperiments", out)
