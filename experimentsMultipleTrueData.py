#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from numpy import genfromtxt
from Campaign import *
from Environment import *
from AuctionTrueData import *
from Core import *
import datetime
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




gg = str(datetime.datetime.now().day)
hh = str(datetime.datetime.now().hour)
min = str(datetime.datetime.now().minute)


path = '../results_'+gg+hh+min+'_multipleSettings/'
ensure_dir(path)

# Experiment setting
nBids = 5
nIntervals = 10
maxBudget = 100.0
maxBid = 1.0



deadline = 100
nExperiments = 100
nSettings = 10
nSimul = 100
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

for s in range(0,nSettings):
    pathSetting = path + str(s)+"/"
    ensure_dir(pathSetting)

    index = np.random.randint(0, 100, nCampaigns)
    probClick = np.random.beta(allData[index, 4], allData[index, 5])
    #probClick = np.array([ 0.02878113  ,0.24013416,  0.02648224,  0.01104576,  0.06390204])
    #probClick = np.array([ 0.02878113  , 0.02648224,  0.01104576,  0.0134576 ,0.0639])

    campaigns = []

    # Campaign setting
    for c in range(0, nCampaigns):
        a = AuctionTrueData(nBidders=int(nBidders[c]), nSlots=nSlots,lambdas=lambdas, myClickProb=probClick[c])
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

    print "Budget policy: ", optBud
    print "Bid policy: ", optBid
    print "Optimal conversion given by the oracle: ", optConv
    if (np.sum(optConv) >=3.0):
        oracle.initGPs()
        oracle.initGPs3D()
        print "initGPs"
        oracle.updateMultiGP(nTrainingInputs)
        oracle.updateMultiGP3D(nTrainingInputs)
        oracle.updateCostsPerBids()
        print "updated GPS"
    if save == True:
        np.save(pathSetting + "opt", optConv)
        np.save(pathSetting + "optPolicy", [optBud,optBid])
        np.save(pathSetting + "OracleBidBudMatrix", oracle.bidBudgetMatrix)
        np.save(pathSetting + "ConversionValues", convparams)
        np.save(pathSetting + "Deadline", deadline)
    print "budget policy", optBud

    agentPath = ["Sampling/", "Mean/", "UCB/", "3D/"]
    if save == True:
        np.save(pathSetting+ "Agents", agentPath)



    def experiment(k):
        # Agent initialization
        np.random.seed()
        agents = []
        agents.append(AgentFactoredExperiment(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, method="Sampling"))
        agents.append(AgentFactoredExperiment(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, method="Mean"))
        agents.append(AgentFactoredExperiment(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids,  maxBid=maxBid,maxBudget=maxBudget, method="UCB"))
        agents.append(AgentPrior(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBid=maxBid, maxBudget=maxBudget, usePrior=False))
        results = []
        for idxAgent, agent in enumerate(agents):
            agent.initGPs()
            print "Experiment : ", k

            # Set the GPs hyperparameters
            for c in range(0, nCampaigns):
                if agentPath[idxAgent] == "3D/":
                    print "AOOO"
                    agent.setGPKernel(c, oracle.gps3D[c].kernel_, oracle.alphasClicksGP[c])
                else:
                    print "\n"
                    print "alphaCosts:       ", oracle.alphasPotCostsGP
                    print "alphaClicks:       ", oracle.alphasPotClicksGP
                    agent.setGPKernel(c, oracle.gpsClicks[c].kernel_, oracle.gpsCosts[c].kernel_,
                                      alphaClicks=oracle.alphasPotClicksGP[c], alphaCosts=oracle.alphasPotCostsGP[c])

            # Init the Core and execute the experiment
            envi = Environment(copy.copy(campaigns))
            core = Core(agent, copy.copy(envi), deadline)
            core.runEpisode()
            ensure_dir(pathSetting + agentPath[idxAgent])
            np.save(pathSetting + agentPath[idxAgent] + "policy_" + str(k), [agent.prevBids, agent.prevBudgets])
            np.save(pathSetting + agentPath[idxAgent] + "experiment_" + str(k), np.sum(agent.prevConversions, axis=1))
            results.append(np.sum(agent.prevConversions, axis=1))
        return [results, agents, envi]


    if (np.sum(optConv) >=3.0):
        out = Parallel(n_jobs=-1)(
                delayed(experiment)(k) for k in xrange(nExperiments))

        np.save(pathSetting + "allExperiments", out)



