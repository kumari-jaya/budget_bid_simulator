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


path = '../results_'+gg+hh+min+'NCampaignsSettingsOracle/'
ensure_dir(path)

# Experiment setting
nBids = 5
nIntervals = 10
maxBudget = 100.0
maxBid = 1.0



deadline = 100
nExperiments = 64
nSimul = 300
nTrainingInputs = 500



# Auction setting
campaignsSettings = np.array([6],dtype='int')
maxNCampaigns = np.max(campaignsSettings)
nBidders = np.ones(int(maxNCampaigns)) * 10
nSlots = 5



# Conversion probabilities
convparams = np.array([[0.5, 100, 200],[0.6, 100, 200],[0.35, 100, 200],[0.4, 100, 200],[0.5, 100, 200],[0.3, 100, 200]])

## Discount probabilities
lambdas = np.array([1.0, 0.71, 0.56, 0.53, 0.49, 0.47, 0.44, 0.44, 0.43, 0.43])
## Number of research per day
nMeanResearch = np.ones(maxNCampaigns) * 1000.0
sigmaResearch = np.ones(maxNCampaigns) * 1.0

#sigmaResearch = np.array([400,400,400,10,10,10,10])
## Number of other bidders in the auction
#nBidders = [5, 6, 6, 5, 5, 5, 6, 5, 6, 6]

## Click probabilities of the considered ad
auctionsFile = './data/BidData.csv'
allData = genfromtxt(auctionsFile, delimiter=',')



for s in range(0,len(campaignsSettings)):

    nCampaigns = campaignsSettings[s]

    pathSetting = path + str(campaignsSettings[s])+"_campaigns/"
    ensure_dir(pathSetting)

    #probClick = np.random.beta(allData[index, 4], allData[index, 5])
    #probClick = np.array([ 0.02878113  ,0.24013416,  0.02648224,  0.01104576,  0.06390204])
    probClick = np.array([0.02878113, 0.02648224, 0.0639, 0.01104576, 0.0134576,01104576])


    index = np.array([2, 6, 22, 8, 60,8])

    campaigns = []

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

    print "Budget policy: ", optBud
    print "Bid policy: ", optBid
    print "Optimal conversion given by the oracle: ", optConv

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






