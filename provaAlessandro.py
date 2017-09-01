#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
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



show = False
save = False
path = '../results/'

# Auction parameter initialization
convparams=np.array([0.4,100,200])
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
probClick = np.array([0.5, 0.6, 0.6, 0.5,0.4,0.1,0.4,0.5,0.2,0.4])
nMeanResearch = np.array([1000.0,1500,1500,1000,1250,4000,1250,2000,4000,1250])
sigmaResearch = 0.2
nBidders = [5,6,6,5,5,5,6,5,6,6] # non uguali a quelli di Guglielmo
nSlots  = 5
mu =    [ 0.59, 0.67, 0.47, 0.59, 0.57, 0.5 , 0.44, 0.5, 0.4, 0.61 ]
sigma = [0.2 , 0.4, 0.25, 0.39, 0.15, 0.4 ,0.39, 0.4,0.2,0.15,0.15,0.25]

nCampaigns =5
campaigns = []
for c in range(0,nCampaigns):
    a = Auction(nBidders=nBidders[c], nslots=nSlots, mu=mu[c], sigma= sigma[c], lambdas=lambdas, myClickProb =probClick[c])
    campaigns.append(Campaign(a,nMeanResearch=nMeanResearch[c],nStdResearch=sigmaResearch, probClick=probClick[c],convParams=convparams))


# Environment setting
env = Environment(copy.copy(campaigns))

# Experiment setting
nBids = 10
nIntervals = 10
deadline = 500
maxBudget = 100

# Baseline computation
oracle = Oracle(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns,
                     nBudget=nIntervals, nBids=nBids, maxBudget=100.0, environment=copy.copy(env))
oracle.generateBidBudgetMatrix(nSimul=100)
values = np.ones(nCampaigns) * convparams[0]
oracle.updateValuesPerClick(values)
[optBud,optBid,optConv]=oracle.chooseAction()
print "policy val",oracle.bidBudgetMatrix[2,2,-1]
print optConv
oracle.initGPs()
print "initGPs"
oracle.updateMultiGP(500)
print "updated GPS"
np.save(path+"opt",optConv)
np.save(path+"oracle",oracle)


def experiment(k):
    # Agent initialization
    np.random.seed()

    agent = AgentFactored(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBudget=100.0)
    agent.initGPs()
    print "Experiment : ",k
    print "A"

    if show:
        plotter = Plotter(agent, env)
        [trueClicks, trueBudgets] = plotter.trueSample(1.0, maxBudget=100.0, nsimul=400)
    if save:
        trueClicks = np.array([trueClicks])
        trueBudgets = np.array([trueBudgets])
        np.save(path + 'trueClicks', trueClicks)
        np.save(path + 'trueBudgets', trueBudgets)

    # Set the GPs hyperparameters
    for c in range(0,nCampaigns):
        agent.setGPKernel(c , oracle.gpsClicks[c].kernel_ , oracle.gpsCosts[c].kernel_)

    # Init the Core and execute the experiment
    env = Environment(copy.copy(campaigns))
    core = Core(agent, copy.copy(env), deadline)

    core.runEpisode()
    np.save(path+"policy_" +str(k), [agent.prevBids,agent.prevBudgets])
    np.save(path+"experiment_" + str(k),np.sum(agent.prevConversions,axis=1))
    return np.sum(agent.prevConversions,axis=1),agent



nExperiments = 60

out = Parallel(n_jobs=-1)(
        delayed(experiment)(k) for k in xrange(nExperiments))

np.save(path+"allExperiments", out)
#plt.plot(np.sum(agent.prevConversions, axis=1))

