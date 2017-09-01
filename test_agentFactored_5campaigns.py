#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test for the factored agent on a synthetic environment with 5 campaigns
Each auction has 5 slots and 5-6 other bidders
"""

import copy

from joblib import Parallel, delayed
from environment.Core import *
from environment.Oracle import *
from environment.Auction import *
from environment.Campaign import *

from agent.AgentFactored import *
from graphicalTool.Plotter import *

show = False
save = True
path = '../results/'

# Auction parameter initialization
convParams = np.array([0.4, 100, 200])
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
probClick = np.array([0.5, 0.6, 0.6, 0.5, 0.4, 0.1, 0.4, 0.5, 0.2, 0.4])
nMeanResearch = np.array([1000.0, 1500.0, 1500.0, 1000.0, 1250.0,
                          4000.0, 1250.0, 2000.0, 4000.0, 1250.0])
sigmaResearch = 0.2

nBidders = [5, 6, 6, 5, 5, 5, 6, 5, 6, 6] # non uguali a quelli di Guglielmo
nSlots = 5
mu =    [0.59, 0.67, 0.47, 0.59, 0.57, 0.5 , 0.44, 0.5, 0.4, 0.61 ]
sigma = [0.2 , 0.4, 0.25, 0.39, 0.15, 0.4, 0.39, 0.4, 0.2, 0.15, 0.15, 0.25]

# Campaign setting
nCampaigns = 5
campaigns = []
for c in range(0, nCampaigns):
    a = Auction(nBidders=nBidders[c], nslots=nSlots, mu=mu[c],
                sigma=sigma[c], lambdas=lambdas, myClickProb=probClick[c])
    campaigns.append(Campaign(a, nMeanResearch=nMeanResearch[c],
                              nStdResearch=sigmaResearch, probClick=probClick[c],
                              convParams=convParams))
print 'Campaigns initialized'

# Environment setting
envi = Environment(copy.copy(campaigns))
print 'Environment initialized'


# Experiment setting
nBids = 10
nIntervals = 10
deadline = 250
maxBudget = 100

# Baseline computation
oracle = Oracle(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns,
                nBudget=nIntervals, nBids=nBids, maxBudget=100.0,
                environment=copy.copy(envi))
oracle.generateBidBudgetMatrix(nSimul=50)
values = np.ones(nCampaigns) * convParams[0]
oracle.updateValuesPerClick(values)
[optBud, optBid, optConv] = oracle.chooseAction()
print optConv
oracle.initGPs()
print "initGPs finished"
oracle.updateMultiGP(500)
print "Updated GPS finished"

if save:
    np.save(path + "oracle", oracle)
    np.save(path + "optValue", optConv)

# Definition of a single experiment
def experiment(k):
    # Agent initialization
    agent = AgentFactored(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns,
                          Budget=nIntervals, nBids=nBids, maxBudget=100.0)
    agent.initGPs()
    print "Experiment : ", k

    if show:
        plotter = Plotter(agent, envi)
        [trueClicks, trueBudgets] = plotter.trueSample(1.0, maxBudget=100.0, nsimul=400)

    # Set the GPs hyperparameters
    for c in range(0, nCampaigns):
        agent.setGPKernel(c, oracle.gpsClicks[c].kernel_, oracle.gpsCosts[c].kernel_)

    # Init the Core and execute the experiment
    env = Environment(copy.copy(campaigns))
    core = Core(agent, copy.copy(env), deadline)
    core.runEpisode()

    if save:
        np.save(path + "policy_" + str(k), [agent.prevBudgets, agent.prevBids])
        np.save(path + "experiment_" + str(k), np.sum(agent.prevConversions, axis=1))
    return np.sum(agent.prevConversions, axis=1)


# Execution of the experiments
nExperiments = 5
out = Parallel(n_jobs=2)(
        delayed(experiment)(k) for k in xrange(nExperiments))
if save:
    np.save(path + "allExperiments", out)

#plt.plot(np.sum(agent.prevConversions, axis=1))