#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import copy

from joblib import Parallel, delayed
# from Agent import *
from lost_and_found.environment.Core import *
from lost_and_found.environment.Oracle import *

from agent.AgentFactored import *
from agent.AgentPrior import *
from graphicalTool.Plotter import *

show = False
save = False
path = '../results3D/'

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
deadline = 200
maxBudget = 100

# Baseline computation
oracle = Oracle(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns,
                     nBudget=nIntervals, nBids=nBids, maxBudget=100.0, environment=copy.copy(env))
oracle.generateBidBudgetMatrix(nSimul=50)
values = np.ones(nCampaigns) * convparams[0]
oracle.updateValuesPerClick(values)
[optBud,optBid,optConv]=oracle.chooseAction()
"""
print optConv
oracle.initGPs()
print "initGPs"
oracle.updateMultiGP(500)
print "updated GPS"
"""
np.save(path+"opt",optConv)

def experiment(k):
    # Agent initialization
    agent = AgentPrior(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBudget=100.0)
    agent.initGPs()
    print "Experiment3D : ",k

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
    np.save(path+"policy3D_" +str(k), agent.prevBudgets)
    np.save(path+"experiment3D_" + str(k),np.sum(agent.prevConversions,axis=1))
    return np.sum(agent.prevConversions,axis=1)



nExperiments = 60

out = Parallel(n_jobs=-1)(
        delayed(experiment)(k) for k in xrange(nExperiments))

np.save(path+"allExperiments3D", out)
#plt.plot(np.sum(agent.prevConversions, axis=1))

