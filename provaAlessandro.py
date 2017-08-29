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
from AgentRandomPolicy import *
from AgentFactored import *
from PlotterFinal import *
from AgentPrior import *
from AgentOracle import *


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

nCampaigns =10
campaigns = []
for c in range(0,nCampaigns):
    a = Auction(nbidders=nBidders[c], nslots=nSlots, mu=mu[c], sigma= sigma[c],lambdas=lambdas, myClickProb =probClick[c])
    campaigns.append(Campaign(a,nMeanResearch=nMeanResearch[c],nStdResearch=sigmaResearch, probClick=probClick[c],convParams=convparams))


# Environment setting
env = Environment(campaigns)

# Experiment setting
nBids = 5
nIntervals = 5
deadline = 30
maxBudget = 100

# Baseline computation
oracle = Oracle(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns,
                     nBudget=nIntervals, nBids=nBids, maxBudget=100.0, environment=env)
oracle.generateBidBudgetMatrix()
values = np.ones(nCampaigns) * convparams[0]
oracle.updateValuesPerClick(values)
oracle.chooseAction()

oracle.initGPs()
oracle.updateMultiGP()

# Agent initialization
agent = AgentFactored(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns, nBudget=nIntervals, nBids=nBids, maxBudget=100.0)
agent.initGPs()


if show:
    plotter = Plotter(agent, env)
    [trueClicks, trueBudgets] = plotter.trueSample(1.0, maxBudget=100.0, nsimul=400)
if save:
    trueClicks = np.array([trueClicks])
    trueBudgets = np.array([trueBudgets])
    np.save(path + 'trueClikcs', trueClicks)
    np.save(path + 'trueBudgets', trueBudgets)

# Set the GPs hyperparameters
for c in range(0,nCampaigns):
    agent.setGPKernel(c , oracle.gpsClicks[c].kernel_ , oracle.gpsCosts[c].gpsCorts[c].kernel_)

# Init the Core and execute the experiment
core = Core(agent, env, deadline)

for t in range(0, deadline):
    print t+1
    core.step()



plt.figure(3)
plt.plot(np.sum(agent.prevConversions, axis=1))

