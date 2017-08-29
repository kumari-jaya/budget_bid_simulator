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

probClick = np.array([0.5, 0.3, 0.4, 0.3])

# Auction setting
nBidders = np.array([5,6,6,5,5])
aAuction(nbidders=5, nslots=5,  mu=0.49 , sigma=0.2, lambdas=lambdas, myClickProb=probClick[0])
a2 = Auction(nbidders=6, nslots=5,  mu=0.33 , sigma=0.2,lambdas=lambdas, myClickProb=probClick[1])
a3 = Auction(nbidders=4, nslots=3, mu=0.79 , sigma=0.32, lambdas=lambdas, myClickProb=probClick[2])
a4 = Auction(nbidders=7, nslots=7,  mu=0.29 , sigma=0.2,lambdas=lambdas, myClickProb=probClick[3])

# Campaign setting
campaigns = []
campaigns.append(Campaign(a1, nMeanResearch=1000.0, nStdResearch=50.0, probClick=probClick[0], convParams=convparams))
campaigns.append(Campaign(a2, nMeanResearch=1500.0, nStdResearch=50.0, probClick=probClick[1], convParams=convparams))
campaigns.append(Campaign(a3, nMeanResearch=1500.0, nStdResearch=50.0, probClick=probClick[2], convParams=convparams))
campaigns.append(Campaign(a4, nMeanResearch=1250.0, nStdResearch=50.0, probClick=probClick[3], convParams=convparams))
nCampaigns = len(campaigns)

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

