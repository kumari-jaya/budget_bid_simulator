#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# from Agent import *
from lost_and_found.environment.Core import *
from lost_and_found.environment.Oracle import *

from agent.AgentFactored import *
from graphicalTool.Plotter import *

show = False
save = False
path = '../results/'

# Auction parameter initialization
convparams = np.array([0.4, 100, 200])
lambdas1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
lambdas2 = np.array([0.9, 0.8, 0.7])
lambdas3 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
probClick = np.array([0.5, 0.3, 0.4, 0.3])

# Auction setting
a1 = Auction(nBidders=5, nslots=5, mu=7.49, sigma=0.2, lambdas=lambdas1, myClickProb=probClick[0])
a2 = Auction(nBidders=6, nslots=5, mu=0.33, sigma=0.2, lambdas=lambdas1, myClickProb=probClick[1])
a3 = Auction(nBidders=4, nslots=3, mu=0.79, sigma=0.32, lambdas=lambdas2, myClickProb=probClick[2])
a4 = Auction(nBidders=7, nslots=7, mu=0.29, sigma=0.2, lambdas=lambdas3, myClickProb=probClick[3])

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

# Execute the experiments
core = Core(agent, env, deadline)
core.runEpisode()

for t in range(0, deadline):
    print t+1
    agent.gpsClicks = oracle.gpsClicks
    agent.gpsCosts = oracle.gpsCosts


    #plotter.plotGP_prior(0, fixedBid=True, bid=1.0, y_min=0,y_max=200)
    #plt.savefig('myfig_'+str(t))

    #plotter.plotGPComparison( 0, trueClicks, trueBudgets, fixedBid=False, bid=1)

#plotter.trueSample(1.0,maxBudget=100.0,nsimul=1)

#agent.plotGP(0,fixedBid=False)
#plt.plot(np.sum((agent.revenues - agent.costs),axis=1))


#trueFun = plotter.trueSample(1.0,maxBudget=100.0,nsimul=2)
#plotter.plotGP_prior(0,fixedBid=True,bid=0.11111)

#plotter.plotGP_prior(0,fixedBid=True,bid=1.0)

#plotter.plotGP(0,fixedBid=True,bid=0.11111)
plt.figure(3)
plt.plot(np.sum(agent.prevConversions, axis=1))

"""
plt.ylim(-10,600)
plotter.plotClicksAgentBid(1)

plotter.plotCostsAgentBid(1)
#plotter.plotCostsAgentBid(2)
"""