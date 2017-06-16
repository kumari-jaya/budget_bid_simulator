#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from AgentAware import *
from Core import *
from Plotter import *
from matplotlib import pyplot as plt
import copy

convparams1=np.array([0.2,100,101])
convparams2=np.array([0.25,150,151])
# ho messo prob di conversione a 0.4 a caso,mentre 100 e 200 sono i due estremi della uniforme per generare le revenues
# 1 0.71 0.56 0.53 0.49 0.47 0.44 0.44 0.43 0.43
lambdas = np.array([1.0 ,0.71, 0.56, 0.53, 0.49, 0.47])
deadline=50
nExperiments =10
revenuesOpt = np.zeros((nExperiments,deadline))
revenuesTest = np.zeros((nExperiments,deadline))



nBids=10
nIntervals=10

a1 = Auction(nbidders=10, nslots=6, mu=0.21, sigma=0.1, lambdas=lambdas)
a2 = Auction(nbidders=10, nslots=6, mu=0.21, sigma=0.1, lambdas=lambdas)  # nbidders deve essere > nslots
c1 = Campaign(a1, nusers=1000.0, probClick=0.5, convParams=convparams1)
c2 = Campaign(a2, nusers=2000.0, probClick=0.35, convParams=convparams2)
c3 = Campaign(a2, nusers=2000.0, probClick=0.35, convParams=convparams2*0.98)

val1 = convparams1[0]*np.mean(convparams1[1:2])
val2 = convparams2[0]*np.mean(convparams2[1:2])
values = np.array([val1,val2])

campaigns = [c1,c2,c3]
env2 = Environment(copy.copy(campaigns))


agentAware = AgentAware(budgetTot=1000, deadline= deadline,environment=env2, ncampaigns=len(campaigns), nIntervals=nIntervals, nBids=nBids,maxBudget=100.0)
#agentAware.setValuesPerClick(values)

for e in range(0,nExperiments):
    agentOpt = copy.copy(agentAware)
    agentTest = Agent(budgetTot=1000, deadline= deadline, ncampaigns=len(campaigns), nIntervals=nIntervals, nBids=nBids,maxBudget=100.0)
    agentTest.initGPs()
    envOpt = Environment(copy.copy(campaigns))
    envTest =Environment(copy.copy(campaigns))
    coreOpt = Core(agentOpt, envOpt, deadline)
    coreTest = Core(agentTest,envTest,deadline)

    #core.step()
    #core.step()
    coreOpt.runEpisode()
    coreTest.runEpisode()
    #agent.plotGP(0,fixedBid=True,bid=0.1111)
    #agent.plotGP(0,fixedBid=True,bid=0.55555)
    #agent.plotGP(0,fixedBid=True,bid=1.0)

    #agent.plotGP(0,fixedBid=False)


    #plotter = Plotter(agent)
    #plotter.plotGP(0,fixedBid=True,bid=0.55555)
    revenuesOpt[e,:]=np.sum(agentOpt.revenues,axis=1)
    revenuesTest[e,:]=np.sum(agentTest.revenues,axis=1)
plt.figure(4)
plt.plot(np.mean(revenuesOpt,axis=0))
plt.plot(np.mean(revenuesTest,axis=0))
plt.ylim(-10000,25000)