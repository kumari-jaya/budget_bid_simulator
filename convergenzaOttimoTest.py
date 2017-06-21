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


convparams=np.array([0.4,100,200])
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

deadline=100
nExperiments = 2
revenuesOpt = np.zeros((nExperiments,deadline))
revenuesTest = np.zeros((nExperiments,deadline))



nBids=5
nIntervals=9

a1= Auction(nbidders=5 , nslots=5, mu=0.61 , sigma=0.2, lambdas=lambdas)
a2= Auction(nbidders=6 , nslots=5, mu=0.67 , sigma=0.4, lambdas=lambdas)
a3= Auction(nbidders=8 , nslots=5, mu=0.47 , sigma=0.25, lambdas=lambdas)
a4= Auction(nbidders=5 , nslots=5, mu=0.57 , sigma=0.39, lambdas=lambdas)

campaigns=[]
campaigns.append(Campaign(a1, nusers=1000.0 , probClick=0.5 ,convParams= convparams))
campaigns.append(Campaign(a2, nusers=1500.0 , probClick=0.6 ,convParams= convparams))
campaigns.append(Campaign(a3, nusers=1500.0 , probClick=0.6 ,convParams= convparams))
campaigns.append(Campaign(a2, nusers=1000.0 , probClick=0.5 ,convParams= convparams))
campaigns.append( Campaign(a4, nusers=1250.0 , probClick=0.4 ,convParams= convparams))



env2 = Environment(copy.copy(campaigns))


agentAware = AgentAware(budgetTot=1000, deadline= deadline,environment=env2, ncampaigns=len(campaigns), nIntervals=nIntervals, nBids=nBids,maxBudget=100.0)
#agentAware.setValuesPerClick(values)

for e in range(0,nExperiments):
    print "\n"
    print "Experiment: ", e
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

np.save('revOpt5_2exp',revenuesOpt)
np.save('revTest5_2exp',revenuesTest)
plt.figure(4)
plt.plot(np.mean(revenuesOpt,axis=0))
plt.plot(np.mean(revenuesTest,axis=0))

plt.ylim(-10000,45000)