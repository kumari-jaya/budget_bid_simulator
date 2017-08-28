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
from AgentAle import *


convparams1=np.array([0.2, 100, 101])
convparams2=np.array([0.25, 150, 151])
# ho messo prob di conversione a 0.4 a caso,mentre 100 e 200 sono i due estremi della uniforme per generare le revenues
# 1 0.71 0.56 0.53 0.49 0.47 0.44 0.44 0.43 0.43
lambdas = np.array([1.0 ,0.71, 0.56, 0.53, 0.49, 0.47])
deadline=30
#lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

convparams=np.array([0.4,100,200])
a1= Auction(nbidders=5 , nslots=5, mu=0.59 , sigma=0.2, lambdas=lambdas)
a2= Auction(nbidders=6 , nslots=5, mu=0.67 , sigma=0.4, lambdas=lambdas)
a3= Auction(nbidders=6 , nslots=5, mu=0.47 , sigma=0.25, lambdas=lambdas)
a4= Auction(nbidders=5 , nslots=5, mu=0.57 , sigma=0.39, lambdas=lambdas)


c=[]
c.append(Campaign(a1, nUsers=1000.0, probClick=0.5, convParams= convparams))
c.append(Campaign(a2, nUsers=1500.0, probClick=0.6, convParams= convparams))
c.append(Campaign(a3, nUsers=1500.0, probClick=0.6, convParams= convparams))
c.append(Campaign(a2, nUsers=1000.0, probClick=0.5, convParams= convparams))
#c.append(Campaign(a4, nusers=1250.0 , probClick=0.4 ,convParams= convparams))
ncampaigns=len(c)

nBids=10
nIntervals=10
agent = AgentMarcello(budgetTot=1000, deadline= deadline, ncampaigns=ncampaigns, nIntervals=nIntervals, nBids=nBids,maxBudget=100.0)
agent.initGPs()
env = Environment(c)
plotter = Plotter(agent,env=env)
"""
[trueClicks,trueBudgets] = plotter.trueSample(1.0,maxBudget=100.0,nsimul=400)
trueClicks = np.array([trueClicks])
trueBudgets = np.array([trueBudgets])
np.save('presentazione/trueClikcs',trueClicks)
np.save('presentazione/trueBudgets',trueBudgets)
#trueFun = np.load("presentazione/trueFun.py")
"""


core = Core(agent, env, deadline)

for t in range(0,deadline):
    print t
    core.step()

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
plt.plot(np.sum(agent.prevClicks,axis=1))

"""
plt.ylim(-10,600)
plotter.plotClicksAgentBid(1)

plotter.plotCostsAgentBid(1)
#plotter.plotCostsAgentBid(2)
"""