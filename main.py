#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from Agent import *
from Core import *
from Plotter import *
from matplotlib import pyplot as plt


convparams1=np.array([0.2,100,101])
convparams2=np.array([0.25,150,151])
# ho messo prob di conversione a 0.4 a caso,mentre 100 e 200 sono i due estremi della uniforme per generare le revenues
# 1 0.71 0.56 0.53 0.49 0.47 0.44 0.44 0.43 0.43
lambdas = np.array([1.0 ,0.71, 0.56, 0.53, 0.49, 0.47])
deadline=15

a1= Auction(nbidders=10 , nslots=6, mu=0.21 , sigma=0.1, lambdas=lambdas)
a2= Auction(nbidders=10 , nslots=6, mu=0.21 , sigma=0.1, lambdas=lambdas) # nbidders deve essere > nslots
c1 = Campaign(a1, nusers=1000.0 , probClick=0.5 ,convParams= convparams1)
c2 = Campaign(a2,nusers=1500.0,probClick= 0.5,convParams=convparams2)



nBids=10
nIntervals=20
agent = Agent(budgetTot=1000, deadline= deadline, ncampaigns=2, nIntervals=nIntervals, nBids=nBids,maxBudget=100.0)
agent.initGPs()
env = Environment([c1,c2])
core = Core(agent, env, deadline)

#core.step()
core.runEpisode()
#agent.plotGP(0,fixedBid=True,bid=0.1111)
#agent.plotGP(0,fixedBid=True,bid=0.55555)
#agent.plotGP(0,fixedBid=True,bid=1.0)

#agent.plotGP(0,fixedBid=False)
plt.figure(4)
plt.plot(np.sum((agent.revenues - agent.costs),axis=1))


plotter = Plotter(agent)
plotter.plotGP(0,fixedBid=True,bid=0.55555)
plotter.plotGP(0,fixedBid=True,bid=0.11111)
plotter.plotGP(0,fixedBid=True,bid=1.0)