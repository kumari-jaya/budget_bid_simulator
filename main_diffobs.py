#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from Agent import *
from Core import *
from matplotlib import pyplot as plt
from Plotter import *

convparams=np.array([0.4,100,200])
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])


a1= Auction(nbidders=5 , nslots=5, mu=0.51 , sigma=0.2, lambdas=lambdas)
a2= Auction(nbidders=6 , nslots=5, mu=0.67 , sigma=0.4, lambdas=lambdas)


ncampaigns=2
c1 = Campaign(a1, nusers=1000.0 , probClick=0.5 ,convParams= convparams)
c2 = Campaign(a2, nusers=1500.0 , probClick=0.6 ,convParams= convparams)

env = Environment([c1,c2])


nBids=10
nIntervals=100


bid1 = 0.1111
bid2 = 0.55555
bid3 = 0.65
bid3b = 0.77
bid4=  1.0

deadline = 20
#deadline = 120
#deadline = 150
#deadline = 180
#deadline = 200
#deadline = 250


maxBudget = 200
agent = Agent(1000, deadline, 2,nIntervals,nBids,maxBudget)
agent.initGPs()
core = Core(agent, env, deadline)
core.runEpisode()
"""
plotter = Plotter(agent=agent,env=env)
[clicks,budgets] = plotter.trueSample(bid2,maxBudget,nsimul=10)
#campagna 1
#### uguale a plotGP ma gli passo in fondo i veri valori per plottarli insieme
plotter.plotGPComparison(0,clicks,budgets,fixedBid=True,bid=bid2)
#agent.plotGP(0,fixedBid=True,bid=bid2)
#agent.plotGP(0,fixedBid=True,bid=bid3)
#agent.plotGP(0,fixedBid=True,bid=bid4)

#campagna 2
plotter.plotGPComparison(1,clicks,budgets,fixedBid=True,bid=bid2)
#agent.plotGP(1,fixedBid=True,bid=bid2)
#agent.plotGP(1,fixedBid=True,bid=bid3b)
#agent.plotGP(1,fixedBid=True,bid=bid4)



"""


"""
deadline = np.linspace(100,300,15)

for i,d in enumerate(deadline):

    print "Deadline: ",d
    agent = Agent(1000, int(d), 2,nIntervals,nBids,100)
    agent.initGPs()
    core = Core(agent, env, int(d))
    core.runEpisode()

    #campagna 1
    #agent.plotGP(0,fixedBid=True,bid=bid1)
    agent.plotGP(0,fixedBid=True,bid=bid2)
    #agent.plotGP(0,fixedBid=True,bid=bid3)
    #agent.plotGP(0,fixedBid=True,bid=bid4)


    #campagna 2
    #agent.plotGP(1,fixedBid=True,bid=bid1)
    #agent.plotGP(1,fixedBid=True,bid=bid2)
    #agent.plotGP(1,fixedBid=True,bid=bid3)
    #agent.plotGP(1,fixedBid=True,bid=bid4)

"""
