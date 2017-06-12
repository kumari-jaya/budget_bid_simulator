#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from numpy import matlib
import math
from Campaign import *
from Environment import *
from Auction import *
from matplotlib import pyplot as plt


convparams=np.array([0.4,100,200]) #inutili per lo scopo di questo script ma sono necessari
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

a1= Auction(nbidders=5 , nslots=5, mu=0.31 , sigma=0.2, lambdas=lambdas)
a2= Auction(nbidders=6 , nslots=5, mu=0.67 , sigma=0.4, lambdas=lambdas)

ncampaigns=2
c1 = Campaign(a1, nusers=1000.0 , probClick=0.5 ,convParams= convparams)
c2 = Campaign(a2, nusers=1500.0 , probClick=0.6 ,convParams= convparams)

env = Environment([c1,c2])

## setto la bid che voglio usare e mi creo un array di budgets
bid = 0.65
budgets = np.linspace(0,200,100)

for i,b in enumerate(budgets):
    vettBids = np.matlib.repmat(bid,1,ncampaigns).reshape(-1)
    vettBudgets = np.matlib.repmat(b,1,ncampaigns).reshape(-1)
    observations = env.generateObservationsforCampaigns(vettBids,vettBudgets)
    if i == 0:
        clicks = np.array([observations[0]])
    else:
        clicks = np.append(clicks, [observations[0]],axis=0)
    print "Iterazione: ",i+1

# faccio il plot della funzione vera
fig = plt.figure()

plt.plot(budgets,clicks[:,0] , 'r-', label=u'Campaign 1')
plt.plot(budgets,clicks[:,1] , 'b-', label=u'Campaign 2')

plt.xlabel('Budget')
plt.ylabel('Clicks')
#plt.ylim(-10, np.max(self.prevClicks[:,gpIndex])*1.5)
plt.legend(loc='upper left')
plt.show()
