#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from numpy import matlib
import math
from Campaign import *
from Environment import *
from Auction import *
from matplotlib import pyplot as plt


def trueSample(env,bid,maxBudget):
    ncampaigns = len(env.campaigns)
    budgets = np.linspace(0,maxBudget,200)

    for i,b in enumerate(budgets):
        vettBids = np.matlib.repmat(bid,1,ncampaigns).reshape(-1)
        vettBudgets = np.matlib.repmat(b,1,ncampaigns).reshape(-1)
        observations = env.generateObservationsforCampaigns(vettBids,vettBudgets)
        if i == 0:
            clicks = np.array([observations[0]])
        else:
            clicks = np.append(clicks, [observations[0]],axis=0)
        if i%10 == 0:
            print "Simulation ",i," out of 200"
    return [clicks,budgets]
    """
    # faccio il plot della funzione vera
    fig = plt.figure()

    plt.plot(budgets,clicks[:,0] , 'r-', label=u'Campaign 1')
    plt.plot(budgets,clicks[:,1] , 'b-', label=u'Campaign 2')

    plt.xlabel('Budget')
    plt.ylabel('Clicks')
    #plt.ylim(-10, np.max(self.prevClicks[:,gpIndex])*1.5)
    plt.legend(loc='upper left')
    plt.show()
    """
