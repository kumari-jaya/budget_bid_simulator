#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:51:29 2017

@author: margherita
"""

import numpy as np
import math
from Auction import *


class Campaign:

    def __init__(self, auction, nMeanResearch=100, nStdResearch=50, probClick=0.5, convParams=np.array([0.5, 50, 200])):
        self.nMeanResearch = nMeanResearch
        self.nStdResearch = nStdResearch
        self.auction = auction
        self.probClick = probClick
        self.convParams = convParams
        self.clicks = np.array([])
        self.conversions = np.array([])
        self.hours = np.array([])  # orario esaurimento budget
        self.costs = np.array([])
        self.revenues = np.array([])
        self.budget = np.array([])
        self.bid = np.array([])

    def generateClicksAndCost(self, myBid, budget):

        if(myBid ==0 or budget ==0):
            self.clicks = np.append(self.clicks, 0.0)
            self.costs = np.append(self.costs, 0.0)
            self.hours = np.append(self.hours, 0)   # o 24???????????
            self.budget = np.append(self.budget, budget)
            self.bid = np.append(self.bid, myBid)

        else:
            nResearch = math.floor(np.random.randn(1) * self.nStdResearch + self.nMeanResearch)
            [cpc, mypos, pobs] = self.auction.simulateMultipleAuctions(int(nResearch), myBid)
            nResearch = np.maximum(nResearch,1)
            clickEvents = pobs * self.probClick > np.random.uniform(0, 1, int(nResearch))
            costTot = cpc * clickEvents.astype(int)

            # Computing the time of total consumption of the budget
            index = self.findIndex(costTot, budget)
            sumClicks = np.maximum(np.sum(clickEvents.astype(int)), 1.0)
            gainedClicks = np.sum(clickEvents[0:index].astype(int))
            hours = (gainedClicks / sumClicks) * 24

            #print "CLICKS: " , gainedClicks

            self.clicks = np.append(self.clicks, gainedClicks)
            self.costs = np.append(self.costs, np.sum(costTot[0:index]))
            self.hours = np.append(self.hours, hours)
            self.budget = np.append(self.budget, budget)
            self.bid = np.append(self.bid, myBid)


    def findIndex(self, cpcArray, budget):
        for i in range(0, len(cpcArray)):
            if (np.sum(cpcArray[0:i]) > budget):
                return i-1
        return len(cpcArray)-1

    def generateConversions(self):
        #Conversion equal to conversion probability times number of clicks
        convs = math.floor(self.convParams[0] * self.clicks[-1])
        self.conversions = np.append(self.conversions, convs)

    def generateRevenues(self):
        rpc = np.random.uniform(self.convParams[1], self.convParams[2], int(self.conversions[-1]))
        revenues = np.sum(rpc)
        self.revenues = np.append(self.revenues, revenues)

    def generateObservations(self, bid, budget):
        self.generateClicksAndCost(bid, budget)
        self.generateConversions()
        self.generateRevenues()
        return [self.clicks[-1], self.conversions[-1], self.costs[-1], self.revenues[-1], self.hours[-1]]
