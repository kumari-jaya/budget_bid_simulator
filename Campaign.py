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

    def __init__(self, auction, nusers=100, probClick=0.5, convParams=[]):
        self.nusers = nusers
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

    """
    def generateClicksAndCost(self, bid, budget):

        errore = np.random.randn()*self.clickParams[3]  # clickParams[3] è un parametro dipendente da nusers per definire la dev standard
        potentialClicks = self.clickParams[0]*(1 /(1+np.exp(-(self.clickParams[1]*(bid - self.clickParams[2]))))) + errore
        cpc = np.random.uniform(0.1, bid, int(potentialClicks))
        index = self.findIndex(cpc, budget)
        clicks = index+1
        cost = np.sum(cpc[0:index])
        hours = (clicks/potentialClicks)*24
        self.clicks = np.append(self.clicks, clicks)
        self.costs = np.append(self.costs, cost)
        self.hours = np.append(self.hours, hours)
        self.budget= np.append(self.budget,budget)
        self.bid= np.append(self.bid,bid)
        #print self.clicks
        #print self.costs
        #print "Costo per click medio: %f" % np.mean(cpc)
        #print "Numero click potenziali: %f" % potentialClicks
        #print self.hours
    """
    def findIndex(self, cpcArray, budget):

        for i in range(0,len(cpcArray)):

            if (np.sum(cpcArray[0:i]) > budget):
                return i-1
        return len(cpcArray)-1


    def generateClicksAndCost(self, bid, budget):
        [cpc,mypos,pobs] = self.auction.simulateMultipleAuctions(self.nusers,bid)
        nclicks = np.zeros(int(self.nusers))
        for i in range(0,int(self.nusers)):
            soglia = np.random.uniform(0,1)
            obsEvent = int(pobs[i] > soglia)
            soglia = np.random.uniform(0,1)
            clickEvent = int(self.probClick > soglia) * obsEvent
            nclicks[i] = clickEvent
        costTot = cpc * nclicks
        index = self.findIndex(costTot,budget)
        sumClicks = np.maximum(np.sum(nclicks),1.0)
        hours = (np.sum(nclicks[0:index])/sumClicks)*24
        #print "BID: ",bid
        #print "CLICKS: " ,np.sum(nclicks[0:index])
        self.clicks = np.append(self.clicks, np.sum(nclicks[0:index]))
        self.costs = np.append(self.costs, np.sum(costTot[0:index]))
        self.hours = np.append(self.hours, hours)



    def generateConversions(self):
        convs = math.floor(self.convParams[0]*self.clicks[len(self.clicks)-1])   #Conversioni uguali a probabilità di conversione per numero di clicks
        self.conversions = np.append(self.conversions,convs)

    def generateRevenues(self):
        rpc = np.random.uniform(self.convParams[1], self.convParams[2], int(self.conversions[len(self.conversions)-1]))
        revenues = np.sum(rpc)
        self.revenues = np.append(self.revenues,revenues)
        #print "Revenue per conversione media: %f" %np.mean(rpc)

    def generateObservations(self,bid,budget):
        self.generateClicksAndCost(bid,budget)
        self.generateConversions()
        self.generateRevenues()
        return[self.clicks[-1],self.conversions[-1],self.costs[-1],self.revenues[-1],self.hours[-1]]
