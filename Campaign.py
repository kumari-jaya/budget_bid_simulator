#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:51:29 2017

@author: margherita
"""

import numpy as np
import math

class Campaign:

    def __init__(self, nusers=100, clickParams=[], convParams=[]):
        self.nusers = nusers
        self.clickParams = clickParams
        self.convParams = convParams
        self.clicks = np.array([])
        self.conversions = np.array([])
        self.hours = np.array([])  # orario esaurimento budget
        self.costs = np.array([])
        self.revenues = np.array([])
        self.budget = np.array([])
        self.bid = np.array([])


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
        #print self.clicks
        #print self.costs
        print "Costo per click medio: %f" % np.mean(cpc)
        print "Numero click potenziali: %f" % potentialClicks
        #print self.hours

    def findIndex(self, cpcArray, budget):
        for i in range(0,len(cpcArray)):
            if (np.sum(cpcArray[0:i]) > budget):
                return i-1
        return len(cpcArray)-1

    def generateConversions(self):
        convs = math.floor(self.convParams[0]*self.clicks[len(self.clicks)-1])   #Conversioni uguali a probabilità di conversione per numero di clicks
        self.conversions = np.append(self.conversions,convs)

    def generateRevenues(self):
        rpc = np.random.uniform(self.convParams[1], self.convParams[2], int(self.conversions[len(self.conversions)-1]))
        revenues = np.sum(rpc)-self.costs[len(self.costs)-1]
        self.revenues = np.append(self.revenues,revenues)
        print "Revenue per conversione media: %f" %np.mean(rpc)

clickParams=np.array([1000.0,0.2,30.0, 0.1])
convparams=np.array([0.4,100,200])
# ho messo prob di conversione a 0.4 a caso,mentre 100 e 200 sono i due estremi della uniforme per generare le revenues
c = Campaign(10000.0,clickParams,convparams)
c.generateClicksAndCost(40.0,16300.0)
c.generateConversions()
c.generateRevenues()
print "Numero clicks: %f" % c.clicks
print "Costi: %f" % c.costs
print "Ora esaurimento: %f" % c.hours
print "Numero conversioni: %f" % c.conversions
print "Totale revenues: %f" % c.revenues
