# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:44:49 2017

@author: alessandro
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#from matplotlib import pyplot as plt

class Agent:

    def __init__(self,budgetTot,deadline,ncampaigns,nIntervals,nBids,maxBudget=100.0,maxBid=1.0):
        self.budgetTot = budgetTot
        self.deadline = deadline
        self.ncampaigns = ncampaigns
        self.costs = np.zeros(ncampaigns)
        self.revenues = np.zeros(ncampaigns)
        self.t = 0
        self.gps = []
        self.prevBudgets = np.array([])
        self.prevBids = np.array([])
        self.prevClicks = np.array([])
        self.prevConversions = np.array([])
        self.prevHours = np.array([])
        self.valuesPerClick = np.zeros(ncampaigns)

        self.maxTotDailyBudget = maxBudget
        self.maxBid = maxBid
        self.nBudgetIntervals = nIntervals
        self.nBids = nBids

        self.budgets = np.linspace(0, self.maxTotDailyBudget,nIntervals)
        self.bids = np.linspace(0,self.maxBid,nBids)
        self.optimalBidPerBudget = np.zeros(ncampaigns,nIntervals)


    def updateValuesPerClick(self):
        for c in range(0,self.ncampaigns):
            self.valuesPerClick[c] = np.sum(self.prevConversions[:,c])/np.sum(self.prevClicks[:,c])

    def initGPs(self):
        for c in range(0,self.ncampaigns):
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            alpha=1e-10
            self.gps.append(GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10))


    def updateGP(self,c):
        self.prevBids=np.atleast_2d(self.prevBids)
        self.prevBudgets=np.atleast_2d(self.prevBudgets)
        self.prevClicks=np.atleast_2d(self.prevClicks)
        X=np.array([self.prevBids.T[c,:],self.prevBudgets.T[c,:]])
        X=np.atleast_2d(X).T
        potentialClicks = self.prevClicks * 24.0/self.prevHours
        y=potentialClicks.T[c,:].ravel()
        #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        #alpha=0.0
        #self.gps[c] = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10)
        self.gps[c].fit(X, y)

    def updateMultiGP(self):
        for c in range(0,self.ncampaigns):
            self.updateGP(c)

    def updateState(self,bids,budgets,clicks,conversions,costs,revenues,hours):
        bids=np.atleast_2d(bids)
        budgets=np.atleast_2d(budgets)
        clicks=np.atleast_2d(clicks)
        conversions=np.atleast_2d(conversions)
        hours=np.atleast_2d(hours)
        #print np.shape(bids)
        """
        self.prevBids=np.atleast_2d(self.prevBids)
        self.prevBudgets=np.atleast_2d(self.prevBudgets)
        self.prevClicks=np.atleast_2d(self.prevClicks)
        self.prevConversions=np.atleast_2d(self.prevConversions)
        self.prevHours=np.atleast_2d(self.prevHours)
        """
        print bids
        if(self.t==0):
            self.prevBudgets = budgets
            self.prevBids = bids
            self.prevClicks = clicks
            self.prevConversions = conversions
            self.prevHours = hours
        else:
            self.prevBudgets = np.append(self.prevBudgets,budgets, axis=0)
            self.prevBids = np.append(self.prevBids,bids, axis=0)
            self.prevClicks = np.append(self.prevClicks,clicks, axis=0)
            self.prevConversions = np.append(self.prevConversions,conversions, axis=0)
            self.prevHours = np.append(self.prevHours,hours, axis=0)
        self.updateMultiGP()
        self.costs += costs
        self.revenues += revenues
        self.t +=1

    def valueForBudget(self,itemIdx,budget):
        idx = np.argwhere(budget>=self.budgets)
        #print "Indici: ",idx
        #print "maxIndice",idx.max()
        return values[itemIdx,idx.max()]

    def firstRow(self,values):
        firstRow = np.zeros(len(self.budgets)).tolist()
        for i,b in enumerate(self.budgets):
            firstRow[i]=[[values[0,i]],[0],[b]]
        return firstRow

    def optimize(self,values):
        valIdx = 0
        itIdx = 1
        bIdx = 2
        h[0] = firstRow(values)


        for i in range(1,self.ncampaigns):
            for j,b in enumerate(self.budgets):
                h[i][j] = h[i-1][j][:]
                maxVal = 0
                for bi in range(0,j+1):
                    #print (np.sum(h[i-1][valIdx]) + valueForBudget(i,b - budgets[bi]))
                    #print maxVal

                    if ((np.sum(h[i-1][bi][valIdx]) + valueForBudget(i,b - self.budgets[bi])) >maxVal):
                        val = h[i-1][bi][valIdx][:]
                        val.append(valueForBudget(i,b - self.budgets[bi]))
                        newValues = val[:]
                        #print newValues
                        #print valueForBudget(i,b - budgets[bi])
                        items = h[i-1][bi][itIdx][:]
                        items.append(i)
                        newItems = items[:]
                        print newItems
                        selBudgets = h[i-1][bi][bIdx][:]
                        selBudgets.append(b - self.budgets[bi])
                        newSelBudgets = selBudgets[:]
                        h[i][j]=[newValues,newItems,newSelBudgets]
                        maxVal = np.sum(newValues)
        newBudgets=h[-1][-1][2]
        newCampaigns=h[-1][-1][1]
        return [newBudgets,newCampaigns]

    def valuesForCampaigns(self):
        values = np.zeros(shape=(self.ncampaigns, len(self.budgets)))
        for c in range(0,self.ncampaigns):
            for b,j in enumerate(self.budgets):
                x= np.array([np.matlib.repmat(b,1,self.nBids),self.bids.T])
                valuesforBids=self.gps[c].predict(x)
                self.optimalBidPerBudget[c,j] = self.bids[np.argmax(valuesforBids)]
                values[c,j] = valuesforBids.max()
        return values


    def chooseAction(self):
        values = valuesForCampaigns()
        [newBudgets,newCampaigns] = optimize(values)
        finalBudgets = np.zeros(self.ncampaigns)
        finalBids = np.zeros(self.ncampaigns)
        for i,c in enumerate(newCampaigns):
            finalBudgets[c] = newBudgets[i]
            idx = np.argwhere(self.budgets == newBudgets[i]).reshape(-1)
            finalBids[c] = self.optimalBidPerBudget[c,idx]
        return [finalBudgets,finalBids]
