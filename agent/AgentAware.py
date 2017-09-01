# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:44:49 2017

@author: alessandro
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt


class AgentAware:

    def __init__(self,budgetTot,deadline,ncampaigns,environment,nIntervals,nBids,maxBudget=100.0,maxBid=1.0):
        self.budgetTot = budgetTot
        self.deadline = deadline
        self.ncampaigns = ncampaigns
        self.costs = np.array([])
        self.revenues = np.array([])
        self.t = 0
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

        self.budgets = np.linspace(0, maxBudget,nIntervals)
        self.bids = np.linspace(0,maxBid,nBids)
        self.optimalBidPerBudget = np.zeros((ncampaigns,nIntervals))
        self.environment = environment
        self.ymax=np.ones((ncampaigns))
        self.campaignsValues=self.valuesForCampaigns()

    def updateValuesPerClick(self):
        for c in range(0,self.ncampaigns):
            convParams = self.environment.campaigns[c].convParams
            value = convParams[0]*np.mean(convParams[1:2])
            if(np.isnan(value) or np.isinf(value)):
                self.valuesPerClick[c] = 0
            else:
                self.valuesPerClick[c] = value
        return self.valuesPerClick


    def setValuesPerClick(self,values):
        self.valuesPerClick=values

    def divideFloat(self,numerator,denominator):
        div = numerator/denominator
        div[np.isnan(div)]=0
        div[np.isinf(div)]=0
        return div



    def updateState(self,bids,budgets,clicks,conversions,costs,revenues,hours):
        bids=np.atleast_2d(bids)
        budgets=np.atleast_2d(budgets)
        clicks=np.atleast_2d(clicks)
        conversions=np.atleast_2d(conversions)
        hours=np.atleast_2d(hours)
        costs=np.atleast_2d(costs)
        revenues=np.atleast_2d(revenues)
        """
        self.prevBids=np.atleast_2d(self.prevBids)
        self.prevBudgets=np.atleast_2d(self.prevBudgets)
        self.prevClicks=np.atleast_2d(self.prevClicks)
        self.prevConversions=np.atleast_2d(self.prevConversions)
        self.prevHours=np.atleast_2d(self.prevHours)
        """
        if(self.t==0):
            self.prevBudgets = budgets
            self.prevBids = bids
            self.prevClicks = clicks
            self.prevConversions = conversions
            self.prevHours = hours
            self.costs = costs
            self.revenues = revenues
        else:
            self.prevBudgets = np.append(self.prevBudgets,budgets, axis=0)
            self.prevBids = np.append(self.prevBids,bids, axis=0)
            self.prevClicks = np.append(self.prevClicks,clicks, axis=0)
            self.prevConversions = np.append(self.prevConversions,conversions, axis=0)
            self.prevHours = np.append(self.prevHours,hours, axis=0)
            self.costs = np.append(self.costs, costs, axis=0)
            self.revenues = np.append(self.revenues, revenues, axis=0)
        self.updateValuesPerClick()
        self.t +=1

    def valueForBudget(self,itemIdx,budget,values):
        idx = np.argwhere(budget>=self.budgets)
        return values[itemIdx,idx.max()]

    def firstRow(self,values):
        firstRow = np.zeros(len(self.budgets)).tolist()
        for i,b in enumerate(self.budgets):
            firstRow[i]=[[values[0,i]],[0],[b]]
        return firstRow

    def valuesForCampaigns(self):
        self.updateValuesPerClick()
        estimatedClicks = np.zeros(shape=(self.ncampaigns, len(self.budgets)))
        for c in range(0, self.ncampaigns):
            for j, budget in enumerate(self.budgets):
                estimatedClicksforBids = np.zeros(len(self.bids))
                for k,bid in enumerate(self.bids):
                    nExperiments = 1
                    estimatedClicksForBidForExperiment= np.zeros(nExperiments)
                    for e in range(0,nExperiments):
                        clicks = self.environment.campaigns[c].generateObservations(bid,budget)[0]
                        estimatedClicksForBidForExperiment[e] = clicks
                    estimatedClicksforBids[k] = np.mean(estimatedClicksForBidForExperiment)
                # if self.t>2:self.plotGP(c)
                idxs = np.argwhere(estimatedClicksforBids == estimatedClicksforBids.max()).reshape(-1)
                idx = np.random.choice(idxs)
                self.optimalBidPerBudget[c, j] = self.bids[idx]
                estimatedClicks[c, j] = estimatedClicksforBids.max()
        self.campaignsValues = estimatedClicks * self.valuesPerClick.reshape((self.ncampaigns, 1))
        return estimatedClicks * self.valuesPerClick.reshape((self.ncampaigns, 1))  # TESTARE !!!


    def optimize(self,values):
        valIdx = 0
        itIdx = 1
        bIdx = 2
        h = np.zeros(shape=(self.ncampaigns,len(self.budgets)))
        h=h.tolist()
        h[0] = self.firstRow(values)
        for i in range(1,self.ncampaigns):
            for j,b in enumerate(self.budgets):
                h[i][j] = h[i-1][j][:]
                maxVal = 0
                for bi in range(0,j+1):
                    if ((np.sum(h[i-1][bi][valIdx]) + self.valueForBudget(i,b - self.budgets[bi],values)) >maxVal):
                        val = h[i-1][bi][valIdx][:]
                        val.append(self.valueForBudget(i,b - self.budgets[bi],values))
                        newValues = val[:]
                        items = h[i-1][bi][itIdx][:]
                        items.append(i)
                        newItems = items[:]
                        selBudgets = h[i-1][bi][bIdx][:]
                        selBudgets.append(b - self.budgets[bi])
                        newSelBudgets = selBudgets[:]
                        h[i][j]=[newValues,newItems,newSelBudgets]
                        maxVal = np.sum(newValues)
        newBudgets=h[-1][-1][2]
        newCampaigns=h[-1][-1][1]
        return [newBudgets,newCampaigns]



    def chooseAction(self,sampling=False, fixedBid=False, fixedBudget=False, fixedBidValue=1.0, fixedBudgetValue=1000.0):

        values = self.campaignsValues
        [newBudgets,newCampaigns] = self.optimize(values)
        finalBudgets = np.zeros(self.ncampaigns)
        finalBids = np.zeros(self.ncampaigns)
        for i,c in enumerate(newCampaigns):
            finalBudgets[c] = newBudgets[i]
            idx = np.argwhere(np.isclose(self.budgets,newBudgets[i])).reshape(-1)
            finalBids[c] = self.optimalBidPerBudget[c,idx]
        return [finalBudgets,finalBids]


    def normalizeBudgetArray(self,budgetArray):
        return budgetArray/self.maxTotDailyBudget

    def normalizeBidsArray(self,bidsArray):
        return bidsArray/self.maxBid


    def normalize(self,X):
        X[:,0] = X[:,0]/(self.maxBid)
        X[:,1] = X[:,1]/(self.maxTotDailyBudget)
        return X

    def normalizeOutput(self,y,campaign):
        return y
        if(y.max()!=0):
            self.ymax[campaign] = y.max()
            y=y/y.max()
        return y


    def denormalizeOutput(self,y,campaign):
        return y
        return y*self.ymax[campaign]




    def findBestBidPerBudget(self,budget,bidsArray,gpIndex):
        x = np.array([bidsArray.T, np.matlib.repmat(budget, 1, len(bidsArray)).reshape(-1)])
        x = np.atleast_2d(x).T
        x = self.normalize(x)
        valuesforBids = self.denormalizeOutput(self.gps[gpIndex].predict(x), gpIndex)
        idxs = np.argwhere(valuesforBids == valuesforBids.max()).reshape(-1)
        idx = np.random.choice(idxs)
        return bidsArray[idx]


    def bestBidsPerBudgetsArray(self,budgetArray,bidsArray,gpIndex):
        bestBidsArray= np.zeros(len(budgetArray))
        for i,b in enumerate(budgetArray):
            bestBidsArray[i]=self.findBestBidPerBudget(b,bidsArray,gpIndex)
        return bestBidsArray
