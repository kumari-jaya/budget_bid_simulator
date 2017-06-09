# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:44:49 2017

@author: alessandro
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt


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

        self.budgets = np.linspace(0, maxBudget,nIntervals)
        self.bids = np.linspace(0,maxBid,nBids)
        self.optimalBidPerBudget = np.zeros((ncampaigns,nIntervals))

        self.campaignsValues = []
        self.ymax=np.ones((ncampaigns))


    def updateValuesPerClick(self):
        for c in range(0,self.ncampaigns):
            self.valuesPerClick[c] = np.sum(self.prevConversions[:,c])/np.sum(self.prevClicks[:,c])

    def initGPs(self):
        for c in range(0,self.ncampaigns):
            #C(1.0, (1e-3, 1e3))
            kernel = C(1.0, (1e-3, 1e1))*RBF(200, (100, 300))
            alpha=100
            self.gps.append(GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10,normalize_y=True))

    def dividePotentialClicks(self,numerator,denominator):
        div = numerator/denominator
        div[np.isnan(div)]=0
        div[np.isinf(div)]=0
        return div

    def updateGP(self,c):
        self.prevBids=np.atleast_2d(self.prevBids)
        self.prevBudgets=np.atleast_2d(self.prevBudgets)
        self.prevClicks=np.atleast_2d(self.prevClicks)
        X=np.array([self.prevBids.T[c,:],self.prevBudgets.T[c,:]])
        X=np.atleast_2d(X).T
        X=self.normalize(X)
        potentialClicks = self.dividePotentialClicks(self.prevClicks * 24.0, self.prevHours)
        y=potentialClicks.T[c,:].ravel()
        y=self.normalizeOutput(y,c)
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
        #print bids
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
        self.updateValuesPerClick()
        self.t +=1

    def valueForBudget(self,itemIdx,budget,values):
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
        h = np.zeros(shape=(self.ncampaigns,len(self.budgets)))
        h=h.tolist()
        h[0] = self.firstRow(values)


        for i in range(1,self.ncampaigns):
            for j,b in enumerate(self.budgets):
                h[i][j] = h[i-1][j][:]
                maxVal = 0
                for bi in range(0,j+1):
                    #print (np.sum(h[i-1][valIdx]) + valueForBudget(i,b - budgets[bi]))
                    #print maxVal

                    if ((np.sum(h[i-1][bi][valIdx]) + self.valueForBudget(i,b - self.budgets[bi],values)) >maxVal):
                        val = h[i-1][bi][valIdx][:]
                        val.append(self.valueForBudget(i,b - self.budgets[bi],values))
                        newValues = val[:]
                        #print newValues
                        #print valueForBudget(i,b - budgets[bi])
                        items = h[i-1][bi][itIdx][:]
                        items.append(i)
                        newItems = items[:]
                        #print newItems
                        selBudgets = h[i-1][bi][bIdx][:]
                        selBudgets.append(b - self.budgets[bi])
                        newSelBudgets = selBudgets[:]
                        h[i][j]=[newValues,newItems,newSelBudgets]
                        maxVal = np.sum(newValues)
        newBudgets=h[-1][-1][2]
        newCampaigns=h[-1][-1][1]
        return [newBudgets,newCampaigns]

    def valuesForCampaigns(self,sampling=False):
        values = np.zeros(shape=(self.ncampaigns, len(self.budgets)))
        if( sampling==False):
            for c in range(0,self.ncampaigns):
                for j,b in enumerate(self.budgets):
                    x= np.array([self.bids.T,np.matlib.repmat(b,1,self.nBids).reshape(-1)])
                    x=np.atleast_2d(x).T
                    x = self.normalize(x)
                    valuesforBids=self.denormalizeOutput(self.gps[c].predict(x),c)
                    idxs = np.argwhere(valuesforBids == valuesforBids.max()).reshape(-1)
                    idx = np.random.choice(idxs)
                    self.optimalBidPerBudget[c,j] = self.bids[idx]
                    values[c,j] = valuesforBids.max()
            self.campaignsValues=values
            return values
        else:
            for c in range(0, self.ncampaigns):
                for j, b in enumerate(self.budgets):
                    x= np.array([self.bids.T,np.matlib.repmat(b,1,self.nBids).reshape(-1)])
                    x = np.atleast_2d(x).T
                    x = self.normalize(x)
                    print x
                    [means,sigmas] = self.gps[c].predict(x,return_std=True)
                    means = self.denormalizeOutput(means,c)
                    sigmas = self.denormalizeOutput(sigmas,c)
                    valuesForBids = np.random.normal(means,sigmas)
                    #print valuesForBids
                    idxs = np.argwhere(valuesForBids == valuesForBids.max()).reshape(-1)
                    idx = np.random.choice(idxs)
                    print "best Bid",self.bids[idx]
                    self.optimalBidPerBudget[c, j] = self.bids[idx]
                    values[c, j] = valuesForBids.max()
            self.campaignsValues=values

            return values








    def chooseAction(self):

        finalBudgets = np.zeros(self.ncampaigns)
        finalBids = np.zeros(self.ncampaigns)
        for i in range(0,self.ncampaigns):
            finalBudgets[i] = np.random.choice(self.budgets)
            finalBids[i] = np.random.choice(self.bids)
        """
        values = self.valuesForCampaigns(sampling=False)
        [newBudgets,newCampaigns] = self.optimize(values)
        finalBudgets = np.zeros(self.ncampaigns)
        finalBids = np.zeros(self.ncampaigns)
        for i,c in enumerate(newCampaigns):
            finalBudgets[c] = newBudgets[i]
            idx = np.argwhere(np.isclose(self.budgets,newBudgets[i])).reshape(-1)
            finalBids[c] = self.optimalBidPerBudget[c,idx]
        """
        return [finalBudgets,finalBids]



    def normalizeBudgetArray(self,budgetArray):
        return budgetArray/self.maxTotDailyBudget

    def normalizeBidsArray(self,bidsArray):
        return bidsArray/self.maxBid


    def normalize(self,X):
        X[:,0] = X[:,0]/(self.maxTotDailyBudget)
        X[:,1] = X[:,1]/(self.maxBid)
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







    def plotGP(self,gpIndex,fixedBid = False,bid=0.1):
        if (fixedBid==False):
            budgetPoints = np.linspace(0,self.maxTotDailyBudget,1000)
            bidsPoints = np.linspace(0,self.maxBid,1000)
            bestBids = self.bestBidsPerBudgetsArray(budgetPoints,bidsPoints, gpIndex)

        else:
            budgetPoints = np.linspace(0,self.maxTotDailyBudget,1000)
            bestBids = np.ones(1000)*bid


        fig = plt.figure()
        observedInput = self.prevBudgets[:,gpIndex]
        observedOutput = self.prevClicks[:,gpIndex]

        x = np.array([bestBids,budgetPoints])
        x = np.atleast_2d(x).T
        budgetPointsNorm = self.normalizeBudgetArray(budgetPoints)
        bestBidsNorm = self.normalizeBidsArray(bestBids)
        xnorm = np.array([bestBidsNorm,budgetPointsNorm])
        xnorm = np.atleast_2d(x).T
        [means,sigmas] = self.gps[gpIndex].predict(x,return_std=True)
        means = self.denormalizeOutput(means,gpIndex)
        sigmas = self.denormalizeOutput(sigmas,gpIndex)

        plt.plot(observedInput, observedOutput, 'r.', markersize=10, label=u'Observations')

        plt.plot(budgetPoints, means, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([budgetPoints, budgetPoints[::-1]]),
                 np.concatenate([means - 1.9600 * sigmas,
                                 (means + 1.9600 * sigmas)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')

        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, np.max(self.prevClicks[:,gpIndex])*1.5)
        plt.legend(loc='upper left')






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

