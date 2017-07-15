# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:44:49 2017

@author: alessandro
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt
import time as time


class AgentMarcello:

    def __init__(self,budgetTot,deadline,ncampaigns,nIntervals,nBids,maxBudget=100.0,maxBid=2.0):
        self.budgetTot = budgetTot
        self.deadline = deadline
        self.ncampaigns = ncampaigns
        self.costs = np.array([])
        self.revenues = np.array([])
        self.t = 0
        self.gpsClicks = []
        self.gpsCosts = []
        self.prevBudgets = np.array([])
        self.prevBids = np.array([])
        self.prevClicks = np.array([])
        self.prevConversions = np.array([])
        self.prevCosts = np.array([])
        self.prevHours = np.array([])
        self.valuesPerClick = np.zeros(ncampaigns)
        self.maxTotDailyBudget = maxBudget
        self.maxBid = maxBid
        self.nBudgetIntervals = nIntervals
        self.nBids = nBids
        self.bidBudgetMatrix = np.zeros(shape=(self.ncampaigns,self.nBids,self.nBudgetIntervals))

        self.budgets = np.linspace(0, maxBudget,nIntervals)
        self.bids = np.linspace(0,maxBid,nBids)
        self.optimalBidPerBudget = np.zeros((ncampaigns,nIntervals))

        self.campaignsValues = np.zeros(shape=(self.ncampaigns, len(self.budgets)))
        self.clicksPerBid = np.zeros(shape=(self.ncampaigns, len(self.bids)))
        self.costsPerBid = np.zeros(shape=(self.ncampaigns, len(self.bids)))
        self.budgetPerBid =  np.zeros(shape=(self.ncampaigns, len(self.bids)))
        self.ymax=np.ones((ncampaigns))


    def updateValuesPerClick(self):
        for c in range(0,self.ncampaigns):
            value = np.sum(self.prevConversions[:,c])/np.sum(self.prevClicks[:,c])
            if(np.isnan(value) or np.isinf(value)):
                self.valuesPerClick[c] = 0
            else:
                self.valuesPerClick[c] = value




    def initGPs(self):
        for c in range(0,self.ncampaigns):
            #C(1.0, (1e-3, 1e3))
            #l= np.array([200,200])
            #kernel = C(1, (1e-3, 1e1))*RBF(l, ((100, 300),(100,300)))
            l1 = np.array([1.0])
            l2 = np.array([1.0])
            kernel1 = C(1.0, (1e-3, 1e3))*RBF(l1,(1e-3, 1e3))
            kernel2 = C(1.0, (1e-3, 1e3))*RBF(l2,(1e-3, 1e3))
            #l=1.0
            #kernel = C(1.0, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
            alpha1=200
            alpha2 = 200
            self.gpsClicks.append(GaussianProcessRegressor(kernel=kernel2, alpha=alpha1, n_restarts_optimizer=10,normalize_y=True))
            self.gpsCosts.append(GaussianProcessRegressor(kernel=kernel1, alpha=alpha2, n_restarts_optimizer=10,normalize_y=True))

    def dividePotentialClicks(self,numerator,denominator):
        div = numerator/denominator
        div[np.isnan(div)]=0
        div[np.isinf(div)]=0
        return div


    def updateClickGP(self,c):
        self.prevBids=np.atleast_2d(self.prevBids)
        self.prevClicks=np.atleast_2d(self.prevClicks)
        self.prevBudgets = np.atleast_2d(self.prevBudgets)
        bud =self.prevBudgets.T[c,:]
        x=self.prevBids.T[c,:]
        #x=np.atleast_2d(x).T

        xnorm=self.normalize(x)
        idxsNoZero = np.argwhere(bud!=0).reshape(-1)
        xnorm = xnorm[idxsNoZero]
        xnorm = np.atleast_2d(xnorm)
        potentialClicks = self.dividePotentialClicks(self.prevClicks* 24.0, self.prevHours)
        y = potentialClicks.T[c, :].ravel()

        # remove 0 in training
        y = y[idxsNoZero]
        if(len(y)>0):
            self.gpsClicks[c].fit(xnorm.T, y)


    def updateCostGP(self,c):
        self.prevBids=np.atleast_2d(self.prevBids)
        self.prevCosts=np.atleast_2d(self.prevCosts)
        self.prevBudgets = np.atleast_2d(self.prevBudgets)
        bud = self.prevBudgets.T[c, :]

        x=self.prevBids.T[c,:]
        #x=np.atleast_2d(x).T
        xnorm=self.normalize(x)

        idxsNoZero = np.argwhere(bud!=0).reshape(-1)
        xnorm = xnorm[idxsNoZero]
        xnorm = np.atleast_2d(xnorm)
        potentialCosts = self.dividePotentialClicks(self.prevCosts * 24.0, self.prevHours)
        y = potentialCosts.T[c, :].ravel()
        y = y[idxsNoZero]
        if(len(y)>0):
            self.gpsCosts[c].fit(xnorm.T, y)




    def updateGP(self,c):
        self.updateCostGP(c)
        self.updateClickGP(c)



    def updateMultiGP(self):
        for c in range(0,self.ncampaigns):
            self.updateGP(c)

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
            self.prevCosts = costs
            self.prevRevenues = revenues
        else:
            self.prevBudgets = np.append(self.prevBudgets,budgets, axis=0)
            self.prevBids = np.append(self.prevBids,bids, axis=0)
            self.prevClicks = np.append(self.prevClicks,clicks, axis=0)
            self.prevConversions = np.append(self.prevConversions,conversions, axis=0)
            self.prevHours = np.append(self.prevHours,hours, axis=0)
            self.prevCosts = np.append(self.prevCosts, costs, axis=0)
            self.prevRevenues = np.append(self.prevRevenues, revenues, axis=0)
        self.updateMultiGP()
        self.updateValuesPerClick()
        self.t +=1

    def valueForBudget(self,itemIdx,budget,values):
        idx = np.isclose(budget,self.budgets)
        if (len(idx)>0):
            return values[itemIdx,idx]
        idx = np.argwhere(budget>=self.budgets)
        return values[itemIdx,idx.max()]

    def firstRow(self,values):
        firstRow = np.zeros(len(self.budgets)).tolist()
        for i,b in enumerate(self.budgets):
            firstRow[i]=[[values[0,i]],[0],[b]]
        return firstRow



    def optimize(self,values):
        #start = time.time()
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
        #print "Time Optimization :",time.time() - start
        return [newBudgets,newCampaigns]


    def updateCostsPerBids(self):
        for c in range(0,self.ncampaigns):
            bidPoints = self.bids[:]
            x = np.array([bidPoints.T])
            x = np.atleast_2d(x).T
            x = self.normalize(x)

            [mean,sigma]=self.gpsCosts[c].predict(x,return_std=True)
            self.costsPerBid[c,:] = np.random.normal(mean,sigma)
            idxs = np.argwhere(self.bids==0).reshape(-1)
            for i in idxs:
                self.costsPerBid[c,i]=0.0
        return self.costsPerBid


    def updateClicksPerBids(self):

        for c in range(0,self.ncampaigns):
            bidPoints = self.bids[:]
            x = np.array([bidPoints.T])
            x = np.atleast_2d(x).T
            x = self.normalize(x)

            [mean,sigma]=self.gpsClicks[c].predict(x,return_std=True)
            self.clicksPerBid[c,:] = np.random.normal(mean,sigma)

            idxs = np.argwhere(self.bids==0).reshape(-1)
            for i in idxs:
                self.clicksPerBid[c,i]=0.0
        return self.clicksPerBid


    def divideFloat(self,numerator,denominator):
        res = numerator/denominator
        if np.isnan(res):
            return 0.0
        if np.isinf(res):
            return 0.0
        return res

    def generateBidBudgetMatrix(self):
        for c in range(0,self.ncampaigns):
            for i,bid in enumerate(self.bids):
                for j,bud in enumerate(self.budgets):
                    if(self.costsPerBid[c,i]<bud):
                        self.bidBudgetMatrix[c,i,j] = self.clicksPerBid[c,i]
                    else:
                        self.bidBudgetMatrix[c,i,j] = self.divideFloat(self.clicksPerBid[c,i] * bud,self.costsPerBid[c,i])


    def updateOptimalBidPerBudget(self):
        for c in range(0,self.ncampaigns):
            for b in range(0,self.nBudgetIntervals):
                idx = np.argwhere(self.bidBudgetMatrix[c,:,b]==self.bidBudgetMatrix[c,:,b].max()).reshape(-1)
                idx = np.random.choice(idx)
                self.optimalBidPerBudget[c,b] = self.bids[idx]


    def valuesForCampaigns(self,sampling=False):
        for c in range(0,self.ncampaigns):
            for b in range(0,self.nBudgetIntervals):
                self.campaignsValues[c,b] = self.bidBudgetMatrix[c,:,b].max() * self.valuesPerClick[c]

        return self.campaignsValues

        #self.campaignsValues[c,b] = self.campaignsValues *



    """
    def valuesForCampaigns(self,sampling=False ,bidSampling = True):

        for c in range(0,self.ncampaigns):
            x = np.array([self.bids.T])
            x = np.atleast_2d(x).T
            x = self.normalize(x)
            [meanClicksPerBid,sigmaClicksPerBid] = self.gpsClicks[c].predict(x,return_std=True)

            #self.clicksPerBid[c,:] = meanClicksPerBid
            self.clicksPerBid[c,:] = np.random.normal(meanClicksPerBid,sigmaClicksPerBid)
            # RICORDA FARE SAMPLING TODO AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
            [meanCostsPerBid, sigmaCostsPerBid] = self.gpsCosts[c].predict(x, return_std=True)

            self.costsPerBid[c,:] =  np.random.normal(meanCostsPerBid, sigmaCostsPerBid)

            for i,b in enumerate(self.budgets):
                self.campaignsValues[c,i] = self.valuesPerClick[c]*self.clicksForBudget(c,b)
                self.defineOptimalBidPerBudget(c,i)
        return self.campaignsValues
    """




    def chooseAction(self,sampling=False, fixedBid=False, fixedBudget=False, fixedBidValue=1.0, fixedBudgetValue=1000.0):
        if self.t<=4:
            return [ np.ones(self.ncampaigns)*self.maxTotDailyBudget/self.ncampaigns , np.random.choice(self.bids,self.ncampaigns)]
        self.updateCostsPerBids()
        self.updateClicksPerBids()
        self.generateBidBudgetMatrix()
        self.updateOptimalBidPerBudget()
        values = self.valuesForCampaigns()
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
       return X/(self.maxBid)

    def normalizeOutput(self,y,campaign):
        return y
        if(y.max()!=0):
            self.ymax[campaign] = y.max()
            y=y/y.max()
        return y


    def denormalizeOutput(self,y,campaign):
        return y
        return y*self.ymax[campaign]

    """
    def findBestBidPerBudget(self,budget,bidsArray,gpIndex):
        x = np.array([bidsArray.T, np.matlib.repmat(budget, 1, len(bidsArray)).reshape(-1)])
        x = np.atleast_2d(x).T
        x = self.normalize(x)
        valuesforBids = self.denormalizeOutput(self.predictPrior(gpIndex,x,False))
        idxs = np.argwhere(valuesforBids == valuesforBids.max()).reshape(-1)
        idx = np.random.choice(idxs)
        return bidsArray[idx]
    """

    def bestBidsPerBudgetsArray(self,budgetArray,bidsArray,gpIndex):
        bestBidsArray= np.zeros(len(budgetArray))
        for i,b in enumerate(budgetArray):
            bestBidsArray[i]=self.findBestBidPerBudget(b,bidsArray,gpIndex)
        return bestBidsArray


    def prior(self,x,y):
        return 0
        if(self.t<=10):
            return 0
        max_y= np.max(y)
        return max_y * x[:,1]

    def fitPrior(self,gpIndex,x,y):

        y_new = y - self.prior(x,y)
        self.gps[gpIndex].fit(x,y_new)

    def predictPrior(self,gpIndex,x,returnStd = False):
        if(returnStd ==False):
            y = self.gps[gpIndex].predict(x,return_std=returnStd)
            y_new = y + self.prior(x,self.prevClicks[:,gpIndex])
            return y_new
        else:

            [mean_y,sigma_y] = self.gps[gpIndex].predict(x,return_std=returnStd)
            if (len(self.prevClicks)==0):
                return mean_y,sigma_y
            y_new = mean_y + self.prior(x,self.prevClicks[:,gpIndex])
            """
            print "\n"
            #print "prediction without prior:",mean_y
            y_new = mean_y + self.prior(x,self.prevClicks[:,gpIndex])
            #print "prediction with prior",y_new
            #print "x ",x
            #print "prior ",self.prior(x,self.prevClicks[:,gpIndex])
            print "prediction without prior:",mean_y
            print "prediction with prior",y_new
            print "x ",x
            print "prior ",self.prior(x,self.prevClicks[:,gpIndex])
            """
            return y_new,sigma_y
