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
from Agent import *
from Utils import *
import copy

class AgentFactored:

    def __init__(self, budgetTot, deadline, nCampaigns, nBudget, nBids, maxBudget=100.0, maxBid=2.0, method='Sampling'):
        self.budgetTot = budgetTot
        self.deadline = deadline
        self.nCampaigns = nCampaigns
        self.costs = np.array([])
        self.revenues = np.array([])
        self.t = 0

        # Historical data: GP prediction
        self.gpsClicks = []
        self.gpsCosts = []

        # Historical data: realizations
        self.prevBudgets = np.array([])
        self.prevBids = np.array([])
        self.prevClicks = np.array([])
        self.prevConversions = np.array([])
        self.prevCosts = np.array([])
        self.prevHours = np.array([])

        # Estimated value per click (initially small)
        self.valuesPerClick = np.zeros(nCampaigns)

        self.maxTotDailyBudget = maxBudget
        self.maxBid = maxBid
        self.nBudget = nBudget
        self.nBids = nBids
        self.bidBudgetMatrix = np.zeros(shape=(self.nCampaigns, self.nBids, self.nBudget))

        self.budgets = np.linspace(0, maxBudget, nBudget)
        self.bids = np.linspace(0, maxBid, nBids)
        self.optimalBidPerBudget = np.zeros((nCampaigns, nBudget))

        self.campaignsValues = np.zeros(shape=(self.nCampaigns, nBudget))
        self.clicksPerBid = np.zeros(shape=(self.nCampaigns, nBids))
        self.costsPerBid = np.zeros(shape=(self.nCampaigns, nBids))
        self.budgetPerBid =  np.zeros(shape=(self.nCampaigns, nBids))
        self.ymax = np.ones((nCampaigns))

        self.method = method



    def updateValuesPerClick(self):
        """
        Estimation of the expected value per click for each subcampaign
        :return:
        """
        for c in range(0, self.nCampaigns):
            value = np.sum(self.prevConversions[:, c]) / np.sum(self.prevClicks[:, c])
            if np.isnan(value) or np.isinf(value):
                self.valuesPerClick[c] = 0
            else:
                self.valuesPerClick[c] = value

    def initGPs(self, ):
        for c in range(0, self.nCampaigns):

            # Number of clicks
            lClick = np.array([1.0])
            kernelClick = C(1.0, (1e-3, 1e3)) * RBF(lClick, (1e-3, 1e3))
            alphaClick = 200
            self.gpsClicks.append(GaussianProcessRegressor(kernel=kernelClick, alpha=alphaClick, n_restarts_optimizer=10, normalize_y=True))

            # Costs
            lCost = np.array([1.0])
            kernelCost = C(1.0, (1e-3, 1e3)) * RBF(lCost,(1e-3, 1e3))
            alphaCost = 200
            self.gpsCosts.append(GaussianProcessRegressor(kernel=kernelCost, alpha=alphaCost, n_restarts_optimizer=10, normalize_y=True))

    def updateClickGP(self, c):
        """
        Updates the number of click GP with incoming data
        :param c:
        :return:
        """
        # Format data for the GP (2D vector)
        self.prevBids = np.atleast_2d(self.prevBids)
        self.prevClicks = np.atleast_2d(self.prevClicks)
        self.prevBudgets = np.atleast_2d(self.prevBudgets)

        bud = self.prevBudgets.T[c, :]
        bid = self.prevBids.T[c, :]

        # Click for unlimited budget
        bidNorm = self.normalize(bid)
        idxsNoZero = np.argwhere(bud != 0).reshape(-1)
        bidNorm = bidNorm[idxsNoZero]
        bidNorm = np.atleast_2d(bidNorm)
        potentialClicks = dividePotentialClicks(self.prevClicks * 24.0, self.prevHours)
        potentialClicks = potentialClicks.T[c, :].ravel()

        # remove 0 in training
        potentialClicks = potentialClicks[idxsNoZero]
        if len(potentialClicks) > 0:
            self.gpsClicks[c].fit(bidNorm.T, potentialClicks)



    def setGPparameters(self, sigmaClick, sigmaCost, lClick, lCost, alphaClick, alphaCost):
        """
        Set customized parameters to the GP
        :param sigmaClick:
        :param sigmaCost:
        :param lClick:
        :param lCost:
        :param alphaClick:
        :param alphaCost:
        :return:
        """
        for c in range(0, self.nCampaigns):
            kernelClick = C(sigmaClick[c], (1e-3, 1e3)) * RBF(lClick[c], (1e-3, 1e3))
            kernelCost = C(sigmaCost[c], (1e-3, 1e3)) * RBF(lCost[c], (1e-3, 1e3))
            self.gpsClicks[c] = GaussianProcessRegressor(kernel=kernelClick, alpha=alphaClick, optimizer=None,
                                                         normalize_y=True)
            self.gpsCosts[c] = GaussianProcessRegressor(kernel=kernelCost, alpha=alphaCost, optimizer=None,
                                                        normalize_y=True)

    def setGPKernel(self, c, kernelClick, kernelCost, alpha=200):
        """
        Set customized kernel to the c-th GP
        :param c: index of the GP to set
        :param kernelClick:
        :param kernelCost:
        :param alpha:
        :return:
        """
        self.gpsClicks[c] = GaussianProcessRegressor(kernel=kernelClick, alpha=alpha, optimizer=None, normalize_y=True)
        self.gpsCosts[c] = GaussianProcessRegressor(kernel=kernelCost, alpha=alpha, optimizer=None, normalize_y=True)

    def updateCostGP(self, c):
        """
        prevBids = copy.copy(self.prevBids)
        prevCosts = copy.copy(self.prevCosts)
        prevBudgets = copy.copy(self.prevBudgets)
        prevHours = copy.copy(self.prevHours)

        prevBids = np.append(prevBids,0.0)
        prevCosts = np.append(prevCosts, 0.0)
        prevBudgets = np.append(prevBudgets,10)
        prevHours = np.append(prevHours,10)

        prevBids = np.atleast_2d(prevBids)
        prevCosts = np.atleast_2d(prevCosts)
        prevBudgets = np.atleast_2d(prevBudgets)
        """

        self.prevBids = np.atleast_2d(self.prevBids)
        self.prevCosts = np.atleast_2d(self.prevCosts)
        self.prevBudgets = np.atleast_2d(self.prevBudgets)


        bud = self.prevBudgets.T[c, :]
        bid = self.prevBids.T[c,:]



        bidNorm = self.normalize(bid)

        idxsNoZero = np.argwhere(bud != 0).reshape(-1)
        bidNorm = bidNorm[idxsNoZero]

        bidNorm = np.append(bidNorm,0.0)

        bidNorm = np.atleast_2d(bidNorm)

        potentialCosts = dividePotentialClicks(self.prevCosts * 24.0, self.prevHours)
        potentialCosts = potentialCosts.T[c, :].ravel()
        potentialCosts = potentialCosts[idxsNoZero]


        potentialCosts = np.append(potentialCosts,0.0)

        if len(potentialCosts) > 0:
            self.gpsCosts[c].fit(bidNorm.T, potentialCosts)

    """
    def updateGPCostZero(self):
        for c in range(0,self.nCampaigns):
            bid = np.array([0.0])
            cost = np.array([0.0]).ravel()
            self.gpsCosts[c].fit(bid.T,cost)
    """



    def updateGP(self, c):
        self.updateCostGP(c)
        self.updateClickGP(c)

    def updateMultiGP(self):
        for c in range(0, self.nCampaigns):
            self.updateGP(c)

    def updateState(self, bids, budgets, clicks, conversions, costs, revenues, hours):
        bids = np.atleast_2d(bids)
        budgets = np.atleast_2d(budgets)
        clicks = np.atleast_2d(clicks)
        conversions = np.atleast_2d(conversions)
        hours = np.atleast_2d(hours)
        costs = np.atleast_2d(costs)
        revenues = np.atleast_2d(revenues)

        if self.t == 0:
            self.prevBudgets = budgets
            self.prevBids = bids
            self.prevClicks = clicks
            self.prevConversions = conversions
            self.prevHours = hours
            self.prevCosts = costs
            self.prevRevenues = revenues
        else:
            self.prevBudgets = np.append(self.prevBudgets, budgets, axis=0)
            self.prevBids = np.append(self.prevBids, bids, axis=0)
            self.prevClicks = np.append(self.prevClicks, clicks, axis=0)
            self.prevConversions = np.append(self.prevConversions, conversions, axis=0)
            self.prevHours = np.append(self.prevHours, hours, axis=0)
            self.prevCosts = np.append(self.prevCosts, costs, axis=0)
            self.prevRevenues = np.append(self.prevRevenues, revenues, axis=0)

        self.updateMultiGP()
        self.updateValuesPerClick()
        self.t += 1

    def valueForBudget(self, itemIdx, budget, values):
        """
        Find nearest value to the specified budget
        :param itemIdx:
        :param budget:
        :param values:
        :return:
        """
        idx = np.isclose(budget, self.budgets)
        if len(idx) > 0:
            return values[itemIdx, idx]
        else:
            idx = np.argwhere(budget >= self.budgets)
            return values[itemIdx, idx.max()]

    def firstRow(self, values):
        res = np.zeros(len(self.budgets)).tolist()
        for budIdx, bud in enumerate(self.budgets):
            res[budIdx] = [[values[0, budIdx]], [0], [bud]] # (cumulate value, indexes, list of budgets)
        return res

    def optimize(self, values):
        # Inded in the results where the data are
        valIdx = 0
        itIdx = 1
        bIdx = 2

        h = np.zeros(shape=(self.nCampaigns, len(self.budgets)))
        h = h.tolist()
        h[0] = self.firstRow(values)
        for i in range(1, self.nCampaigns):
            for j, b in enumerate(self.budgets):
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
                        h[i][j] = [newValues, newItems, newSelBudgets]
                        maxVal = np.sum(newValues)
        newBudgets = h[-1][-1][2]
        newCampaigns = h[-1][-1][1]
        return [newBudgets, newCampaigns]

    def updateCostsPerBids(self):
        for c in range(0, self.nCampaigns):
            bidPoints = self.bids[:]
            x = np.array([bidPoints.T])
            x = np.atleast_2d(x).T
            x = self.normalize(x)
            [mean, sigma] = self.gpsCosts[c].predict(x,return_std=True)
            self.costsPerBid[c, :] = self.sampleCost(mean,sigma)

            # Reasonable assumption that if bid is zero also costs are null
            idxs = np.argwhere(self.bids == 0).reshape(-1)
            for i in idxs:
                self.costsPerBid[c, i] = 0.0
        return self.costsPerBid


    def updateClicksPerBids(self):
        for c in range(0, self.nCampaigns):
            bidPoints = self.bids[:]
            x = np.array([bidPoints.T])
            x = np.atleast_2d(x).T
            x = self.normalize(x)

            [mean, sigma] = self.gpsClicks[c].predict(x,return_std=True)
            self.clicksPerBid[c,:] = self.sampleClick(mean,sigma)

            idxs = np.argwhere(self.bids == 0).reshape(-1)
            for i in idxs:
                self.clicksPerBid[c, i] = 0.0
        return self.clicksPerBid


    def sampleClick(self, mean, sigma):
        if self.method == "Sampling":
            return np.random.normal(mean,sigma)
        if self.method == "Mean":
            return mean
        if self.method =="UCB":
            return mean + 3.0*sigma

    def sampleCost(self,mean,sigma):
        if self.method == "Sampling":
            return np.random.normal(mean,sigma)
        if self.method == "Mean":
            return mean
        if self.method =="UCB":
            return np.maximum(mean - 3.0*sigma,0.0)

    def generateBidBudgetMatrix(self):
        """
        Matrix of the number of clicks for each (bid/budget) pair
        :return:
        """
        for c in range(0, self.nCampaigns):
            for i, bid in enumerate(self.bids):
                for j, bud in enumerate(self.budgets):
                    if(self.costsPerBid[c, i] < bud):
                        self.bidBudgetMatrix[c, i, j] = self.clicksPerBid[c, i]
                    else:
                        self.bidBudgetMatrix[c, i, j] = divideFloat(self.clicksPerBid[c, i] * bud,self.costsPerBid[c, i])

# [NOTA] due funzioni successive da fondere
    def updateOptimalBidPerBudget(self):
        for c in range(0, self.nCampaigns):
            for b in range(0, self.nBudget):
                idx = np.argwhere(self.bidBudgetMatrix[c, :, b] == self.bidBudgetMatrix[c, :, b].max()).reshape(-1)
                idx = np.random.choice(idx)
                self.optimalBidPerBudget[c, b] = self.bids[idx]

    def valuesForCampaigns(self):
        for c in range(0, self.nCampaigns):
            for b in range(0, self.nBudget):
                self.campaignsValues[c, b] = self.bidBudgetMatrix[c, :, b].max() * self.valuesPerClick[c]

        return self.campaignsValues

    def chooseAction(self, initialExploration=4):
        if self.t <= initialExploration:
            # Equally shared budget and random bid for each subcampaign
            equalBud = self.maxTotDailyBudget / self.nCampaigns
            bud = self.budgets[np.max(np.argwhere(self.budgets <= equalBud))] #prendo il primo budget più piccolo della ripartiizone equa
            buds = np.ones(self.nCampaigns) * bud
            buds[-1] = (self.maxTotDailyBudget - (self.nCampaigns - 1) * bud)
            return [buds, np.random.choice(self.bids, self.nCampaigns)]
        else:
            self.updateCostsPerBids()
            self.updateClicksPerBids()
            self.generateBidBudgetMatrix()
            self.updateOptimalBidPerBudget()

            values = self.valuesForCampaigns()
            [selectedBudgets, selectedCampaigns] = self.optimize(values)
            finalBudgets = np.zeros(self.nCampaigns)
            finalBids = np.zeros(self.nCampaigns)
            for i, c in enumerate(selectedCampaigns):
                finalBudgets[c] = selectedBudgets[i]
                idx = np.argwhere(np.isclose(self.budgets, selectedBudgets[i])).reshape(-1)
                finalBids[c] = self.optimalBidPerBudget[c, idx]
            return [finalBudgets, finalBids]

    def normalize(self, X):
       return X / self.maxBid
