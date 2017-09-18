# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:44:49 2017

@author: alessandro
"""

import numpy as np
from matplotlib import pyplot as plt
import time as time
from Agent import *
from Utils import *
import copy

class AgentZoom:

    def __init__(self, budgetTot, deadline, nCampaigns, nBudget, maxBudget=100.0, maxBid=1.0):
        self.budgetTot = budgetTot
        self.deadline = deadline
        self.nCampaigns = nCampaigns
        self.costs = np.array([])
        self.revenues = np.array([])
        self.t = 0


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
        self.maxBudget = maxBudget
        self.nBudget = nBudget
        #self.bidBudgetMatrix = np.zeros(shape=(self.nCampaigns, self.nBids, self.nBudget))

        self.budgets = np.linspace(0, maxBudget, nBudget)
        self.normbudgets = self.budgets/maxBudget
        self.activebids = list()
        self.ntimeplayed = list()
        self.meanClicks = list()
        self.meanCosts = list()
        self.confRadius = list()
        self.optimalBidPerBudget = np.zeros((nCampaigns, nBudget))
        self.maxClicks = np.zeros(shape=(self.nCampaigns))
        self.maxCosts = np.zeros(shape=(self.nCampaigns))

        self.campaignsValues = np.zeros(shape=(self.nCampaigns, nBudget))


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

    def updateEstimates(self,bids,clicks,costs,budgets):
        bids = bids.reshape(-1)
        clicks = clicks.reshape(-1)
        costs = costs.reshape(-1)
        budgets = budgets.reshape(-1)
        for c in range(0, self.nCampaigns):
            if budgets.reshape(-1)[c] != 0:
                index = np.argwhere(self.activebids[c] == bids[c])
                self.ntimeplayed[c][index] += 1
                meanupdateclicks = ((self.meanClicks[c][index] * (self.ntimeplayed[c][index]-1)) + clicks[c])/self.ntimeplayed[c][index]
                meanupdatecosts = ((self.meanCosts[c][index] * (self.ntimeplayed[c][index]-1)) + costs[c])/self.ntimeplayed[c][index]
                self.meanClicks[c][index] = meanupdateclicks
                self.meanCosts[c][index] = meanupdatecosts
                self.confRadius[c][index] = np.sqrt(2*np.log(self.deadline))/(self.ntimeplayed[c][index]+1)
                if clicks[c] > self.maxClicks[c]:
                    self.maxClicks[c] = clicks[c]
                if costs[c] > self.maxCosts[c]:
                    self.maxCosts[c] = costs[c]



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

        self.updateEstimates(bids,clicks,costs,budgets)
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
        idx = np.isclose(budget, self.normbudgets)
        if len(idx) > 0:
            return values[itemIdx, idx]
        else:
            idx = np.argwhere(budget >= self.normbudgets)
            return values[itemIdx, idx.max()]

    def firstRow(self, values):
        res = np.zeros(len(self.budgets)).tolist()
        for budIdx, bud in enumerate(self.normbudgets):
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
            for j, b in enumerate(self.normbudgets):
                h[i][j] = h[i-1][j][:]
                maxVal = 0
                for bi in range(0,j+1):
                    if ((np.sum(h[i-1][bi][valIdx]) + self.valueForBudget(i,b - self.normbudgets[bi],values)) >maxVal):
                        val = h[i-1][bi][valIdx][:]
                        val.append(self.valueForBudget(i,b - self.normbudgets[bi],values))
                        newValues = val[:]
                        items = h[i-1][bi][itIdx][:]
                        items.append(i)
                        newItems = items[:]
                        selBudgets = h[i-1][bi][bIdx][:]
                        selBudgets.append(b - self.normbudgets[bi])
                        newSelBudgets = selBudgets[:]
                        h[i][j] = [newValues, newItems, newSelBudgets]
                        maxVal = np.sum(newValues)
        newBudgets = h[-1][-1][2]
        newCampaigns = h[-1][-1][1]
        return [newBudgets, newCampaigns]

    def normalizeClicks(self, c):
       return self.meanClicks[c]/self.maxClicks[c]

    def normalizeCosts(self, c):
       return self.meanCosts[c]/self.maxCosts[c]

    def updateCostsPerBids(self):
        costsPerBid = list()
        for c in range(0, self.nCampaigns):
            costnorm = self.normalizeCosts(c)
            upperBound = costnorm + 2*self.confRadius[c]
            costsPerBid.append(upperBound)
        return costsPerBid


    def updateClicksPerBids(self):
        clicksPerBid = list()
        for c in range(0, self.nCampaigns):
            clicknorm = self.normalizeClicks(c)
            upperBound = clicknorm + 2*self.confRadius[c]
            clicksPerBid.append(upperBound)
        return clicksPerBid


    def generateBidBudgetMatrix(self,clicksPerBid,costsPerBid):
        """
        Matrix of the number of clicks for each (bid/budget) pair
        :return:
        """
        bidBudgetMatrix = list()
        for c in range(0, self.nCampaigns):
            bidBudgetMatrix.append(np.zeros(shape=(len(self.activebids[c]),len(self.budgets))))
            for i, bid in enumerate(self.activebids[c]):
                for j, bud in enumerate(self.normbudgets):
                    if(costsPerBid[c][i] < bud):
                        bidBudgetMatrix[c][i, j] = clicksPerBid[c][i]
                    else:
                        bidBudgetMatrix[c][i, j] = divideFloat(clicksPerBid[c][i] * bud, costsPerBid[c][i])
        return bidBudgetMatrix

# [NOTA] due funzioni successive da fondere
    def updateOptimalBidPerBudget(self,bidBudgetMatrix):
        for c in range(0, self.nCampaigns):
            for b in range(0, self.nBudget):
                idx = np.argwhere(bidBudgetMatrix[c][:, b] == bidBudgetMatrix[c][:, b].max()).reshape(-1)
                idx = np.random.choice(idx)
                self.optimalBidPerBudget[c, b] = self.activebids[c][idx]

    def valuesForCampaigns(self,bidBudgetMatrix):
        for c in range(0, self.nCampaigns):
            for b in range(0, self.nBudget):
                self.campaignsValues[c, b] = bidBudgetMatrix[c][:, b].max() * self.valuesPerClick[c]

        return self.campaignsValues

    def chooseAction(self):
        if self.t == 0:
            # Equally shared budget and random bid for each subcampaign
            equalBud = self.maxTotDailyBudget / self.nCampaigns
            bud = self.budgets[np.max(np.argwhere(self.budgets <= equalBud))] #prendo il primo budget più piccolo della ripartiizone equa
            buds = np.ones(self.nCampaigns) * bud
            buds[-1] = (self.maxTotDailyBudget - (self.nCampaigns - 1) * bud)
            for i in range(self.nCampaigns):
                newArm = np.random.uniform(0,self.maxBid)
                self.activebids.append(np.array([newArm]))
                self.ntimeplayed.append(np.array([0]))
                self.meanClicks.append(np.array([0]))
                self.meanCosts.append(np.array([0]))
                self.confRadius.append(np.array([0]))
            return [buds, np.array(self.activebids).reshape(-1)]
        else:
            self.createNewArms()
            costsPerBid = self.updateCostsPerBids()
            clicksPerBid = self.updateClicksPerBids()
            bidBudgetMatrix = self.generateBidBudgetMatrix(clicksPerBid,costsPerBid)
            self.updateOptimalBidPerBudget(bidBudgetMatrix)

            values = self.valuesForCampaigns(bidBudgetMatrix)
            [selectedBudgets, selectedCampaigns] = self.optimize(values)
            finalBudgets = np.zeros(self.nCampaigns)
            finalBids = np.zeros(self.nCampaigns)
            for i, c in enumerate(selectedCampaigns):
                idx = np.argwhere(np.isclose(self.normbudgets, selectedBudgets[i])).reshape(-1)
                finalBudgets[c] = self.budgets[idx]
                finalBids[c] = self.optimalBidPerBudget[c, idx]
            return [finalBudgets, finalBids]


    def createNewArms(self):
        for c in range(self.nCampaigns):
            orderedStrategies = np.sort(self.activebids[c])
            indexes = np.argsort(self.activebids[c])
            # controllo che il primo intorno copra fino allo zero
            if orderedStrategies[0]-self.confRadius[c][indexes[0]]>0:
                diff = orderedStrategies[0]-self.confRadius[c][indexes[0]]
                self.activebids[c] = np.append(self.activebids[c],np.random.uniform(0,diff))
                self.confRadius[c] = np.append(self.confRadius[c],np.sqrt(2*np.log(self.deadline)))
                self.ntimeplayed[c] = np.append(self.ntimeplayed[c],0)
                self.meanClicks[c] = np.append(self.meanClicks[c],0)
                self.meanCosts[c] = np.append(self.meanCosts[c],0)

            #controllo tutti i buchi in centro tra due arms consecutivi
            for i in range(1,len(orderedStrategies)):
                if orderedStrategies[i]-self.confRadius[c][indexes[i]] > orderedStrategies[i-1]+self.confRadius[c][indexes[i-1]]:
                    diffup = orderedStrategies[i]-self.confRadius[c][indexes[i]]
                    difflow = orderedStrategies[i-1]+self.confRadius[c][indexes[i-1]]
                    self.activebids[c] = np.append(self.activebids[c],np.random.uniform(difflow,diffup))
                    self.confRadius[c] = np.append(self.confRadius[c],np.sqrt(2*np.log(self.deadline)))
                    self.ntimeplayed[c] = np.append(self.ntimeplayed[c],0)
                    self.meanClicks[c] = np.append(self.meanClicks[c],0)
                    self.meanCosts[c] = np.append(self.meanCosts[c],0)

            # controllo che l'ultimo copra fino ad 1
            if orderedStrategies[-1]+self.confRadius[c][indexes[-1]] < self.maxBid:
                diff = orderedStrategies[-1]+self.confRadius[c][indexes[-1]]
                self.activebids[c] = np.append(self.activebids[c],np.random.uniform(diff,self.maxBid))
                self.confRadius[c] = np.append(self.confRadius[c],np.sqrt(2*np.log(self.deadline)))
                self.ntimeplayed[c] = np.append(self.ntimeplayed[c],0)
                self.meanClicks[c] = np.append(self.meanClicks[c],0)
                self.meanCosts[c] = np.append(self.meanCosts[c],0)
        return
