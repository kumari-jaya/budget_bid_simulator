# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
import time as time
from Agent import *
from Utils import *
import copy

class AgentCUCB:

    def __init__(self, budgetTot, deadline, nCampaigns, nBudget, nBids, maxBudget=100.0, maxBid=1.0):
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
        self.nBids = nBids
        self.bidBudgetMatrix = np.zeros(shape=(self.nCampaigns, self.nBids, self.nBudget))

        self.budgets = np.linspace(0, maxBudget, nBudget)
        self.bids = np.linspace(0, maxBid, nBids)

        self.ntimeplayed = list()
        self.meanClicks = list()
        self.meanCosts = list()
        self.optimalBidPerBudget = np.zeros((nCampaigns, nBudget))

        self.campaignsValues = np.zeros(shape=(self.nCampaigns, nBudget))
        self.clicksPerBid = np.zeros(shape=(self.nCampaigns, nBids))
        self.costsPerBid = np.zeros(shape=(self.nCampaigns, nBids))
        self.budgetPerBid =  np.zeros(shape=(self.nCampaigns, nBids))


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
                index = np.argwhere(self.bids[c] == bids[c])
                self.ntimeplayed[c][index] += 1
                meanupdateclicks = ((self.meanClicks[c][index] * (self.ntimeplayed[c][index]-1)) + clicks[c])/self.ntimeplayed[c][index]
                meanupdatecosts = ((self.meanCosts[c][index] * (self.ntimeplayed[c][index]-1)) + costs[c])/self.ntimeplayed[c][index]
                self.meanClicks[c][index] = meanupdateclicks
                self.meanCosts[c][index] = meanupdatecosts


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


    def updateCostsPerBids(self):
        for c in range(0, self.nCampaigns):
            upperbound = np.sqrt(3*np.log(self.t+1)/(2*self.ntimeplayed[c]))
            self.costsPerBid[c, :] = np.max(self.meanCosts[c] - upperbound, 0)

            # Reasonable assumption that if bid is zero also costs are null
            idxs = np.argwhere(self.bids == 0).reshape(-1)
            for i in idxs:
                self.costsPerBid[c, i] = 0.0
        return self.costsPerBid


    def updateClicksPerBids(self):
        for c in range(0, self.nCampaigns):
            upperbound = np.sqrt(3*np.log(self.t+1)/(2*self.ntimeplayed[c]))
            self.clicksPerBid[c,:] = self.meanClicks[c] + upperbound

            idxs = np.argwhere(self.bids == 0).reshape(-1)
            for i in idxs:
                self.clicksPerBid[c, i] = 0.0
        return self.clicksPerBid

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
            if self.t == 0:
                self.initlists()
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

    def initlists(self):
        for c in range(self.nCampaigns):
            zero = np.zeros(self.nBids)
            self.ntimeplayed.append(zero)
            self.meanClicks.append(zero)
            self.meanCosts.append(zero)
