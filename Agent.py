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
    
    def __init__(self,budgetTot,deadline,ncampaigns):
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
    
    def updateValuesPerClick(self):
        for c in range(0,self.ncampaigns):
            self.valuesPerClick[c] = np.sum(self.prevConversions[:,c])/np.sum(self.prevClicks[:,c])
        
    def initGPs(self):
        for c in range(0,self.ncampaigns):
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            alpha=0.0
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
        print np.shape(bids)
        self.prevBids=np.atleast_2d(self.prevBids)
        self.prevBudgets=np.atleast_2d(self.prevBudgets)
        self.prevClicks=np.atleast_2d(self.prevClicks)
        self.prevConversions=np.atleast_2d(self.prevConversions)
        self.prevHours=np.atleast_2d(self.prevHours)
        
        self.prevBudgets = np.append(self.prevBudgets,budgets, axis=0)
        self.prevBids = np.append(self.prevBids,bids, axis=0)
        self.prevClicks = np.append(self.prevClicks,clicks, axis=0)
        self.prevConversions = np.append(self.prevConversions,conversions, axis=0)
        self.prevHours = np.append(self.prevHours,hours, axis=0)
        self.updateMultiGP()
        self.costs += costs
        self.revenues += revenues
        self.t +=1
        
        
    def chooseAction(self):
        return  [np.ones(self.ncampaigns), 100*np.ones(self.ncampaigns)]
        
    