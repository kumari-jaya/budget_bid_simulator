#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from Agent import *
from Core import *
from matplotlib import pyplot as plt
from Plotter import *

convparams=np.array([0.4,100,200])
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])


a1= Auction(nbidders=5 , nslots=5, mu=0.61 , sigma=0.2, lambdas=lambdas)
a2= Auction(nbidders=6 , nslots=5, mu=0.67 , sigma=0.4, lambdas=lambdas)
a3= Auction(nbidders=8 , nslots=5, mu=0.47 , sigma=0.25, lambdas=lambdas)
a4= Auction(nbidders=5 , nslots=5, mu=0.57 , sigma=0.39, lambdas=lambdas)


ncampaigns=5
c1 = Campaign(a1, nusers=1000.0 , probClick=0.5 ,convParams= convparams)
c2 = Campaign(a2, nusers=1500.0 , probClick=0.6 ,convParams= convparams)
c3 = Campaign(a3, nusers=1500.0 , probClick=0.6 ,convParams= convparams)
c4 = Campaign(a2, nusers=1000.0 , probClick=0.5 ,convParams= convparams)
c5 = Campaign(a4, nusers=1250.0 , probClick=0.4 ,convParams= convparams)


env = Environment([c1,c2,c3,c4,c5])
nBids=5
nIntervals=10
deadline = 250
maxBudget = 100
agent = Agent(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
agent.initGPs()
plotter = Plotter(agent=agent,env=env)

# mi creo una lista con tutte le matrici dell'oracolo di ogni campagna
listMatrices = list()
for i in range(0,ncampaigns):
    matrix = plotter.oracleMatrix(indexCamp=i,nsimul=10)
    listMatrices.append(matrix)
    if i==0:
        optMatrix = np.array([matrix.max(axis=1)])
    else:
        maxrow = np.array([matrix.max(axis=1)])
        optMatrix = np.concatenate((optMatrix,maxrow))


[newBudgets,newCampaigns] = agent.optimize(optMatrix)

# ora ricerco nelle matrici originali il numero di click nell'allocazione ottima
optValue = 0
for i in range(0,ncampaigns):
    print i
    index = np.argwhere(np.isclose(agent.budgets,newBudgets[i]))
    tempValue = listMatrices[i][index,:].max()
    optValue += tempValue
optValue = optValue * convparams[0]  #converto i click in conversioni
## questo è il valore dell'oracolo per il plot ora devo simulare i valori del thompson!
nexperiments = 3
# mi salvo le tre realizzazioni degli esperimenti e poi alla fine le medio!
matrixValues = np.zeros((nexperiments,deadline))
matrixEst = np.zeros((nexperiments,deadline))
for k in range(0,nexperiments):
    print "Experiment: ",k+1
    agent = Agent(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
    agent.initGPs()
    core = Core(agent, env, deadline)
    chosenValues = np.zeros((deadline))
    estValues = np.zeros((deadline))
    for t in range(0,deadline):
        print "Day: ",t+1
        core.step()
        lastBudgets = agent.prevBudgets[-1,:]
        lastBids = agent.prevBids[-1,:]
        for i in range(0,ncampaigns):
            indBud = np.argwhere(np.isclose(agent.budgets, lastBudgets[i]))
            indBid = np.argwhere(np.isclose(agent.bids, lastBids[i]))
            # il valore dei click medio relativi ai bid e budget scelti li prendo dalle matrici dell'oracolo!
            chosenValues[t] += listMatrices[i][indBud,indBid] *convparams[0]
            estValues[t] += agent.campaignsValues[i,indBud]
    matrixValues[k,:] = chosenValues
    matrixEst [k,:] = estValues

finalValues = matrixValues.mean(axis=0)
finalEst = matrixEst.mean(axis=0)
plotter.performancePlot(optValue,finalValues,finalEst)
plotter.regretPlot(optValue, finalValues)
