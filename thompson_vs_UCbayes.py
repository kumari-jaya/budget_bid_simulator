#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from Agent import *
from AgentPrior import *
from AgentUCB import *
from AgentMarcello import *
from Core import *
from matplotlib import pyplot as plt
from Plotter import *
from joblib import Parallel, delayed

def experiment(k):
    np.random.seed()
    print "Esperimento: ", k
    agentThomp = AgentPrior(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
    agentUCB = AgentUCB(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
    agentThomp.initGPs()
    agentUCB.initGPs()
    coreThomp = Core(agentThomp, env, deadline)
    coreUCB = Core(agentUCB, env, deadline)
    chosenValuesThomp = np.zeros((deadline))
    chosenValuesUCB = np.zeros((deadline))
    for t in range(0,deadline):
        coreThomp.step()
        coreUCB.step()
        lastBudgetsT = agentThomp.prevBudgets[-1,:]
        lastBidsT = agentThomp.prevBids[-1,:]
        lastBudgetsUC = agentUCB.prevBudgets[-1,:]
        lastBidsUC = agentUCB.prevBids[-1,:]
        for i in range(0,ncampaigns):
            indBudT = np.argwhere(np.isclose(agentThomp.budgets, lastBudgetsT[i]))
            indBidT = np.argwhere(np.isclose(agentThomp.bids, lastBidsT[i]))
            chosenValuesThomp[t] += listMatrices[i][indBudT,indBidT] *convparams[0]
            indBudUC = np.argwhere(np.isclose(agentUCB.budgets, lastBudgetsUC[i]))
            indBidUC = np.argwhere(np.isclose(agentUCB.bids, lastBidsUC[i]))
            chosenValuesUCB[t] += listMatrices[i][indBudUC,indBidUC] *convparams[0]
    return chosenValuesThomp,chosenValuesUCB



convparams=np.array([0.4,100,200])
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])


a1= Auction(nbidders=5 , nslots=5, mu=0.59 , sigma=0.2, lambdas=lambdas)
a2= Auction(nbidders=6 , nslots=5, mu=0.67 , sigma=0.4, lambdas=lambdas)
a3= Auction(nbidders=6 , nslots=5, mu=0.47 , sigma=0.25, lambdas=lambdas)
a4= Auction(nbidders=5 , nslots=5, mu=0.57 , sigma=0.39, lambdas=lambdas)


ncampaigns=3
c1 = Campaign(a1, nusers=1000.0 , probClick=0.5 ,convParams= convparams)
c2 = Campaign(a2, nusers=1500.0 , probClick=0.6 ,convParams= convparams)
c3 = Campaign(a3, nusers=1500.0 , probClick=0.6 ,convParams= convparams)
c4 = Campaign(a2, nusers=1000.0 , probClick=0.5 ,convParams= convparams)
c5 = Campaign(a4, nusers=1250.0 , probClick=0.4 ,convParams= convparams)


env = Environment([c1,c2,c3])
nBids=10
nIntervals=10
deadline = 2
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
matrixValuesThomp = np.zeros((nexperiments,deadline))
matrixValuesUCB = np.zeros((nexperiments,deadline))

out = Parallel(n_jobs=3)(
        delayed(experiment)(k) for k in xrange(nexperiments))

for i in range(nexperiments):
    matrixValuesThomp[i,:] = out[i][0]
    matrixValuesUCB[i,:] = out[i][1]

print "opt value:", optValue
#np.save("/home/alessandro/Dropbox/thesis_agos/plot/dati_plot_alessandro/valore_ottimo_3c",optValue)
#np.save("/home/alessandro/Dropbox/thesis_agos/plot/dati_plot_alessandro/matrice_thompson_3c",matrixValuesThomp)
#np.save("/home/alessandro/Dropbox/thesis_agos/plot/dati_plot_alessandro/matrice_UCB_3c",matrixValuesUCB)
finalValuesThomp = matrixValuesThomp.mean(axis=0)
finalValuesUCB = matrixValuesUCB.mean(axis=0)
plotter.performancePlotComparison(optValue,finalValuesThomp,finalValuesUCB,"/home/gugohb/Dropbox/thesis_agos/plot/thompson_vs_marc_3c.pdf")
