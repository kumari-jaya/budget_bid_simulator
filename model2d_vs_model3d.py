#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from AgentPrior import *
from AgentMarcello import *
from Core import *
from matplotlib import pyplot as plt
from PlotterFinal import *
from joblib import Parallel, delayed
import copy

def experiment(k):
    np.random.seed()
    print "Esperimento: ", k
    agent2D = AgentMarcello(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
    agent3D  = AgentPrior(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
    agent2D.initGPs()
    agent3D.initGPs()
    env2D = Environment(copy.copy(campaigns))
    env3D = Environment(copy.copy(campaigns))
    core2D = Core(agent2D, env2D, deadline)
    core3D = Core(agent3D,env3D,deadline)
    chosenValues2D = np.zeros((deadline))
    chosenValues3D = np.zeros((deadline))
    """
    for t in range(0,deadline):
        print "Day: ",t+1
        core2D.step()
        core3D.step()
        lastBudgets2D = agent2D.prevBudgets[-1,:]
        lastBids2D = agent2D.prevBids[-1,:]
        lastBudgets3D = agent3D.prevBudgets[-1,:]
        lastBids3D = agent3D.prevBids[-1,:]
        for i in range(0,ncampaigns):
            indBudT = np.argwhere(np.isclose(agent2D.budgets, lastBudgets2D[i]))
            indBidT = np.argwhere(np.isclose(agent2D.bids, lastBids2D[i]))
            chosenValues2D[t] += listMatrices[i][indBudT,indBidT] *convparams[0]
            indBudUC = np.argwhere(np.isclose(agent3D.budgets, lastBudgets3D[i]))
            indBidUC = np.argwhere(np.isclose(agent3D.bids, lastBids3D[i]))
            chosenValues3D[t] += listMatrices[i][indBudUC,indBidUC] *convparams[0]
    """
    core2D.runEpisode()
    core3D.runEpisode()
    conv2D=np.sum(agent2D.prevConversions,axis=1)
    conv3D= np.sum(agent3D.prevConversions,axis=1)
    position2d =  "/home/mmm/cartella_guglielmo/dati/2dvs3d/valori_2d_" + str(k)
    np.save(postiion2d,conv2D)
    position3d =  "/home/mmm/cartella_guglielmo/dati/2dvs3d/valori_3d_" + str(k)
    np.save(postiion3d,conv3D)
    return chosenValues2D,chosenValues3D,conv2D,conv3D



convparams=np.array([0.4,100,200])
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])


a1= Auction(nbidders=5 , nslots=5, mu=0.59 , sigma=0.2, lambdas=lambdas)
a2= Auction(nbidders=6 , nslots=5, mu=0.67 , sigma=0.4, lambdas=lambdas)
a3= Auction(nbidders=6 , nslots=5, mu=0.47 , sigma=0.25, lambdas=lambdas)
a4= Auction(nbidders=5 , nslots=5, mu=0.57 , sigma=0.39, lambdas=lambdas)
a5= Auction(nbidders=5 , nslots=5, mu=0.5 , sigma=0.15, lambdas=lambdas)


campaigns=[]
campaigns.append(Campaign(a1, nUsers=1000.0, probClick=0.5, convParams= convparams))
campaigns.append(Campaign(a2, nUsers=1500.0, probClick=0.6, convParams= convparams))
campaigns.append(Campaign(a3, nUsers=1500.0, probClick=0.6, convParams= convparams))
campaigns.append(Campaign(a2, nUsers=1000.0, probClick=0.5, convParams= convparams))
campaigns.append(Campaign(a4, nUsers=1250.0, probClick=0.4, convParams= convparams))
campaigns.append(Campaign(a2, nUsers=4000.0, probClick=0.1, convParams= convparams))
campaigns.append(Campaign(a1, nUsers=1250.0, probClick=0.4, convParams= convparams))
campaigns.append(Campaign(a5, nUsers=2000.0, probClick=0.5, convParams= convparams))
campaigns.append(Campaign(a5, nUsers=4000.0, probClick=0.2, convParams= convparams))
campaigns.append(Campaign(a3, nUsers=1250.0, probClick=0.4, convParams= convparams))

ncampaigns = len(campaigns)

env = Environment(campaigns)
nBids=10
nIntervals=10
deadline = 100
maxBudget = 100

agent = AgentPrior(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
agent.initGPs()
plotter = PlotterFinal(agent=agent,env=env)

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
np.save("/home/mmm/cartella_guglielmo/dati/2dvs3d/valore_ottimo",optValue)
"""
nexperiments = 100
# mi salvo le tre realizzazioni degli esperimenti e poi alla fine le medio!
matrixValues2D = np.zeros((nexperiments,deadline))
matrixValues3D = np.zeros((nexperiments,deadline))
results2D = np.zeros(shape=(nexperiments,deadline))
results3D = np.zeros(shape=(nexperiments,deadline))

out = Parallel(n_jobs=20)(
        delayed(experiment)(k) for k in xrange(nexperiments))

for i in range(nexperiments):
    matrixValues2D[i,:] = out[i][0]
    matrixValues3D[i,:] = out[i][1]
    results2D[i,:] = out[i][2]
    results3D[i,:] = out[i][3]

print "opt value:", optValue
#np.save("/home/mmm/cartella_guglielmo/dati/matrice_valori_2D_10c",results2D)
#np.save("/home/mmm/cartella_guglielmo/dati/matrice_valori_3D_10c",results3D)
#finalValues2D = matrixValues2D.mean(axis=0
#finalValues3D = matrixValues3D.mean(axis=0)
#plotter.performancePlotComparison(optValue,finalValues2D,finalValues3D,"/home/gugohb/Dropbox/Tesi/figures/2D_vs_3D_10c_100.pdf")
means2d = np.mean(results2D,axis=0)
means3d = np.mean(results3D,axis=0)
sigmas2d = np.sqrt(np.var(results2D,axis=0))
sigmas3d = np.sqrt(np.var(results3D,axis=0))
plotter.performancePlotComparison(optValue,means2d,sigmas2d,means3d,sigmas3d,"/home/mmm/cartella_guglielmo/figure/2dvs3d/2D_vs_3D_10c_sumconv_100.pdf")
"""
