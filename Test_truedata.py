#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign_TrueData import *
from Environment import *
from Auction_TrueData import *
from AgentMarcello import *
from AgentPrior import *
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
    core2D.runEpisode()
    core3D.runEpisode()
    conv2D=np.sum(agent2D.prevConversions,axis=1)
    conv3D= np.sum(agent3D.prevConversions,axis=1)
    position2d =  "/home/mmm/cartella_guglielmo/dati/truedata/valori_2d_" + str(k)
    np.save(position2d,conv2D)
    position3d =  "/home/mmm/cartella_guglielmo/dati/truedata/valori_3d_" + str(k)
    np.save(position3d,conv3D)
    return conv2D,conv3D



convparams=np.array([0.4,100,200])
lambdas1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
lambdas2 = np.array([0.9, 0.8, 0.7])
lambdas3 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])


a1= Auction_TrueData(nbidders=5 , nslots=5, lambdas=lambdas1)
a2= Auction_TrueData(nbidders=6 , nslots=5, lambdas=lambdas1)
a3= Auction_TrueData(nbidders=4 , nslots=3, lambdas=lambdas2)
a4= Auction_TrueData(nbidders=7 , nslots=7, lambdas=lambdas3)
a5= Auction_TrueData(nbidders=8 , nslots=7, lambdas=lambdas3)


campaigns=[]
campaigns.append(Campaign_TrueData(a1, nusers=1000.0 ,convParams= convparams))
campaigns.append(Campaign_TrueData(a2, nusers=1500.0 ,convParams= convparams))
campaigns.append(Campaign_TrueData(a3, nusers=1500.0 ,convParams= convparams))
campaigns.append(Campaign_TrueData(a2, nusers=1000.0 ,convParams= convparams))
campaigns.append(Campaign_TrueData(a4, nusers=1250.0 ,convParams= convparams))
#campaigns.append(Campaign_TrueData(a2, nusers=4000.0 ,convParams= convparams))
#campaigns.append(Campaign_TrueData(a1, nusers=1250.0 ,convParams= convparams))
#campaigns.append(Campaign_TrueData(a5, nusers=2000.0 ,convParams= convparams))
#campaigns.append(Campaign_TrueData(a5, nusers=4000.0 ,convParams= convparams))
#campaigns.append(Campaign_TrueData(a3, nusers=1250.0 ,convParams= convparams))

ncampaigns = len(campaigns)

env = Environment(campaigns)
nBids=10
nIntervals=10
deadline = 60
maxBudget = 100

agent = AgentPrior(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
agent.initGPs()
plotter = PlotterFinal(agent=agent,env=env)

# mi creo una lista con tutte le matrici dell'oracolo di ogni campagna
listMatrices = list()
for i in range(0,ncampaigns):
    matrix = plotter.oracleMatrix(indexCamp=i,nsimul=8)
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
np.save("/home/mmm/cartella_guglielmo/dati/truedata/valore_ottimo",optValue)

nexperiments = 50
# mi salvo le tre realizzazioni degli esperimenti e poi alla fine le medio!
results2D = np.zeros(shape=(nexperiments,deadline))
results3D = np.zeros(shape=(nexperiments,deadline))

out = Parallel(n_jobs=20)(
        delayed(experiment)(k) for k in xrange(nexperiments))

for i in range(nexperiments):
    results2D[i,:] = out[i][0]
    results3D[i,:] = out[i][1]

#np.save("/home/mmm/cartella_guglielmo/dati/2dvs3d/valore_ottimo_10c",optValue)
#np.save("/home/mmm/cartella_guglielmo/dati/matrice_valori_2D_10c",results2D)
#np.save("/home/mmm/cartella_guglielmo/dati/matrice_valori_3D_10c",results3D)
#finalValues2D = matrixValues2D.mean(axis=0)
#finalValues3D = matrixValues3D.mean(axis=0)
#plotter.performancePlotComparison(optValue,finalValues2D,finalValues3D,"/home/gugohb/Dropbox/Tesi/figures/2D_vs_3D_10c_100.pdf")
means2d = np.mean(results2D,axis=0)
means3d = np.mean(results3D,axis=0)
sigmas2d = np.sqrt(np.var(results2D,axis=0))
sigmas3d = np.sqrt(np.var(results3D,axis=0))
plotter.performancePlotComparison(optValue,means2d,sigmas2d,means3d,sigmas3d,"/home/mmm/cartella_guglielmo/figure/truedata/2D_vs_3D_truedata.pdf")
