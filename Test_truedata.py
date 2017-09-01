#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import copy

from joblib import Parallel, delayed
from lost_and_found.environment.AuctionTrueData import *
from lost_and_found.environment.Core import *
from lost_and_found.environment.Environment import *

from agent.AgentFactored import *
from agent.AgentPrior import *
from graphicalTool.PlotterFinal import *


def experiment(k):
    np.random.seed()
    print "Esperimento: ", k
    agent2D = AgentFactored(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
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



convparams = np.array([0.4, 100, 200])
lambdas1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
lambdas2 = np.array([0.9, 0.8, 0.7])
lambdas3 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

probClick = np.array([0.5, 0.3, 0.4, 0.6])

a1 = AuctionTrueData(nBidders=5, nslots=5, lambdas=lambdas1, myClickProb=probClick[0])
a2 = AuctionTrueData(nBidders=6, nslots=5, lambdas=lambdas1, myClickProb=probClick[1])
a3 = AuctionTrueData(nBidders=4, nslots=3, lambdas=lambdas2, myClickProb=probClick[2])
a4 = AuctionTrueData(nBidders=7, nslots=7, lambdas=lambdas3, myClickProb=probClick[3])

campaigns = []
campaigns.append(Campaign(a1, nMeanResearch=1000.0, nStdResearch=50.0, probClick=probClick[0], convParams=convparams))
campaigns.append(Campaign(a2, nMeanResearch=1500.0, nStdResearch=50.0, probClick=probClick[1], convParams=convparams))
campaigns.append(Campaign(a3, nMeanResearch=1500.0, nStdResearch=50.0, probClick=probClick[2], convParams=convparams))
campaigns.append(Campaign(a4, nMeanResearch=1250.0, nStdResearch=50.0, probClick=probClick[3], convParams=convparams))


ncampaigns = len(campaigns)

env = Environment(campaigns)
nBids = 5
nIntervals = 5
deadline = 20
maxBudget = 100

agent = AgentPrior(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
agent.initGPs()
plotter = PlotterFinal(agent=agent, env=env)

# mi creo una lista con tutte le matrici dell'oracolo di ogni campagna
listMatrices = list()
for i in range(0, ncampaigns):
    matrix = plotter.oracleMatrix(indexCamp=i,nsimul=8)
    listMatrices.append(matrix)
    if i == 0:
        optMatrix = np.array([matrix.max(axis=1)])
    else:
        maxrow = np.array([matrix.max(axis=1)])
        optMatrix = np.concatenate((optMatrix,maxrow))


[newBudgets,newCampaigns] = agent.optimize(optMatrix)

# ora ricerco nelle matrici originali il numero di click nell'allocazione ottima
optValue = 0
for i in range(0, ncampaigns):
    print i
    index = np.argwhere(np.isclose(agent.budgets, newBudgets[i]))
    tempValue = listMatrices[i][index, :].max()
    optValue += tempValue
optValue = optValue * convparams[0]  #converto i click in conversioni
## questo è il valore dell'oracolo per il plot ora devo simulare i valori del thompson!
np.save("./sourceData/valore_ottimo", optValue)

nexperiments = 2
# mi salvo le tre realizzazioni degli esperimenti e poi alla fine le medio!
results2D = np.zeros(shape=(nexperiments,deadline))
results3D = np.zeros(shape=(nexperiments,deadline))

out = Parallel(n_jobs=1)(
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
