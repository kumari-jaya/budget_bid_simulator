#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from AgentZoom import *
from AgentUCB import *
from AgentFactored import *
from Core import *
from matplotlib import pyplot as plt
from PlotterFinal import *
from joblib import Parallel, delayed
import copy

def experiment(k):
    np.random.seed()
    print "Esperimento: ", k
    agentDisc = AgentFactored(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget,1.0)
    agentCont  = AgentZoom(1000, deadline, ncampaigns,nIntervals,maxBudget)
    agentDisc.initGPs()
    envDisc = Environment(copy.copy(campaigns))
    envCont = Environment(copy.copy(campaigns))
    coreDisc = Core(agentDisc, envDisc, deadline)
    coreCont = Core(agentCont,envCont,deadline)
    coreDisc.runEpisode()
    coreCont.runEpisode()
    convDisc=np.sum(agentDisc.prevConversions,axis=1)
    convCont= np.sum(agentCont.prevConversions,axis=1)
    position2d =  "/home/mmm/cartella_guglielmo/dati/zooming/valori_disc_" + str(k)
    np.save(position2d,convDisc)
    position3d =  "/home/mmm/cartella_guglielmo/dati/zooming/valori_cont_" + str(k)
    np.save(position3d,convCont)
    return convDisc,convCont



convparams = np.array([0.4, 100, 200])
lambdas1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
lambdas2 = np.array([0.9, 0.8, 0.7])
lambdas3 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
probClick = np.array([0.5, 0.4, 0.44, 0.37, 0.41])

# Auction setting
a1 = Auction(nBidders=5, nSlots=5, mu=0.49, sigma=0.18, lambdas=lambdas1, myClickProb=probClick[0])
a2 = Auction(nBidders=6, nSlots=5, mu=0.33, sigma=0.22, lambdas=lambdas1, myClickProb=probClick[1])
a3 = Auction(nBidders=4, nSlots=3, mu=0.79, sigma=0.32, lambdas=lambdas2, myClickProb=probClick[2])
a4 = Auction(nBidders=7, nSlots=7, mu=0.29, sigma=0.4, lambdas=lambdas3, myClickProb=probClick[3])
a5 = Auction(nBidders=5, nSlots=3, mu=0.41, sigma=0.25, lambdas=lambdas2, myClickProb=probClick[4])


campaigns = []
campaigns.append(Campaign(a1, nMeanResearch=1000.0, nStdResearch=50.0, probClick=probClick[0], convParams=convparams))
campaigns.append(Campaign(a2, nMeanResearch=1500.0, nStdResearch=50.0, probClick=probClick[1], convParams=convparams))
campaigns.append(Campaign(a3, nMeanResearch=1500.0, nStdResearch=50.0, probClick=probClick[2], convParams=convparams))
campaigns.append(Campaign(a4, nMeanResearch=1250.0, nStdResearch=50.0, probClick=probClick[3], convParams=convparams))
campaigns.append(Campaign(a5, nMeanResearch=1450.0, nStdResearch=50.0, probClick=probClick[4], convParams=convparams))


ncampaigns = len(campaigns)

nBids=10
nIntervals=10
deadline = 100
maxBudget = 100
"""
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

nexperiments = 50
# mi salvo le tre realizzazioni degli esperimenti e poi alla fine le medio!
resultsDisc = np.zeros(shape=(nexperiments,deadline))
resultsCont = np.zeros(shape=(nexperiments,deadline))

out = Parallel(n_jobs=20)(
        delayed(experiment)(k) for k in xrange(nexperiments))

for i in range(nexperiments):
    resultsDisc[i,:] = out[i][0]
    resultsCont[i,:] = out[i][1]

meansDisc = np.mean(resultsDisc,axis=0)
meansCont = np.mean(resultsCont,axis=0)
sigmasDisc = np.sqrt(np.var(resultsDisc,axis=0))
sigmasCont = np.sqrt(np.var(resultsCont,axis=0))
plt.plot(meansDisc, 'r-', label=u'Discrete')
plt.plot(meansCont, 'b-', label=u'Continous')
plt.fill(np.concatenate([range(0,len(meansDisc)), range(0,len(meansDisc))[::-1]]),
         np.concatenate([meansDisc - 1.9600 * sigmasDisc,
                         (meansDisc + 1.9600 * sigmasDisc)[::-1]]),
         alpha=.5, fc='r', ec='None', label='95% confidence interval')
plt.fill(np.concatenate([range(0,len(meansDisc)), range(0,len(meansDisc))[::-1]]),
         np.concatenate([meansCont - 1.9600 * sigmasCont,
                         (meansCont + 1.9600 * sigmasCont)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')


plt.xlabel('Time step')
plt.ylabel('Conversions')
plt.ylim(-100, np.max(meansDisc)*2)
plt.legend(loc='upper left')
plt.savefig("/home/mmm/cartella_guglielmo/figure/zooming/test_zooming.pdf",bbox_inches='tight')
#plt.show()
