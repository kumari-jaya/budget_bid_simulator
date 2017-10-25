#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from AgentCUCB import *
from AgentFactored import *
from Core import *
from matplotlib import pyplot as plt
from PlotterFinal import *
from joblib import Parallel, delayed
import copy

path_dati = "/home/mmm/cartella_guglielmo/dati/tesi_exp1/"

def experiment(k):
    np.random.seed()
    print "Esperimento: ", k
    agentCUCB = AgentCUCB(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget,1.0)
    agentGPUCB = AgentFactored(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget,1.0,"GPUCB")
    agentThomp = AgentFactored(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget,1.0,"Sampling")
    agentGPUCB.initGPs()
    agentThomp.initGPs()
    envCUCB = Environment(copy.copy(campaigns))
    envGPUCB = Environment(copy.copy(campaigns))
    envThomp = Environment(copy.copy(campaigns))
    coreCUCB = Core(agentCUCB, envCUCB, deadline)
    coreGPUCB = Core(agentGPUCB, envGPUCB, deadline)
    coreThomp = Core(agentThomp, envThomp, deadline)
    meanConvCUCB = np.zeros(deadline)
    meanConvGPUCB = np.zeros(deadline)
    meanConvThomp = np.zeros(deadline)

    # in questo ciclo mi salvo le conversioni medie in ogni istante
    for t in range(deadline):
        coreCUCB.step()
        coreGPUCB.step()
        coreThomp.step()
        meanConvCUCB[t] = lastMeanConv(agentCUCB)
        meanConvGPUCB[t] = lastMeanConv(agentGPUCB)
        meanConvThomp[t] = lastMeanConv(agentThomp)

    # ora invece mi salvo le conversioni istantanee
    instConvCUCB = np.sum(agentCUCB.prevConversions, axis=1)
    instconvGPUCB = np.sum(agentGPUCB.prevConversions, axis=1)
    instConvThomp = np.sum(agentThomp.prevConversions, axis=1)


    positionCUCB1 =  path_dati + "inst_conv_CUCB_" + str(k)
    positionCUCB2 =  path_dati + "mean_conv_CUCB_" + str(k)
    np.save(positionCUCB1,instConvCUCB)
    np.save(positionCUCB2,meanConvCUCB)
    positionGPUCB1 =  path_dati + "inst_conv_GPUCB_" + str(k)
    positionGPUCB2 =  path_dati + "mean_conv_GPUCB_" + str(k)
    np.save(positionGPUCB1,instConvGPUCB)
    np.save(positionGPUCB2,meanConvGPUCB)
    positionThomp1 =  path_dati + "inst_conv_Thomp_" + str(k)
    positionThomp2 =  path_dati + "mean_conv_Thomp_" + str(k)
    np.save(positionThomp1,instConvThomp)
    np.save(positionThomp2,meanConvThomp)

    return


def lastMeanConv(agent):
    lastBudgets = agent.prevBudgets[-1,:]
    lastBids = agent.prevBids[-1,:]
    chosenValue = 0
    for i in range(0,ncampaigns):
        indBud = np.argwhere(np.isclose(agent.budgets, lastBudgets[i]))
        indBid = np.argwhere(np.isclose(agent.bids, lastBids[i]))
        # il valore dei click medio relativi ai bid e budget scelti li prendo dalle matrici dell'oracolo!
        chosenValue += listMeans[i][indBud,indBid]
    return chosenValue

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
deadline = 250
maxBudget = 100

agent = AgentFactored(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget,1.0)
agent.initGPs()
env = Environment(campaigns)
plotter = PlotterFinal(agent=agent,env=env)

# mi creo una lista con tutte le matrici dell'oracolo di ogni campagna
listMeans = list()
listVar = list()
for i in range(0,ncampaigns):
    [trueMeans,trueVar] = plotter.oracleMatrix(indexCamp=i,nsimul=20)
    listMeans.append(trueMeans)
    listVar.append(trueVar)
    if i==0:
        optMatrix = np.array([trueMeans.max(axis=1)])
    else:
        maxrow = np.array([trueMeans.max(axis=1)])
        optMatrix = np.concatenate((optMatrix,maxrow))


[newBudgets,newCampaigns] = agent.optimize(optMatrix)

# ora ricerco nelle matrici originali il numero di click nell'allocazione ottima
optValue = 0
optVar = 0
for i in range(0,ncampaigns):
    print i
    index = np.argwhere(np.isclose(agent.budgets,newBudgets[i]))
    indMax = np.argmax(listMeans[i][index,:])
    optValue += listMeans[i][index,indMax]
    optVar += listVar[i][index,indMax]

ottimo = np.array([optValue,optVar])
np.save(path_dati + "ottimo",ottimo)


nexperiments = 50


out = Parallel(n_jobs=20)(
        delayed(experiment)(k) for k in xrange(nexperiments))
