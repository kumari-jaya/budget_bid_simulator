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


a1= Auction(nbidders=5 , nslots=5, mu=0.51 , sigma=0.2, lambdas=lambdas)
a2= Auction(nbidders=6 , nslots=5, mu=0.67 , sigma=0.4, lambdas=lambdas)


ncampaigns=2
c1 = Campaign(a1, nusers=1000.0 , probClick=0.5 ,convParams= convparams)
c2 = Campaign(a2, nusers=1500.0 , probClick=0.6 ,convParams= convparams)

env = Environment([c1,c2])
nBids=5
nIntervals=8
deadline = 20
maxBudget = 100
agent = Agent(1000, deadline, 2,nIntervals,nBids,maxBudget)
agent.initGPs()
plotter = Plotter(agent=agent,env=env)
## questa funzione data una campagna, mi restituisce la matrice di dimensione budgetsxbids con
## all'interno i valori veri dei click mediati su nsimul simulazioni.
matrix0 = plotter.oracleMatrix(indexCamp=0,nsimul=10)
matrix1 = plotter.oracleMatrix(indexCamp=1,nsimul=10)
# mi salvo il massimo sulle righe delle matrici e compngo la matrice per l'ottimizzazione.
optMatrix = np.array([matrix0.max(axis=1),matrix1.max(axis=1)])
[newBudgets,newCampaigns] = agent.optimize(optMatrix)
# ora ricerco nelle matrici originali il numero di click nell'allocazione ottima
indexes = np.array([np.argwhere(agent.budgets == newBudgets[0]), np.argwhere(agent.budgets == newBudgets[1])])
optValue = matrix0[indexes[0],:].max() + matrix1[indexes[1],:].max()
## questo è il valore dell'oracolo per il plot ora devo simulare i valori del thompson!
"""
core = Core(agent, env, deadline)
core.runEpisode()
"""
