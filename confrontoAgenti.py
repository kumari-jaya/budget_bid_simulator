import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from AgentMarcello import *
from AgentPrior import *
from AgentAle import *
from Core import *
from Plotter import *
from matplotlib import pyplot as plt
import copy
from joblib import Parallel, delayed



convparams=np.array([0.4,100,200])
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])


a1= Auction(nbidders=5 , nslots=5, mu=0.59 , sigma=0.2, lambdas=lambdas)
a2= Auction(nbidders=6 , nslots=5, mu=0.67 , sigma=0.4, lambdas=lambdas)
a3= Auction(nbidders=6 , nslots=5, mu=0.47 , sigma=0.25, lambdas=lambdas)
a4= Auction(nbidders=5 , nslots=5, mu=0.57 , sigma=0.39, lambdas=lambdas)


campaigns=[]
campaigns.append(Campaign(a1, nusers=1000.0 , probClick=0.5 ,convParams= convparams))
campaigns.append(Campaign(a2, nusers=1500.0 , probClick=0.6 ,convParams= convparams))
campaigns.append(Campaign(a3, nusers=1500.0 , probClick=0.6 ,convParams= convparams))
campaigns.append(Campaign(a2, nusers=1000.0 , probClick=0.5 ,convParams= convparams))
campaigns.append(Campaign(a4, nusers=1250.0 , probClick=0.4 ,convParams= convparams))
campaigns.append(Campaign(a2, nusers=1000.0 , probClick=0.35 ,convParams= convparams))
campaigns.append(Campaign(a4, nusers=1350.0 , probClick=0.41 ,convParams= convparams))

ncampaigns = len(campaigns)
nIntervals = 15
nBids = 15
maxBudget = 100
deadline = 150

nExperiments = 80
nAlgorithms =2

results2D = np.zeros(shape=(nExperiments,deadline))
results3D = np.zeros(shape=(nExperiments,deadline))

def experiment(k):
    np.random.seed()

    print "Experiment: ",k
    agent2D = AgentMarcello(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
    agent3D  = AgentPrior(1000, deadline, ncampaigns,nIntervals,nBids,maxBudget)
    env2D = Environment()
    env3D = Environment()
    env2D = Environment(copy.copy(campaigns))
    env3D=Environment(copy.copy(campaigns))
    core2D = Core(agent2D, env2D, deadline)
    core3D = Core(agent3D,env3D,deadline)

    agent2D.initGPs()
    agent3D.initGPs()

    core2D.runEpisode()
    core3D.runEpisode()

    conv2D=np.sum(agent2D.prevConversions,axis=1)
    conv3D= np.sum(agent3D.prevConversions,axis=1)
    return [conv2D,conv3D]



out = Parallel(n_jobs=-1)(
        delayed(experiment)(k) for k in xrange(nExperiments))

for e in range(0,nExperiments):
    results2D[e,:] = out[e][0]
    results3D[e,:] = out[e][1]


fileName2D = "results2D_" + str(len(campaigns)) + "_campagne"
fileName3D = "results3D_" + str(len(campaigns)) + "_campagne"

np.save(fileName2D,results2D)
np.save(fileName3D,results3D)

"""
optimum = np.ones(deadline)*146.2

plt.figure(0)
plt.plot(np.mean(results2D,axis=0),color='r')
plt.plot(np.mean(results3D,axis=0),color = 'b')
plt.plot(optimum,color='g')
plt.legend({'2D','3D'})
"""



