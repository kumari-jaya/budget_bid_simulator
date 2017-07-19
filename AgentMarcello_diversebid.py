import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from AgentMarcello import *
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
#campaigns.append(Campaign(a2, nusers=4000.0 , probClick=0.1 ,convParams= convparams))
#campaigns.append(Campaign(a1, nusers=1250.0 , probClick=0.4 ,convParams= convparams))

ncampaigns = len(campaigns)
nIntervals = 10
nBids1 = 5
nBids2 = 10
nBids3 = 20
maxBudget = 100
deadline = 60

nExperiments = 4

results1 = np.zeros(shape=(nExperiments,deadline))
results2 = np.zeros(shape=(nExperiments,deadline))
results3 = np.zeros(shape=(nExperiments,deadline))

def experiment(k):
    np.random.seed()

    print "Experiment: ",k
    agent1 = AgentMarcello(1000, deadline, ncampaigns,nIntervals,nBids1,maxBudget)
    agent2 = AgentMarcello(1000, deadline, ncampaigns,nIntervals,nBids2,maxBudget)
    agent3 = AgentMarcello(1000, deadline, ncampaigns,nIntervals,nBids3,maxBudget)
    env1 = Environment()
    env2 = Environment()
    env3 = Environment()
    env1 = Environment(copy.copy(campaigns))
    env2 = Environment(copy.copy(campaigns))
    env3 = Environment(copy.copy(campaigns))
    core1 = Core(agent1, env1, deadline)
    core2 = Core(agent2, env2, deadline)
    core3 = Core(agent3, env3, deadline)

    agent1.initGPs()
    agent2.initGPs()
    agent3.initGPs()

    core1.runEpisode()
    core2.runEpisode()
    core3.runEpisode()

    conv1 = np.sum(agent1.prevConversions,axis=1)
    conv2 = np.sum(agent2.prevConversions,axis=1)
    conv3 = np.sum(agent3.prevConversions,axis=1)
    return [conv1,conv2,conv3]



out = Parallel(n_jobs=-1)(
        delayed(experiment)(k) for k in xrange(nExperiments))

for e in range(0,nExperiments):
    results1[e,:] = out[e][0]
    results2[e,:] = out[e][1]
    results3[e,:] = out[e][2]

np.save("/home/gugohb/Dropbox/Tesi/figures/risultati/2D_5bid",results1)
np.save("/home/gugohb/Dropbox/Tesi/figures/risultati/2D_10bid",results2)
np.save("/home/gugohb/Dropbox/Tesi/figures/risultati/2D_20bid",results3)


plt.plot(np.mean(results1,axis=0),color='r')
plt.plot(np.mean(results2,axis=0),color = 'b')
plt.plot(np.mean(results3,axis=0),color = 'g')
plt.legend({'5 bid','10 bid','20 bid'})
plt.show
