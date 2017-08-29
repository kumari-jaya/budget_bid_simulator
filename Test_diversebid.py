import numpy as np
import math
from Campaign import *
from Environment import *
from Auction import *
from AgentFactored import *
from Core import *
from Plotter import *
from matplotlib import pyplot as plt
import copy
from joblib import Parallel, delayed

# Parameters of the auction (conversions and slot discounts)
convParams = np.array([0.4, 100, 200])
lambdas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
probClick = np.array([0.5, 0.6, 0.6, 0.4])

a1 = Auction(nBidders=5, nslots=5, mu=0.59, sigma=0.2, lambdas=lambdas, myClickProb=probClick[0])
a2 = Auction(nBidders=6, nslots=5, mu=0.67, sigma=0.4, lambdas=lambdas, myClickProb=probClick[1])
a3 = Auction(nBidders=6, nslots=5, mu=0.47, sigma=0.25, lambdas=lambdas, myClickProb=probClick[2])
a4 = Auction(nBidders=5, nslots=5, mu=0.57, sigma=0.39, lambdas=lambdas, myClickProb=probClick[3])

# Campaign initialization
campaigns = []
campaigns.append(Campaign(a1, nMeanResearch=1000.0, nStdResearch=50.0, probClick=0.5, convParams=convParams))
campaigns.append(Campaign(a2, nMeanResearch=1500.0, nStdResearch=50.0, probClick=0.6, convParams=convParams))
campaigns.append(Campaign(a3, nMeanResearch=1500.0, nStdResearch=50.0, probClick=0.6, convParams=convParams))
campaigns.append(Campaign(a4, nMeanResearch=1250.0, nStdResearch=50.0, probClick=0.4, convParams=convParams))


nCampaigns = len(campaigns)
nIntervals = 10

nBids1 = 5
nBids2 = 10
nBids3 = 20
nBids4 = 50

maxBudget = 100
deadline = 60

# Initialize experiments
nExperiments = 50
results1 = np.zeros(shape=(nExperiments, deadline))
results2 = np.zeros(shape=(nExperiments, deadline))
results3 = np.zeros(shape=(nExperiments, deadline))
results4 = np.zeros(shape=(nExperiments, deadline))

savePath = "../save"


def experiment(k):
    np.random.seed()

    print "Experiment: ", k
    agent1 = AgentMarcello(1000, deadline, nCampaigns, nIntervals, nBids1, maxBudget)
    agent2 = AgentMarcello(1000, deadline, nCampaigns, nIntervals, nBids2, maxBudget)
    agent3 = AgentMarcello(1000, deadline, nCampaigns, nIntervals, nBids3, maxBudget)
    agent4 = AgentMarcello(1000, deadline, nCampaigns, nIntervals, nBids4, maxBudget)

    env1 = Environment(copy.copy(campaigns))
    env2 = Environment(copy.copy(campaigns))
    env3 = Environment(copy.copy(campaigns))
    env4 = Environment(copy.copy(campaigns))

    core1 = Core(agent1, env1, deadline)
    core2 = Core(agent2, env2, deadline)
    core3 = Core(agent3, env3, deadline)
    core4 = Core(agent4, env4, deadline)

    agent1.initGPs()
    agent2.initGPs()
    agent3.initGPs()
    agent4.initGPs()

    core1.runEpisode()
    core2.runEpisode()
    core3.runEpisode()
    core4.runEpisode()

    conv1 = np.sum(agent1.prevConversions,axis=1)
    conv2 = np.sum(agent2.prevConversions,axis=1)
    conv3 = np.sum(agent3.prevConversions,axis=1)
    conv4 = np.sum(agent4.prevConversions,axis=1)

    position5 = savePath + "/valori_5bid_" + str(k)
    np.save(position5,conv1)
    position10 = savePath + "/valori_10bid_" + str(k)
    np.save(position10,conv2)
    position20 = savePath + "/valori_20bid_" + str(k)
    np.save(position20,conv3)
    position50 = savePath + "/valori_50bid_" + str(k)
    np.save(position50, conv4)
    return [conv1, conv2, conv3, conv4]

out = Parallel(n_jobs=1)(
        delayed(experiment)(k) for k in xrange(nExperiments))

for e in range(0,nExperiments):
    results1[e,:] = out[e][0]
    results2[e,:] = out[e][1]
    results3[e,:] = out[e][2]
    results4[e,:] = out[e][3]

#np.save("/home/gugohb/Dropbox/Tesi/figures/risultati/2D_5bid",results1)
#np.save("/home/gugohb/Dropbox/Tesi/figures/risultati/2D_10bid",results2)
#np.save("/home/gugohb/Dropbox/Tesi/figures/risultati/2D_50bid",results3)

means5 = np.mean(results1, axis=0)
means10 = np.mean(results2, axis=0)
means20 = np.mean(results3, axis=0)
means50 = np.mean(results4, axis=0)

sigmas5 = np.sqrt(np.var(results1, axis=0))
sigmas10 = np.sqrt(np.var(results2, axis=0))
sigmas20 = np.sqrt(np.var(results3, axis=0))
sigmas50 = np.sqrt(np.var(results4, axis=0))


plt.plot(means5,color='r',label = u'5 bid')
plt.fill(np.concatenate([range(0,deadline), range(0,deadline)[::-1]]),
         np.concatenate([means5 - 1.9600 * sigmas5,
                         (means5 + 1.9600 * sigmas5)[::-1]]),
         alpha=.5, fc='r', ec='None', label='95% confidence interval 5bid')
plt.plot(means10,color='b',label = u'10 bid')
plt.fill(np.concatenate([range(0,deadline), range(0,deadline)[::-1]]),
         np.concatenate([means10 - 1.9600 * sigmas10,
                         (means10 + 1.9600 * sigmas10)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval 10bid')
plt.plot(means20,color='g',label = u'20 bid')
plt.fill(np.concatenate([range(0,deadline), range(0,deadline)[::-1]]),
         np.concatenate([means20 - 1.9600 * sigmas20,
                         (means20 + 1.9600 * sigmas20)[::-1]]),
         alpha=.5, fc='g', ec='None', label='95% confidence interval 20bid')
plt.plot(means50,color='k',label = u'50 bid')
plt.fill(np.concatenate([range(0,deadline), range(0,deadline)[::-1]]),
         np.concatenate([means50 - 1.9600 * sigmas50,
                         (means50 + 1.9600 * sigmas50)[::-1]]),
         alpha=.5, fc='k', ec='None', label='95% confidence interval 50bid')
plt.legend(loc = 'upper left')
plt.savefig("/home/mmm/cartella_guglielmo/figure/diversebid/test_diversebid.pdf",bbox_inches='tight')
#plt.show
