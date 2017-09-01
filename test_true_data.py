import numpy as np
import math
from Campaign import *
from Environment import *
from AuctionTrueData import *
from Core import *
from AgentFactored import *
from AgentOracle import *
from joblib import Parallel, delayed
import copy

path = '../results/'

# Auction parameter initialization
convparams = np.array([0.4, 100, 200])
## Discount probabilities
lambdas = np.array([1.0, 0.71, 0.56, 0.53, 0.49, 0.47, 0.44, 0.44, 0.43, 0.43])
## Click probabilities of the considered ad
myClickProb = np.array([0.5, 0.6, 0.6, 0.5, 0.4, 0.1, 0.4, 0.5, 0.2, 0.4])
## Number of research per day
nMeanResearch = np.array([1000.0, 1500, 1500, 1000, 1250, 4000, 1250, 2000, 4000, 1250])
sigmaResearch = 0.2
## Number of other bidders in the auction
nBidders = [5, 6, 6, 5, 5, 5, 6, 5, 6, 6]
nSlots = 5

nCampaigns = 5
campaigns = []
for c in range(0, nCampaigns):
    a = AuctionTrueData(nBidders=nBidders[c], nSlots=nSlots,
                        lambdas=lambdas, myClickProb=myClickProb[c])
    campaigns.append(Campaign(a, nMeanResearch=nMeanResearch[c], nStdResearch=sigmaResearch,
                              probClick=myClickProb[c], convParams=convparams))

# Environment setting
envi = Environment(copy.copy(campaigns))

# Experiment setting
nBids = 10
nIntervals = 10
deadline = 500
maxBudget = 100

# Baseline computation
oracle = Oracle(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns,
                nBudget=nIntervals, nBids=nBids, maxBudget=100.0, environment=copy.copy(envi))
oracle.generateBidBudgetMatrix(nSimul=100)
values = np.ones(nCampaigns) * convparams[0] # Same value for each click
oracle.updateValuesPerClick(values)
[optBud, optBid, optConv] = oracle.chooseAction()
print "policy val",oracle.bidBudgetMatrix[2, 2, -1]
print optConv
oracle.initGPs()
print "initGPs"
oracle.updateMultiGP(5)
print "updated GPS"
np.save(path + "opt", optConv)
np.save(path + "oracle", oracle)


def experiment(k):
    # Agent initialization
    np.random.seed()

    agent = AgentFactored(budgetTot=1000, deadline=deadline, nCampaigns=nCampaigns,
                          nBudget=nIntervals, nBids=nBids, maxBudget=100.0)
    agent.initGPs()
    print "Experiment : ", k
    print "A"

    # Set the GPs hyperparameters
    for c in range(0, nCampaigns):
        agent.setGPKernel(c, oracle.gpsClicks[c].kernel_ , oracle.gpsCosts[c].kernel_)

    # Init the Core and execute the experiment
    env = Environment(copy.copy(campaigns))
    core = Core(agent, copy.copy(env), deadline)

    core.runEpisode()
    np.save(path + "policy_" + str(k), [agent.prevBids, agent.prevBudgets])
    np.save(path + "experiment_" + str(k),np.sum(agent.prevConversions, axis=1))
    return np.sum(agent.prevConversions, axis=1), agent

nExperiments = 60
out = Parallel(n_jobs=-1)(
        delayed(experiment)(k) for k in xrange(nExperiments))

np.save(path+"allExperiments", out)
