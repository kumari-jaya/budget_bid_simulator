

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt


class Plotter:

    def __init__(self,agent=[],env=[]):
        self.agent = agent
        self.environment = env


    def plotGP(self,gpIndex,fixedBid = False,bid=0.1):
        if (fixedBid==False):
            budgetPoints = np.linspace(0,self.agent.maxTotDailyBudget,1000)
            bidsPoints = np.linspace(0,self.agent.maxBid,1000)
            bestBids = self.agent.bestBidsPerBudgetsArray(budgetPoints,bidsPoints, gpIndex)

        else:
            budgetPoints = np.linspace(0,self.agent.maxTotDailyBudget,1000)
            bestBids = np.ones(1000)*bid

        fig = plt.figure()
        observedInput = self.agent.prevBudgets[:,gpIndex]
        observedBids = self.agent.prevBids[:,gpIndex]
        observedOutput = self.agent.prevClicks[:,gpIndex]
        idxs = np.isclose(observedBids,bid,atol=0.2*bid)
        observedInput = observedInput[idxs]
        observedOutput = observedOutput[idxs]
        x = np.array([bestBids,budgetPoints])
        x = np.atleast_2d(x).T
        x = self.agent.normalize(x)
        budgetPointsNorm = self.agent.normalizeBudgetArray(budgetPoints)
        bestBidsNorm = self.agent.normalizeBidsArray(bestBids)
        xnorm = np.array([bestBidsNorm,budgetPointsNorm])
        xnorm = np.atleast_2d(x).T
        [means,sigmas] = self.agent.gps[gpIndex].predict(x,return_std=True)
        plt.plot(observedInput, observedOutput, 'r.', markersize=10, label=u'Observations')

        plt.plot(budgetPoints, means, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([budgetPoints, budgetPoints[::-1]]),
                 np.concatenate([means - 1.9600 * sigmas,
                                 (means + 1.9600 * sigmas)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, np.max(self.agent.prevClicks[:,gpIndex])*1.5)
        plt.legend(loc='upper left')
        plt.show()

    def plotGPComparison(self, gpIndex, trueClicks, trueBudgets, fixedBid=False, bid=0.1):
        if (fixedBid == False):
            budgetPoints = np.linspace(0, agent.maxTotDailyBudget, 1000)
            bidsPoints = np.linspace(0, agent.maxBid, 1000)
            bestBids = agent.bestBidsPerBudgetsArray(budgetPoints, bidsPoints, gpIndex)

        else:
            budgetPoints = np.linspace(0, self.agent.maxTotDailyBudget, 1000)
            bestBids = np.ones(1000) * bid

        fig = plt.figure()
        observedInput = self.agent.prevBudgets[:, gpIndex]
        observedBids = self.agent.prevBids[:, gpIndex]
        observedOutput = self.agent.prevClicks[:, gpIndex]
        idxs = np.isclose(observedBids, bid, atol=0.2 * bid)
        observedInput = observedInput[idxs]
        observedOutput = observedOutput[idxs]
        x = np.array([bestBids, budgetPoints])
        x = np.atleast_2d(x).T
        x = self.agent.normalize(x)
        budgetPointsNorm = self.agent.normalizeBudgetArray(budgetPoints)
        bestBidsNorm = self.agent.normalizeBidsArray(bestBids)
        xnorm = np.array([bestBidsNorm, budgetPointsNorm])
        xnorm = np.atleast_2d(x).T
        [means, sigmas] = self.agent.gps[gpIndex].predict(x, return_std=True)
        means = self.agent.denormalizeOutput(means, gpIndex)
        sigmas = self.agent.denormalizeOutput(sigmas, gpIndex)

        trueClicks = trueClicks[:, gpIndex]
        plt.plot(observedInput, observedOutput, 'k.', markersize=10, label=u'Observations')

        plt.plot(budgetPoints, means, 'b-', label=u'Prediction')
        plt.plot(trueBudgets, trueClicks, 'r-', label=u'True values')
        plt.fill(np.concatenate([budgetPoints, budgetPoints[::-1]]),
                 np.concatenate([means - 1.9600 * sigmas,
                                 (means + 1.9600 * sigmas)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')

        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, np.max(self.agent.prevClicks[:, gpIndex]) * 1.5)
        plt.legend(loc='upper left')
        plt.show()

    def trueSample(self,bid,maxBudget):
        ncampaigns = len(self.environment.campaigns)
        budgets = np.linspace(0,maxBudget,200)

        for i,b in enumerate(budgets):
            vettBids = np.matlib.repmat(bid,1,ncampaigns).reshape(-1)
            vettBudgets = np.matlib.repmat(b,1,ncampaigns).reshape(-1)
            observations = self.environment.generateObservationsforCampaigns(vettBids,vettBudgets)
            if i == 0:
                clicks = np.array([observations[0]])
            else:
                clicks = np.append(clicks, [observations[0]],axis=0)
            if i%10 == 0:
                print "Simulation ",i," out of 200"

        fig = plt.figure()
        plt.plot(budgets,clicks , 'r-', label=u'Campaigns')
        #plt.plot(budgets,clicks[:,1] , 'b-', label=u'Campaign 2')
        plt.xlabel('Budget')
        plt.ylabel('Clicks')
        #plt.ylim(-10, np.max(self.prevClicks[:,gpIndex])*1.5)
        plt.legend(loc='upper left')
        plt.show()
        return [clicks,budgets]
