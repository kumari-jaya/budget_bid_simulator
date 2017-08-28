import numpy as np
from numpy import genfromtxt
import scipy.stats as stats
from Auction import *

class AuctionTrueData(Auction):

    def __init__(self, nbidders, nslots, lambdas, myClickProb):
        self.nbidders = nbidders
        self.nslots = nslots
        self.lambdas = lambdas
        self.pClick = myClickProb
        if(nbidders < nslots):
            print "nBidders should be >= than nslots"

    def simulateAuction(self, mybid):
        """
        Simulate an auction with other auctioneer simulated from Yahoo! A3 dataset
        :param mybid: the bid we set for the auction
        :return: cpc: cost per click
        :return: mypos: position in the auction
        :return: pObsSlot: continuation probability of the slot
        """
        [bids, pClicks] = self.sampleFromRealData()

        expValue = np.maximum(bids * pClicks, 0)
        expValue[::-1].sort() #decreasing order
        indexes = np.argwhere(mybid * self.pClick > expValue).reshape(-1)
        if len(indexes) > 0:
            cpc = bids[indexes[0]]
            mypos = indexes[0] # slot ordered from 0 to n-1
        else:
            cpc = 0
            mypos = len(bids)  # last position

        if mypos < self.nslots:
            pObsSlot = self.lambdas[mypos]
        else:
            pObsSlot = 0

        return [cpc, mypos, pObsSlot]

    def sampleFromRealData(self):
        """

        :return: values: bids of the other partecipants in the auction
        :return: qualities: click probabilities of the other partecipants in the auction
        """
        ## conversione in python della funzione ssa_generator.m
        truebids = genfromtxt('./data/BidData.csv', delimiter=',')
        values = np.zeros(self.nbidders)
        qualities = np.zeros(self.nbidders)

        for i in range(0, self.nbidders):
            index = np.random.randint(0, 100)
            minBid = truebids[index, 0]
            maxBid = truebids[index, 1]
            meanBid = truebids[index, 2]
            varBid = truebids[index, 3]
            values[i] = stats.truncnorm.rvs((minBid - meanBid) / np.sqrt(varBid), (maxBid - meanBid) / np.sqrt(varBid),
                                            loc=meanBid, scale=np.sqrt(varBid), size=1)

            betaA = truebids[index,4]
            betaB = truebids[index,5]
            qualities[i] = np.random.beta(betaA, betaB)

        #normalization between 5 cents and 5 euros
        values = 0.05 + (values - values.min()) / (values.max() - values.min()) * (5 - 0.05)

        return [values, qualities]
